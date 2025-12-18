"""
Interpretability analysis for Liora VAE with attention and ODE components.

Provides comprehensive attribution analysis:
1. Attention pathway: Genes → Tokens → Latents → Outputs
2. Encoder pathway: Genes → Latents (discriminative markers)
3. Decoder pathway: Latents → Genes (reconstructive programs)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Set
from scipy.stats import spearmanr
import warnings


class LioraInterpretability:
    """
    Complete interpretability analysis for Liora models.
    
    Handles both MLP and attention-based encoders, with special
    support for trajectory analysis.
    
    Key Methods
    -----------
    - compute_encoder_combined_score_matrix: Genes → Latents (discriminative)
    - compute_decoder_combined_score_matrix: Latents → Genes (reconstructive)
    - compute_both_pathways_combined_scores: Complete bidirectional analysis
    - plot_encoder_decoder_comparison: Unified visualization
    
    Parameters
    ----------
    model : LioraModel
        Trained Liora model instance
    gene_names : list, optional
        Names of input genes/features
    latent_names : list, optional
        Names for latent dimensions
    """
    
    def __init__(
        self,
        model,  # LioraModel instance
        gene_names: Optional[List[str]] = None,
        latent_names: Optional[List[str]] = None
    ):
        self.model = model
        self.vae = model.nn
        self.device = model.device
        
        # Infer dimensions
        self.n_genes = None
        self.n_latent = None
        self.n_bottleneck = None
        
        # Try to infer from encoder
        if hasattr(self.vae.encoder, 'fc1'):
            # MLP encoder
            self.n_genes = self.vae.encoder.fc1.in_features
            self.n_latent = self.vae.encoder.fc3.out_features // 2
        elif hasattr(self.vae.encoder, 'input_proj'):
            # Attention encoder
            self.n_genes = self.vae.encoder.input_proj.in_features
            self.n_latent = self.vae.encoder.attn_pool_fc.out_features // 2
        
        # Bottleneck dimension
        if hasattr(self.vae, 'latent_encoder'):
            self.n_bottleneck = self.vae.latent_encoder.out_features
        
        # Gene and latent names
        if gene_names is None:
            gene_names = [f"Gene_{i}" for i in range(self.n_genes)]
        if latent_names is None:
            latent_names = [f"Latent_{i}" for i in range(self.n_latent)]
        
        self.gene_names = gene_names
        self.latent_names = latent_names
        
        # Check encoder type
        self.is_attention = hasattr(self.vae.encoder, 'transformer')
        self.is_ode = self.vae.use_ode
        
        if self.is_attention:
            self.n_tokens = self.vae.encoder.attn_seq_len
            self.token_dim = self.vae.encoder.attn_embed_dim
    
    # ========================================================================
    # CORE GRADIENT COMPUTATION (LOW-LEVEL)
    # ========================================================================
    
    def compute_encoder_gene_to_latent(
        self,
        x: torch.Tensor,
        latent_idx: Optional[int] = None,
        use_mu: bool = True
    ) -> torch.Tensor:
        """
        Gradient-based encoder interpretability: Which genes affect which latents?
        
        For MLP encoder: Direct gradient computation.
        For Attention encoder: Aggregated through tokens.
        
        IMPORTANT: Gradients from autograd preserve sign information through chain rule.
        We return SIGNED gradients (not absolute value).
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
        latent_idx : int, optional
            Specific latent dimension (None = all dimensions)
        use_mu : bool
            Use mean q_m (True) or sampled q_z (False)
        
        Returns
        -------
        relevance : torch.Tensor
            - If latent_idx is None: shape (n_genes, n_latent)
            - Otherwise: shape (n_genes,)
            SIGNED gradients (preserves direction information)
        """
        if latent_idx is None:
            # Compute for all latent dimensions
            relevance_matrix = torch.zeros(self.n_genes, self.n_latent, device=self.device)
            for ld in range(self.n_latent):
                relevance_matrix[:, ld] = self._compute_single_latent_relevance(
                    x, ld, use_mu
                )
            return relevance_matrix
        else:
            return self._compute_single_latent_relevance(x, latent_idx, use_mu)
    
    def _compute_single_latent_relevance(
        self,
        x: torch.Tensor,
        latent_idx: int,
        use_mu: bool
    ) -> torch.Tensor:
        """Compute gene relevance for a single latent dimension.
        
        Returns SIGNED gradient (not absolute).
        """
        x = x.clone().detach().requires_grad_(True)
        
        # Forward through encoder
        if self.is_ode:
            q_z, q_m, q_s, n, t = self.vae.encoder(x)
        else:
            q_z, q_m, q_s, n = self.vae.encoder(x)
        
        # Select target
        if use_mu:
            target = q_m[:, latent_idx].sum()
        else:
            target = q_z[:, latent_idx].sum()
        
        # Backward
        target.backward()
        
        # Gene relevance: SIGNED gradient magnitude (averaged across batch)
        # Gradient already contains direction information from chain rule
        relevance = x.grad.mean(dim=0)  # No abs()!
        
        return relevance
    
    def compute_decoder_latent_to_gene(
        self,
        z: torch.Tensor,
        gene_idx: Optional[int] = None,
        use_mu: bool = True
    ) -> torch.Tensor:
        """
        Gradient-based decoder interpretability: Which latents affect which genes?
        
        NOTE: Uses UNSCALED decoder output (raw softmax).
        Library size scaling is NOT applied here.
        
        IMPORTANT: Gradients from autograd preserve sign information through chain rule.
        We return SIGNED gradients (not absolute value).
        
        Parameters
        ----------
        z : torch.Tensor, shape (batch, n_latent)
            Latent codes
        gene_idx : int, optional
            Specific gene index (None = all genes)
        use_mu : bool
            Use mean output (True) or sampled (False)
        
        Returns
        -------
        relevance : torch.Tensor
            - If gene_idx is None: shape (n_latent, n_genes)
            - Otherwise: shape (n_latent,)
            SIGNED gradients (preserves direction information)
        """
        if gene_idx is None:
            relevance_matrix = torch.zeros(self.n_latent, self.n_genes, device=self.device)
            for gidx in range(self.n_genes):
                relevance_matrix[:, gidx] = self._compute_single_gene_relevance(
                    z, gidx, use_mu
                )
            return relevance_matrix
        else:
            return self._compute_single_gene_relevance(z, gene_idx, use_mu)
    
    def _compute_single_gene_relevance(
        self,
        z: torch.Tensor,
        gene_idx: int,
        use_mu: bool
    ) -> torch.Tensor:
        """Compute latent relevance for a single output gene.
        
        Returns SIGNED gradient (not absolute).
        """
        z = z.clone().detach().requires_grad_(True)
        
        # Forward through decoder (UNSCALED)
        recon_mu, recon_logvar = self.vae.decoder(z)
        
        if use_mu:
            target = recon_mu[:, gene_idx].sum()
        else:
            std = torch.exp(0.5 * recon_logvar)
            eps = torch.randn_like(std)
            recon = recon_mu + eps * std
            target = recon[:, gene_idx].sum()
        
        target.backward()
        
        # Latent relevance: SIGNED gradient (averaged across batch)
        # Gradient already contains direction information from chain rule
        relevance = z.grad.mean(dim=0)  # No abs()!
        
        return relevance
    
    # ========================================================================
    # ATTENTION INTERPRETABILITY (IF APPLICABLE)
    # ========================================================================
    
    def compute_attention_token_attribution(
        self,
        x: torch.Tensor,
        return_token_embeddings: bool = False,
        perturbation_scale: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Perturbation-based attribution for attention encoder.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
        return_token_embeddings : bool
            Whether to return token embeddings
        perturbation_scale : float
            0 = zero out, >0 = add noise
        
        Returns
        -------
        results : dict
            - 'gene_to_token': (n_genes, n_tokens)
            - 'token_to_latent': (n_tokens, n_latent)
            - 'token_embeddings': (batch, n_tokens, token_dim) [optional]
        """
        if not self.is_attention:
            raise ValueError("This method requires attention-based encoder")
        
        # === Step 1: Gene → Token (gradient-based, SIGNED) ===
        gene_to_token = torch.zeros(self.n_genes, self.n_tokens, device=self.device)
        
        for token_idx in range(self.n_tokens):
            x_grad = x.clone().detach().requires_grad_(True)
            
            proj = self.vae.encoder.input_proj(x_grad)
            bsz = proj.size(0)
            seq = proj.view(bsz, self.n_tokens, self.token_dim)
            
            target = seq[:, token_idx, :].sum()
            target.backward()
            
            # Use absolute for token attribution (meaningful for attention flow)
            gene_to_token[:, token_idx] = torch.abs(x_grad.grad).mean(dim=0)
        
        # === Step 2: Token → Latent (perturbation-based, UNSIGNED) ===
        token_to_latent = torch.zeros(self.n_tokens, self.n_latent, device=self.device)
        
        with torch.no_grad():
            # Baseline
            if self.is_ode:
                _, q_m_baseline, _, _, _ = self.vae.encoder(x)
            else:
                _, q_m_baseline, _, _ = self.vae.encoder(x)
            q_m_baseline = q_m_baseline.mean(dim=0)
            
            # Perturb each token
            for token_idx in range(self.n_tokens):
                proj = self.vae.encoder.input_proj(x)
                bsz = proj.size(0)
                seq = proj.view(bsz, self.n_tokens, self.token_dim)
                
                # Perturb
                seq_perturbed = seq.clone()
                if perturbation_scale == 0.0:
                    seq_perturbed[:, token_idx, :] = 0.0
                else:
                    seq_perturbed[:, token_idx, :] += torch.randn_like(
                        seq[:, token_idx, :]
                    ) * perturbation_scale
                
                # Forward
                seq_perturbed = seq_perturbed.transpose(0, 1)
                seq_out = self.vae.encoder.transformer(seq_perturbed)
                seq_out = seq_out.transpose(0, 1)
                
                if self.vae.encoder.use_layer_norm:
                    seq_out = self.vae.encoder.attn_ln(seq_out)
                
                pooled = seq_out.mean(dim=1)
                output = self.vae.encoder.attn_pool_fc(pooled)
                q_m_perturbed, _ = torch.chunk(output, 2, dim=-1)
                q_m_perturbed = q_m_perturbed.mean(dim=0)
                
                # Measure impact (unsigned)
                impact = torch.abs(q_m_baseline - q_m_perturbed)
                token_to_latent[token_idx, :] = impact
        
        results = {
            'gene_to_token': gene_to_token,
            'token_to_latent': token_to_latent,
        }
        
        if return_token_embeddings:
            with torch.no_grad():
                proj = self.vae.encoder.input_proj(x)
                bsz = proj.size(0)
                seq = proj.view(bsz, self.n_tokens, self.token_dim)
                seq = seq.transpose(0, 1)
                seq_out = self.vae.encoder.transformer(seq)
                seq_out = seq_out.transpose(0, 1)
                if self.vae.encoder.use_layer_norm:
                    seq_out = self.vae.encoder.attn_ln(seq_out)
                results['token_embeddings'] = seq_out
        
        return results
    
    def compute_attention_end_to_end(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        End-to-end attention pathway: Genes → (Tokens) → Latents.
        
        Returns
        -------
        gene_to_latent : torch.Tensor, shape (n_genes, n_latent)
            Direct gene-to-latent attribution through attention pathway
        """
        if not self.is_attention:
            raise ValueError("This method requires attention-based encoder")
        
        results = self.compute_attention_token_attribution(x)
        
        # Matrix multiplication: (n_genes, n_tokens) @ (n_tokens, n_latent)
        gene_to_latent = results['gene_to_token'] @ results['token_to_latent']
        
        return gene_to_latent
    
    # ========================================================================
    # COMBINED SCORE MATRICES (HIGH-LEVEL)
    # ========================================================================
    
    def compute_encoder_combined_score_matrix(
        self,
        x: torch.Tensor,
        use_latent_mean: bool = True,
        correlation_method: str = 'pearson',
        corr_weight: float = 0.6,
        grad_weight: float = 0.4
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined score matrix for encoder: Genes → Latents.
        
        Combines signed correlation with SIGNED gradient magnitude.
        No sign-matching needed: gradient already has direction from chain rule.
        
        BIOLOGICAL INTERPRETATION:
        - Encoder emphasizes DISCRIMINATIVE genes (high information content)
        - May identify rare/variable markers rather than highly expressed genes
        - No softmax competition → different from decoder behavior
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
        use_latent_mean : bool
            If True, use q_m (deterministic mean) for both correlation and gradient
            If False, use q_z (sampled latent) for both
        correlation_method : str
            'pearson' or 'spearman'
        corr_weight : float
            Weight for correlation component (default: 0.6)
        grad_weight : float
            Weight for gradient component (default: 0.4)
        
        Returns
        -------
        results : dict
            - 'correlation_matrix': (n_genes, n_latent) SIGNED correlation
            - 'gradient_matrix': (n_genes, n_latent) SIGNED gradient (no normalization)
            - 'gradient_matrix_normalized': (n_genes, n_latent) normalized to [-1, 1]
            - 'combined_matrix': (n_genes, n_latent) weighted combination
            
        Note: All matrices contain signed values. No need for sign-matching.
        """
        # === 1. Get latent representation ===
        with torch.no_grad():
            if self.is_ode:
                q_z, q_m, q_s, n, t = self.vae.encoder(x)
            else:
                q_z, q_m, q_s, n = self.vae.encoder(x)
            
            # Choose latent representation
            latent = q_m if use_latent_mean else q_z
            latent_np = latent.cpu().numpy()
            genes_np = x.cpu().numpy()
        
        # === 2. Compute SIGNED correlation ===
        combined = np.concatenate([genes_np, latent_np], axis=1)
        
        if correlation_method == 'pearson':
            corr_matrix_full = np.corrcoef(combined.T)
        else:  # spearman
            corr_matrix_full, _ = spearmanr(combined, axis=0)
        
        # Extract gene-latent block: (n_genes, n_latent)
        corr_matrix = corr_matrix_full[:self.n_genes, self.n_genes:]
        # Keep sign → negative correlations are meaningful
        
        # === 3. Compute SIGNED gradient (no abs!) ===
        grad_matrix = self.compute_encoder_gene_to_latent(
            x, latent_idx=None, use_mu=use_latent_mean
        )  # (n_genes, n_latent), SIGNED
        grad_matrix_np = grad_matrix.cpu().numpy()
        
        # === 4. Normalize gradient to [-1, 1] while preserving sign ===
        grad_abs_max = np.abs(grad_matrix_np).max()
        
        if grad_abs_max > 1e-10:  # Avoid division by zero
            grad_matrix_normalized = grad_matrix_np / grad_abs_max
        else:
            grad_matrix_normalized = np.zeros_like(grad_matrix_np)
        
        # === 5. Combine scores (both are signed, no sign-matching needed) ===
        combined_matrix = corr_weight * corr_matrix + grad_weight * grad_matrix_normalized
        
        return {
            'correlation_matrix': torch.from_numpy(corr_matrix).float(),
            'gradient_matrix': torch.from_numpy(grad_matrix_np).float(),
            'gradient_matrix_normalized': torch.from_numpy(grad_matrix_normalized).float(),
            'combined_matrix': torch.from_numpy(combined_matrix).float()
        }
    
    def compute_decoder_combined_score_matrix(
        self,
        x: torch.Tensor,
        use_latent_mean: bool = True,
        correlation_method: str = 'pearson',
        corr_weight: float = 0.6,
        grad_weight: float = 0.4
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined score matrix for decoder: Latents → Genes.
        
        Combines signed correlation with SIGNED gradient magnitude.
        No sign-matching needed: gradient already has direction from chain rule.
        
        BIOLOGICAL INTERPRETATION:
        - Decoder emphasizes HIGHLY EXPRESSED genes (softmax competition)
        - Identifies dominant functional programs / core cellular processes
        - Better for GO enrichment of major biological themes
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
        use_latent_mean : bool
            If True, use q_m (deterministic mean)
            If False, use q_z (sampled latent)
        correlation_method : str
            'pearson' or 'spearman'
        corr_weight : float
            Weight for correlation component (default: 0.6)
        grad_weight : float
            Weight for gradient component (default: 0.4)
        
        Returns
        -------
        results : dict
            - 'correlation_matrix': (n_genes, n_latent) SIGNED correlation
            - 'gradient_matrix': (n_genes, n_latent) SIGNED gradient (no normalization)
            - 'gradient_matrix_normalized': (n_genes, n_latent) normalized to [-1, 1]
            - 'combined_matrix': (n_genes, n_latent) weighted combination
            
        Note: Decoder uses UNSCALED output (raw softmax, no library size).
               All matrices contain signed values.
        """
        # === 1. Get latent and decode ===
        with torch.no_grad():
            if self.is_ode:
                q_z, q_m, q_s, n, t = self.vae.encoder(x)
            else:
                q_z, q_m, q_s, n = self.vae.encoder(x)
            
            # Choose latent
            latent = q_m if use_latent_mean else q_z
            
            # Decode (UNSCALED)
            recon_mu, recon_logvar = self.vae.decoder(latent)
            
            latent_np = latent.cpu().numpy()
            recon_genes_np = recon_mu.cpu().numpy()
        
        # === 2. Compute SIGNED correlation: recon_genes vs latents ===
        combined = np.concatenate([recon_genes_np, latent_np], axis=1)
        
        if correlation_method == 'pearson':
            corr_matrix_full = np.corrcoef(combined.T)
        else:  # spearman
            corr_matrix_full, _ = spearmanr(combined, axis=0)
        
        # Extract gene-latent block: (n_genes, n_latent)
        corr_matrix = corr_matrix_full[:self.n_genes, self.n_genes:]
        # Keep sign
        
        # === 3. Compute SIGNED gradient (no abs!) ===
        grad_matrix = self.compute_decoder_latent_to_gene(
            latent, gene_idx=None, use_mu=True
        )  # Returns (n_latent, n_genes), SIGNED
        
        # Transpose to (n_genes, n_latent)
        grad_matrix = grad_matrix.T
        grad_matrix_np = grad_matrix.cpu().numpy()
        
        # === 4. Normalize gradient to [-1, 1] while preserving sign ===
        grad_abs_max = np.abs(grad_matrix_np).max()
        
        if grad_abs_max > 1e-10:  # Avoid division by zero
            grad_matrix_normalized = grad_matrix_np / grad_abs_max
        else:
            grad_matrix_normalized = np.zeros_like(grad_matrix_np)
        
        # === 5. Combine scores (both are signed, no sign-matching needed) ===
        combined_matrix = corr_weight * corr_matrix + grad_weight * grad_matrix_normalized
        
        return {
            'correlation_matrix': torch.from_numpy(corr_matrix).float(),
            'gradient_matrix': torch.from_numpy(grad_matrix_np).float(),
            'gradient_matrix_normalized': torch.from_numpy(grad_matrix_normalized).float(),
            'combined_matrix': torch.from_numpy(combined_matrix).float()
        }
    
    # ========================================================================
    # TOP GENE EXTRACTION (SIMPLIFIED - ALWAYS BY SIGNED SCORE)
    # ========================================================================

    def get_top_genes_per_latent_from_matrix(
        self,
        score_matrix: torch.Tensor,
        top_k: int = 50,
        return_unique: bool = False
    ) -> Dict[int, pd.DataFrame]:
        """
        Extract top genes for each latent from a score matrix.
        
        ALWAYS sorts by SIGNED score (positive scores ranked highest).
        Direction is preserved throughout the analysis.
        
        Parameters
        ----------
        score_matrix : torch.Tensor, shape (n_genes, n_latent)
            Combined score matrix (from encoder or decoder)
        top_k : int
            Number of top genes per latent
        return_unique : bool
            If True, each gene appears only once (in highest-scoring latent)
            If False, genes can appear in multiple latents
        
        Returns
        -------
        results : dict
            Key: latent_idx, Value: DataFrame with columns [gene, score, abs_score, rank]
        """
        score_matrix_np = score_matrix.cpu().numpy()
        results = {}
        
        if return_unique:
            # Each gene assigned to its best latent only
            assigned_genes = set()
            
            # Build all (gene, latent) pairs
            all_pairs = []
            for gene_idx in range(self.n_genes):
                for latent_idx in range(self.n_latent):
                    score = score_matrix_np[gene_idx, latent_idx]
                    abs_score = abs(score)
                    
                    all_pairs.append({
                        'gene_idx': gene_idx,
                        'latent_idx': latent_idx,
                        'score': score,
                        'abs_score': abs_score
                    })
            
            all_pairs_df = pd.DataFrame(all_pairs)
            
            # Sort by SIGNED score (positive first)
            all_pairs_df = all_pairs_df.sort_values('score', ascending=False)
            
            # Assign top-k unique genes per latent
            latent_counts = {i: 0 for i in range(self.n_latent)}
            
            for _, row in all_pairs_df.iterrows():
                gene_idx = int(row['gene_idx'])
                latent_idx = int(row['latent_idx'])
                score = row['score']
                
                if gene_idx in assigned_genes:
                    continue
                
                if latent_counts[latent_idx] >= top_k:
                    continue
                
                if latent_idx not in results:
                    results[latent_idx] = []
                
                results[latent_idx].append({
                    'gene': self.gene_names[gene_idx],
                    'score': score,
                    'abs_score': abs(score),
                    'rank': latent_counts[latent_idx] + 1
                })
                
                assigned_genes.add(gene_idx)
                latent_counts[latent_idx] += 1
                
                # Stop if all latents have top_k genes
                if all(count >= top_k for count in latent_counts.values()):
                    break
            
            # Convert to DataFrames
            for latent_idx in results:
                results[latent_idx] = pd.DataFrame(results[latent_idx])
        
        else:
            # Genes can appear in multiple latents (non-unique)
            for latent_idx in range(self.n_latent):
                scores = score_matrix_np[:, latent_idx]
                abs_scores = np.abs(scores)
                
                # Sort by SIGNED score (positive first)
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                gene_list = []
                for rank, gene_idx in enumerate(top_indices):
                    gene_list.append({
                        'gene': self.gene_names[gene_idx],
                        'score': scores[gene_idx],
                        'abs_score': abs_scores[gene_idx],
                        'rank': rank + 1
                    })
                
                results[latent_idx] = pd.DataFrame(gene_list)
        
        return results
    
    # ========================================================================
    # COMPLETE WORKFLOW
    # ========================================================================
    
    def compute_both_pathways_combined_scores(
        self,
        x: torch.Tensor,
        use_latent_mean: bool = True,
        correlation_method: str = 'pearson',
        encoder_corr_weight: float = 0.6,
        encoder_grad_weight: float = 0.4,
        decoder_corr_weight: float = 0.6,
        decoder_grad_weight: float = 0.4,
        top_k: int = 50,
        return_unique: bool = False
    ) -> Dict[str, Union[torch.Tensor, Dict[int, pd.DataFrame]]]:
        """
        Compute combined scores for BOTH encoder and decoder pathways.
        
        This is the main entry point for comprehensive interpretability analysis.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
        use_latent_mean : bool
            If True, use q_m (deterministic) for both pathways
            If False, use q_z (sampled) for both pathways
        correlation_method : str
            'pearson' or 'spearman'
        encoder_corr_weight : float
            Weight for encoder correlation
        encoder_grad_weight : float
            Weight for encoder gradient
        decoder_corr_weight : float
            Weight for decoder correlation
        decoder_grad_weight : float
            Weight for decoder gradient
        top_k : int
            Number of top genes per latent
        return_unique : bool
            Whether to enforce unique gene assignment
        
        Returns
        -------
        results : dict
            Complete results including matrices and top genes for both pathways
        """
        # === Encoder Pathway ===
        print("Computing encoder combined scores...")
        encoder_results = self.compute_encoder_combined_score_matrix(
            x,
            use_latent_mean=use_latent_mean,
            correlation_method=correlation_method,
            corr_weight=encoder_corr_weight,
            grad_weight=encoder_grad_weight
        )
        
        encoder_top_genes = self.get_top_genes_per_latent_from_matrix(
            encoder_results['combined_matrix'],
            top_k=top_k,
            return_unique=return_unique
        )
        
        # === Decoder Pathway ===
        print("Computing decoder combined scores...")
        decoder_results = self.compute_decoder_combined_score_matrix(
            x,
            use_latent_mean=use_latent_mean,
            correlation_method=correlation_method,
            corr_weight=decoder_corr_weight,
            grad_weight=decoder_grad_weight
        )
        
        decoder_top_genes = self.get_top_genes_per_latent_from_matrix(
            decoder_results['combined_matrix'],
            top_k=top_k,
            return_unique=return_unique
        )
        
        return {
            # Encoder results
            'encoder_correlation_matrix': encoder_results['correlation_matrix'],
            'encoder_gradient_matrix': encoder_results['gradient_matrix'],
            'encoder_gradient_matrix_normalized': encoder_results['gradient_matrix_normalized'],
            'encoder_combined_matrix': encoder_results['combined_matrix'],
            'encoder_top_genes': encoder_top_genes,
            
            # Decoder results
            'decoder_correlation_matrix': decoder_results['correlation_matrix'],
            'decoder_gradient_matrix': decoder_results['gradient_matrix'],
            'decoder_gradient_matrix_normalized': decoder_results['gradient_matrix_normalized'],
            'decoder_combined_matrix': decoder_results['combined_matrix'],
            'decoder_top_genes': decoder_top_genes,
        }
    
    # ========================================================================
    # STREAMLINED CONVENIENCE METHODS
    # ========================================================================
    
    def compare_encoder_decoder_overlap(
        self,
        res: Dict
    ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        Compare gene overlap between encoder and decoder pathways with detailed scores.
        
        Parameters
        ----------
        res : dict
            Output from compute_both_pathways_combined_scores()
        
        Returns
        -------
        overlap_summary_df : pd.DataFrame
            Summary statistics per latent:
            Columns: latent, n_encoder, n_decoder, n_overlap, jaccard
        
        overlapped_genes_dict : Dict[int, pd.DataFrame]
            For each latent_idx, DataFrame of overlapped genes with columns:
            - gene: Gene name
            - encoder_score: Signed score from encoder
            - encoder_abs_score: Absolute score from encoder
            - encoder_rank: Rank in encoder top genes
            - decoder_score: Signed score from decoder
            - decoder_abs_score: Absolute score from decoder
            - decoder_rank: Rank in decoder top genes
            - score_diff: |encoder_score - decoder_score| (magnitude difference)
            - score_agreement: cos(encoder_score, decoder_score) (directional agreement)
        """
        encoder_top_genes = res['encoder_top_genes']
        decoder_top_genes = res['decoder_top_genes']
        
        overlap_summary = []
        overlapped_genes_dict = {}
        
        for latent_idx in range(self.n_latent):
            # Get encoder and decoder dataframes
            enc_df = encoder_top_genes[latent_idx]
            dec_df = decoder_top_genes[latent_idx]
            
            # Create gene->score mappings for fast lookup
            enc_scores = {}
            for _, row in enc_df.iterrows():
                enc_scores[row['gene']] = {
                    'score': row['score'],
                    'abs_score': row['abs_score'],
                    'rank': int(row['rank'])
                }
            
            dec_scores = {}
            for _, row in dec_df.iterrows():
                dec_scores[row['gene']] = {
                    'score': row['score'],
                    'abs_score': row['abs_score'],
                    'rank': int(row['rank'])
                }
            
            # Find overlapped genes
            enc_genes = set(enc_df['gene'])
            dec_genes = set(dec_df['gene'])
            overlap = enc_genes & dec_genes
            union = enc_genes | dec_genes
            
            # Create detailed DataFrame for overlapped genes
            overlap_details = []
            for gene in sorted(overlap):
                enc_info = enc_scores[gene]
                dec_info = dec_scores[gene]
                
                # Calculate score difference and agreement
                score_diff = abs(enc_info['score'] - dec_info['score'])
                
                # Score agreement: cosine similarity of signed scores
                # Normalized to [-1, 1]: 1 = perfect agreement, -1 = opposite, 0 = orthogonal
                norm_enc = enc_info['score'] / (enc_info['abs_score'] + 1e-10)
                norm_dec = dec_info['score'] / (dec_info['abs_score'] + 1e-10)
                score_agreement = norm_enc * norm_dec  # Simple dot product on [-1, 1] space
                
                overlap_details.append({
                    'gene': gene,
                    'encoder_score': enc_info['score'],
                    'encoder_abs_score': enc_info['abs_score'],
                    'encoder_rank': enc_info['rank'],
                    'decoder_score': dec_info['score'],
                    'decoder_abs_score': dec_info['abs_score'],
                    'decoder_rank': dec_info['rank'],
                    'score_diff': score_diff,
                    'score_agreement': score_agreement
                })
            
            # Sort by score agreement (most consistent first)
            overlap_details_df = pd.DataFrame(overlap_details).sort_values(
                'score_agreement', ascending=False, ignore_index=True
            )
            
            # Add to results
            jaccard = len(overlap) / len(union) if len(union) > 0 else 0.0
            
            overlap_summary.append({
                'latent': latent_idx,
                'n_encoder': len(enc_genes),
                'n_decoder': len(dec_genes),
                'n_overlap': len(overlap),
                'jaccard': jaccard
            })
            
            overlapped_genes_dict[latent_idx] = overlap_details_df
        
        overlap_summary_df = pd.DataFrame(overlap_summary)
        
        return overlap_summary_df, overlapped_genes_dict
    
    def get_latent_order_by_overlap(
        self,
        res: Dict,
        order_by: str = 'count'
    ) -> List[int]:
        """
        Get latent ordering based on overlap statistics.
        
        Parameters
        ----------
        res : dict
            Output from compute_both_pathways_combined_scores()
        order_by : str
            'count': Order by number of overlapped genes (descending)
            'jaccard': Order by Jaccard index (descending)
            'default': Original order [0, 1, 2, ...]
        
        Returns
        -------
        latent_order : list
            Ordered list of latent indices
        """
        if order_by == 'default':
            return list(range(self.n_latent))
        
        overlap_df, _ = self.compare_encoder_decoder_overlap(res)
        
        if order_by == 'count':
            ordered = overlap_df.sort_values('n_overlap', ascending=False)['latent'].tolist()
        elif order_by == 'jaccard':
            ordered = overlap_df.sort_values('jaccard', ascending=False)['latent'].tolist()
        else:
            raise ValueError(f"Unknown order_by: {order_by}. Use 'count', 'jaccard', or 'default'")
        
        return ordered
    
    def export_top_genes_for_enrichment(
        self,
        res: Dict,
        output_dir: str,
        pathway: str = 'both'
    ):
        """
        Export top genes to text files for GO enrichment analysis.
        
        Parameters
        ----------
        res : dict
            Output from compute_both_pathways_combined_scores()
        output_dir : str
            Directory to save gene lists
        pathway : str, default='both'
            'encoder', 'decoder', or 'both'
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        pathways_to_export = []
        if pathway in ['encoder', 'both']:
            pathways_to_export.append(('encoder', res['encoder_top_genes']))
        if pathway in ['decoder', 'both']:
            pathways_to_export.append(('decoder', res['decoder_top_genes']))
        
        for pathway_name, top_genes_dict in pathways_to_export:
            for latent_idx, genes_df in top_genes_dict.items():
                filename = os.path.join(output_dir, f"{pathway_name}_latent_{latent_idx}.txt")
                genes_df['gene'].to_csv(filename, index=False, header=False)
                print(f"Saved {len(genes_df)} genes to {filename}")
    
    def get_latent_summary(
        self,
        res: Dict,
        latent_idx: int,
        n_genes: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Get summary of top genes for a specific latent.
        
        Parameters
        ----------
        res : dict
            Output from compute_both_pathways_combined_scores()
        latent_idx : int
            Latent dimension index
        n_genes : int
            Number of top genes to display
        
        Returns
        -------
        summary : dict
            'encoder_genes': DataFrame of top encoder genes
            'decoder_genes': DataFrame of top decoder genes
        """
        return {
            'encoder_genes': res['encoder_top_genes'][latent_idx].head(n_genes),
            'decoder_genes': res['decoder_top_genes'][latent_idx].head(n_genes)
        }
    
    def print_pathway_summary(
        self,
        res: Dict,
        latent_idx: int = None,
        n_genes: int = 10
    ):
        """
        Print beautiful summary of pathways.
        
        Parameters
        ----------
        res : dict
            Output from compute_both_pathways_combined_scores()
        latent_idx : int, optional
            If None, print for all latents
        n_genes : int
            Number of top genes to show per pathway
        """
        if latent_idx is not None:
            latent_indices = [latent_idx]
        else:
            latent_indices = range(self.n_latent)
        
        for l in latent_indices:
            print(f"\n{'='*70}")
            print(f"LATENT {l} SUMMARY")
            print(f"{'='*70}")
            
            print(f"\n📊 ENCODER PATHWAY (Discriminative Genes)")
            print("-" * 70)
            enc_df = res['encoder_top_genes'][l].head(n_genes)
            for _, row in enc_df.iterrows():
                print(f"  {row['gene']:20s} | Score: {row['score']:7.4f} | Rank: {int(row['rank'])}")
            
            print(f"\n🔄 DECODER PATHWAY (Reconstructive Genes)")
            print("-" * 70)
            dec_df = res['decoder_top_genes'][l].head(n_genes)
            for _, row in dec_df.iterrows():
                print(f"  {row['gene']:20s} | Score: {row['score']:7.4f} | Rank: {int(row['rank'])}")
            
            # Overlap info
            overlap = set(res['encoder_top_genes'][l]['gene']) & set(res['decoder_top_genes'][l]['gene'])
            print(f"\n✓ Overlap: {len(overlap)} genes")
            if overlap:
                print(f"  Shared genes: {', '.join(list(overlap)[:5])}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_encoder_decoder_comparison(
            self,
            adata,
            res: Dict,
            layout: str = 'side_by_side',
            figsize_scale: float = 1.0,
            cmap: str = 'icefire',
            s: int = 55,
            show_scores: bool = True,
            n_top_genes: int = 1,
            latent_order: str = 'default',
            show: bool = True
        ):
            """
            Elegant unified visualization comparing encoder and decoder pathways.
            
            Parameters
            ----------
            adata : AnnData
                Annotated data object
            res : dict
                Output from compute_both_pathways_combined_scores()
            layout : str, default='side_by_side'
                'side_by_side': [Enc_Lat | Enc_Gene | Dec_Lat | Dec_Gene] × n_latents
                'multi_gene': Show multiple top genes per latent
            figsize_scale : float
                Scale factor for figure size
            cmap : str
                Colormap for plotting
            s : int
                Point size
            show_scores : bool
                Whether to show scores in titles
            n_top_genes : int
                Number of top genes to show per latent (multi_gene layout)
            latent_order : str
                'default': Original order [0, 1, 2, ...]
                'overlap_count': Order by number of overlapped genes (descending)
                'overlap_jaccard': Order by Jaccard index (descending)
            show : bool
                Whether to display the figure
            
            Returns
            -------
            fig : matplotlib.figure.Figure
                The figure object
            """
            import scanpy as sc
            import matplotlib.pyplot as plt
            
            # Get latent ordering
            latent_indices = self.get_latent_order_by_overlap(res, order_by=latent_order)
            n_latents = len(latent_indices)
            
            # ====================================================================
            # Layout: side_by_side (recommended for most comparisons)
            # ====================================================================
            if layout == 'side_by_side':
                fig = plt.figure(figsize=(20 * figsize_scale, 4 * n_latents * figsize_scale))
                gs = fig.add_gridspec(n_latents, 4, hspace=0.35, wspace=0.25)
                
                for row_idx, l in enumerate(latent_indices):
                    enc_df = res['encoder_top_genes'][l]
                    dec_df = res['decoder_top_genes'][l]
                    
                    # Column 0: Encoder Latent
                    ax = fig.add_subplot(gs[row_idx, 0])
                    sc.pl.umap(adata, color=[f'L{l}'], ax=ax, show=False, cmap=cmap, s=s)
                    ax.set_title(f'Encoder Latent {l}', fontsize=11, fontweight='bold')
                    
                    # Column 1: Encoder Top Gene
                    ax = fig.add_subplot(gs[row_idx, 1])
                    gene_enc = enc_df['gene'].iloc[0]
                    score_enc = enc_df['score'].iloc[0]
                    title_enc = f'Enc Gene: {gene_enc}'
                    if show_scores:
                        title_enc += f'\nScore: {score_enc:.3f}'
                    sc.pl.umap(adata, color=[gene_enc], ax=ax, show=False, cmap=cmap, s=s)
                    ax.set_title(title_enc, fontsize=10)
                    
                    # Column 2: Decoder Latent
                    ax = fig.add_subplot(gs[row_idx, 2])
                    sc.pl.umap(adata, color=[f'L{l}'], ax=ax, show=False, cmap=cmap, s=s)
                    ax.set_title(f'Decoder Latent {l}', fontsize=11, fontweight='bold')
                    
                    # Column 3: Decoder Top Gene
                    ax = fig.add_subplot(gs[row_idx, 3])
                    gene_dec = dec_df['gene'].iloc[0]
                    score_dec = dec_df['score'].iloc[0]
                    title_dec = f'Dec Gene: {gene_dec}'
                    if show_scores:
                        title_dec += f'\nScore: {score_dec:.3f}'
                    sc.pl.umap(adata, color=[gene_dec], ax=ax, show=False, cmap=cmap, s=s)
                    ax.set_title(title_dec, fontsize=10)
            
            # ====================================================================
            # Layout: multi_gene (show multiple top genes)
            # ====================================================================
            elif layout == 'multi_gene':
                # Get genes to plot from each pathway
                genes_to_plot_enc = {}
                genes_to_plot_dec = {}
                
                for l in latent_indices:
                    # Show top genes from encoder pathway
                    enc_df = res['encoder_top_genes'][l].head(n_top_genes)
                    genes_to_plot_enc[l] = [
                        (row['gene'], row['score']) 
                        for _, row in enc_df.iterrows()
                    ]
                    
                    # Show top genes from decoder pathway
                    dec_df = res['decoder_top_genes'][l].head(n_top_genes)
                    genes_to_plot_dec[l] = [
                        (row['gene'], row['score']) 
                        for _, row in dec_df.iterrows()
                    ]
                
                # Determine max genes across all latents
                max_genes = max(
                    max(len(genes_to_plot_enc[l]) for l in latent_indices) if latent_indices else 0,
                    max(len(genes_to_plot_dec[l]) for l in latent_indices) if latent_indices else 0
                )
                
                if max_genes == 0:
                    print("Warning: No genes to plot!")
                    return None
                
                # Create figure
                fig = plt.figure(figsize=(6 * max_genes * 2 * figsize_scale, 4 * n_latents * figsize_scale))
                gs = fig.add_gridspec(n_latents, max_genes * 2, hspace=0.25, wspace=0.2)
                
                for row_idx, l in enumerate(latent_indices):
                    enc_genes = genes_to_plot_enc[l]
                    dec_genes = genes_to_plot_dec[l]
                    
                    # Plot encoder genes
                    for gene_idx, (gene, score) in enumerate(enc_genes):
                        ax = fig.add_subplot(gs[row_idx, gene_idx * 2])
                        title = f'{gene}'
                        if show_scores:
                            title += f'\n({score:.2f})'
                        sc.pl.umap(adata, color=[gene], ax=ax, show=False, cmap=cmap, s=s)
                        ax.set_title(f'Enc #{gene_idx+1}: {title}', fontsize=10)
                        ax.set_xlabel('')
                        ax.set_ylabel('')
                    
                    # Plot decoder genes
                    for gene_idx, (gene, score) in enumerate(dec_genes):
                        ax = fig.add_subplot(gs[row_idx, gene_idx * 2 + 1])
                        title = f'{gene}'
                        if show_scores:
                            title += f'\n({score:.2f})'
                        sc.pl.umap(adata, color=[gene], ax=ax, show=False, cmap=cmap, s=s)
                        ax.set_title(f'Dec #{gene_idx+1}: {title}', fontsize=10)
                        ax.set_xlabel('')
                        ax.set_ylabel('')
            
            else:
                raise ValueError(f"Unknown layout: {layout}. Use 'side_by_side' or 'multi_gene'")
            
            if show:
                plt.show()
            
            return fig