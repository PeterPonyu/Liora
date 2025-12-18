"""
Interpretability analysis for Liora VAE with attention and ODE components.

Provides comprehensive attribution analysis:
1. Attention pathway: Genes → Tokens → Latents → Outputs
2. ODE pathway: Initial latent → Trajectory dynamics → Final state
3. Cross-pathway: How attention tokens influence ODE behavior
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import spearmanr
import warnings


class LioraInterpretability:
    """
    Complete interpretability analysis for Liora models.
    
    Handles both MLP and attention-based encoders, with special
    support for ODE trajectory analysis.
    
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
    # PART 1: ENCODER INTERPRETABILITY
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
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
        latent_idx : int, optional
            Specific latent dimension (None = all dimensions)
        use_mu : bool
            Use mean (True) or sampled latent (False)
        
        Returns
        -------
        relevance : torch.Tensor
            - If latent_idx is None: shape (n_genes, n_latent)
            - Otherwise: shape (n_genes,)
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
        """Compute gene relevance for a single latent dimension."""
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
        
        # Gene relevance
        relevance = torch.abs(x.grad).mean(dim=0)
        
        return relevance
    
    def compute_attention_token_attribution(
        self,
        x: torch.Tensor,
        return_token_embeddings: bool = False,
        perturbation_scale: float = 0.0  # 0 = zero out, >0 = add noise
    ) -> Dict[str, torch.Tensor]:
        """
        Perturbation-based attribution (most reliable).
        """
        if not self.is_attention:
            raise ValueError("This method requires attention-based encoder")
        
        # === Step 1: Gene → Token (gradient-based, this works) ===
        gene_to_token = torch.zeros(self.n_genes, self.n_tokens, device=self.device)
        
        for token_idx in range(self.n_tokens):
            x_grad = x.clone().detach().requires_grad_(True)
            
            proj = self.vae.encoder.input_proj(x_grad)
            bsz = proj.size(0)
            seq = proj.view(bsz, self.n_tokens, self.token_dim)
            
            target = seq[:, token_idx, :].sum()
            target.backward()
            
            gene_to_token[:, token_idx] = torch.abs(x_grad.grad).mean(dim=0)
        
        # === Step 2: Token → Latent (perturbation-based) ===
        token_to_latent = torch.zeros(self.n_tokens, self.n_latent, device=self.device)
        
        with torch.no_grad():
            # Baseline: Get latent with all tokens
            if self.is_ode:
                _, q_m_baseline, _, _, _ = self.vae.encoder(x)
            else:
                _, q_m_baseline, _, _ = self.vae.encoder(x)
            q_m_baseline = q_m_baseline.mean(dim=0)  # Average over batch
            
            # Perturb each token
            for token_idx in range(self.n_tokens):
                # Get projection
                proj = self.vae.encoder.input_proj(x)
                bsz = proj.size(0)
                seq = proj.view(bsz, self.n_tokens, self.token_dim)
                
                # Zero out (or perturb) this token
                seq_perturbed = seq.clone()
                if perturbation_scale == 0.0:
                    seq_perturbed[:, token_idx, :] = 0.0
                else:
                    seq_perturbed[:, token_idx, :] += torch.randn_like(
                        seq[:, token_idx, :]
                    ) * perturbation_scale
                
                # Forward through rest of encoder
                seq_perturbed = seq_perturbed.transpose(0, 1)
                seq_out = self.vae.encoder.transformer(seq_perturbed)
                seq_out = seq_out.transpose(0, 1)
                
                if self.vae.encoder.use_layer_norm:
                    seq_out = self.vae.encoder.attn_ln(seq_out)
                
                pooled = seq_out.mean(dim=1)
                output = self.vae.encoder.attn_pool_fc(pooled)
                q_m_perturbed, _ = torch.chunk(output, 2, dim=-1)
                q_m_perturbed = q_m_perturbed.mean(dim=0)
                
                # Measure impact
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
        
        Combines gene_to_token and token_to_latent attributions.
        
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
    # PART 2: DECODER INTERPRETABILITY
    # ========================================================================
    
    def compute_decoder_latent_to_gene(
        self,
        z: torch.Tensor,
        gene_idx: Optional[int] = None,
        use_mu: bool = True
    ) -> torch.Tensor:
        """
        Gradient-based decoder interpretability: Which latents affect which genes?
        
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
        """Compute latent relevance for a single output gene."""
        z = z.clone().detach().requires_grad_(True)
        
        # Forward through decoder
        recon_mu, recon_logvar = self.vae.decoder(z)
        
        if use_mu:
            target = recon_mu[:, gene_idx].sum()
        else:
            std = torch.exp(0.5 * recon_logvar)
            eps = torch.randn_like(std)
            recon = recon_mu + eps * std
            target = recon[:, gene_idx].sum()
        
        target.backward()
        
        relevance = torch.abs(z.grad).mean(dim=0)
        
        return relevance


    def compute_encoder_combined_score_matrix(
        self,
        x: torch.Tensor,
        correlation_method: str = 'pearson',
        corr_weight: float = 0.6,
        grad_weight: float = 0.4
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined score matrix for encoder: Genes → Latents.
        
        Combines correlation and gradient-based importance with normalization.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
        correlation_method : str
            'pearson' or 'spearman'
        corr_weight : float
            Weight for correlation component (default: 0.6)
        grad_weight : float
            Weight for gradient component (default: 0.4)
        
        Returns
        -------
        results : dict
            - 'correlation_matrix': (n_genes, n_latent) correlation scores
            - 'gradient_matrix': (n_genes, n_latent) gradient scores (normalized to [0,1])
            - 'combined_matrix': (n_genes, n_latent) weighted combination
            
        Interpretation:
            Entry [i, j] = how much gene_i influences latent_j
        """
        # === 1. Compute Correlation Matrix ===
        with torch.no_grad():
            if self.is_ode:
                q_z, q_m, q_s, n, t = self.vae.encoder(x)
            else:
                q_z, q_m, q_s, n = self.vae.encoder(x)
            
            latent_np = q_z.cpu().numpy()
            genes_np = x.cpu().numpy()
        
        # Compute correlation matrix
        combined = np.concatenate([genes_np, latent_np], axis=1)
        
        if correlation_method == 'pearson':
            corr_matrix_full = np.corrcoef(combined.T)
        else:  # spearman
            corr_matrix_full, _ = spearmanr(combined, axis=0)
        
        # Extract gene-latent block: (n_genes, n_latent)
        corr_matrix = corr_matrix_full[:self.n_genes, self.n_genes:]
        # corr_matrix = np.abs(corr_matrix)  # Use absolute correlation
        
        # === 2. Compute Gradient Matrix ===
        grad_matrix = self.compute_encoder_gene_to_latent(
            x, latent_idx=None, use_mu=True
        )  # (n_genes, n_latent)
        grad_matrix_np = grad_matrix.cpu().numpy()
        
        # === 3. Normalize Gradient to [0, 1] ===
        grad_min = grad_matrix_np.min()
        grad_max = grad_matrix_np.max()
        
        if grad_max > grad_min:
            grad_matrix_normalized = (grad_matrix_np - grad_min) / (grad_max - grad_min)
        else:
            grad_matrix_normalized = np.zeros_like(grad_matrix_np)
        
        # === 4. Combine Scores ===
        combined_matrix = corr_weight * corr_matrix + grad_weight * grad_matrix_normalized
        
        return {
            'correlation_matrix': torch.from_numpy(corr_matrix).float(),
            'gradient_matrix': torch.from_numpy(grad_matrix_normalized).float(),
            'combined_matrix': torch.from_numpy(combined_matrix).float()
        }
    
    
    def compute_decoder_combined_score_matrix(
        self,
        x: torch.Tensor,
        correlation_method: str = 'pearson',
        corr_weight: float = 0.6,
        grad_weight: float = 0.4
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined score matrix for decoder: Latents → Genes.
        
        Combines NEW correlation (between reconstructed genes and latents) 
        with gradient-based importance.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
        correlation_method : str
            'pearson' or 'spearman'
        corr_weight : float
            Weight for correlation component (default: 0.6)
        grad_weight : float
            Weight for gradient component (default: 0.4)
        
        Returns
        -------
        results : dict
            - 'correlation_matrix': (n_genes, n_latent) correlation scores
            - 'gradient_matrix': (n_genes, n_latent) gradient scores (normalized to [0,1])
            - 'combined_matrix': (n_genes, n_latent) weighted combination
            
        Interpretation:
            Entry [i, j] = how much latent_j influences gene_i (in decoder)
        """
        # === 1. Get latent representation and decode ===
        with torch.no_grad():
            if self.is_ode:
                q_z, q_m, q_s, n, t = self.vae.encoder(x)
            else:
                q_z, q_m, q_s, n = self.vae.encoder(x)
            
            # Decode to get reconstructed genes
            recon_mu, recon_logvar = self.vae.decoder(q_z)
            
            latent_np = q_z.cpu().numpy()
            recon_genes_np = recon_mu.cpu().numpy()
        
        # === 2. Compute NEW Correlation: recon_genes vs latents ===
        combined = np.concatenate([recon_genes_np, latent_np], axis=1)
        
        if correlation_method == 'pearson':
            corr_matrix_full = np.corrcoef(combined.T)
        else:  # spearman
            corr_matrix_full, _ = spearmanr(combined, axis=0)
        
        # Extract gene-latent block: (n_genes, n_latent)
        corr_matrix = corr_matrix_full[:self.n_genes, self.n_genes:]
        # corr_matrix = np.abs(corr_matrix)
        
        # === 3. Compute Gradient Matrix ===
        grad_matrix = self.compute_decoder_latent_to_gene(
            q_z, gene_idx=None, use_mu=True
        )  # Returns (n_latent, n_genes)
        
        # Transpose to (n_genes, n_latent) for consistency
        grad_matrix = grad_matrix.T  # Now (n_genes, n_latent)
        grad_matrix_np = grad_matrix.cpu().numpy()
        
        # === 4. Normalize Gradient to [0, 1] ===
        grad_min = grad_matrix_np.min()
        grad_max = grad_matrix_np.max()
        
        if grad_max > grad_min:
            grad_matrix_normalized = (grad_matrix_np - grad_min) / (grad_max - grad_min)
        else:
            grad_matrix_normalized = np.zeros_like(grad_matrix_np)
        
        # === 5. Combine Scores ===
        combined_matrix = corr_weight * corr_matrix + grad_weight * grad_matrix_normalized
        
        return {
            'correlation_matrix': torch.from_numpy(corr_matrix).float(),
            'gradient_matrix': torch.from_numpy(grad_matrix_normalized).float(),
            'combined_matrix': torch.from_numpy(combined_matrix).float()
        }
    
    
    def get_top_genes_per_latent_from_matrix(
        self,
        score_matrix: torch.Tensor,
        top_k: int = 50,
        return_unique: bool = False
    ) -> Dict[int, pd.DataFrame]:
        """
        Extract top genes for each latent from a score matrix.
        
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
            Key: latent_idx, Value: DataFrame with columns [gene, score, rank]
        """
        score_matrix_np = score_matrix.cpu().numpy()
        results = {}
        
        if return_unique:
            # Each gene assigned to its best latent only
            assigned_genes = set()
            
            # Sort all (gene, latent) pairs by score
            all_pairs = []
            for gene_idx in range(self.n_genes):
                for latent_idx in range(self.n_latent):
                    all_pairs.append({
                        'gene_idx': gene_idx,
                        'latent_idx': latent_idx,
                        'score': score_matrix_np[gene_idx, latent_idx]
                    })
            
            all_pairs_df = pd.DataFrame(all_pairs).sort_values('score', ascending=False)
            
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
                
                # Get top-k genes
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                gene_list = []
                for rank, gene_idx in enumerate(top_indices):
                    gene_list.append({
                        'gene': self.gene_names[gene_idx],
                        'score': scores[gene_idx],
                        'rank': rank + 1
                    })
                
                results[latent_idx] = pd.DataFrame(gene_list)
        
        return results
    
    
    def compute_both_pathways_combined_scores(
        self,
        x: torch.Tensor,
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
        
        Returns matrices and top genes for both directions.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)
            Input gene expression
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
            - 'encoder_correlation_matrix': (n_genes, n_latent)
            - 'encoder_gradient_matrix': (n_genes, n_latent)
            - 'encoder_combined_matrix': (n_genes, n_latent)
            - 'encoder_top_genes': dict of DataFrames
            
            - 'decoder_correlation_matrix': (n_genes, n_latent)
            - 'decoder_gradient_matrix': (n_genes, n_latent)
            - 'decoder_combined_matrix': (n_genes, n_latent)
            - 'decoder_top_genes': dict of DataFrames
        """
        # === Encoder Pathway ===
        print("Computing encoder combined scores...")
        encoder_results = self.compute_encoder_combined_score_matrix(
            x,
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
            'encoder_combined_matrix': encoder_results['combined_matrix'],
            'encoder_top_genes': encoder_top_genes,
            
            # Decoder results
            'decoder_correlation_matrix': decoder_results['correlation_matrix'],
            'decoder_gradient_matrix': decoder_results['gradient_matrix'],
            'decoder_combined_matrix': decoder_results['combined_matrix'],
            'decoder_top_genes': decoder_top_genes,
        }