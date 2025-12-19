import time
import torch
import numpy as np
import scanpy as sc  
from sklearn.model_selection import train_test_split  
# scVI imports
from scvi.model import SCVI, PEAKVI  
from scvi.external import POISSONVI  

class DataSplitter:
    """
    Simplified data splitter for consistent train/val/test splits across all models.
    
    Strategy: 70% train, 15% val, 15% test
    """
    
    def __init__(self, n_samples, test_size=0.15, val_size=0.15, random_state=42):
        """
        Parameters
        ----------
        n_samples : int
            Total number of samples
        test_size : float
            Proportion for test set (default 0.15)
        val_size : float
            Proportion for validation set (default 0.15)
        random_state : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.train_val_size = 1 - test_size
        self.val_size_relative = val_size / self.train_val_size
        
        self._create_splits()
    
    def _create_splits(self):
        """Create train/val/test indices"""
        indices = np.arange(self.n_samples)
        
        # First split: separate test set (15%)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Second split: separate train and val from remaining 85%
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.val_size_relative,
            random_state=self.random_state,
            shuffle=True
        )
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.train_val_idx = train_val_idx
        
        print("\nData split sizes:")
        print(f"  Total: {self.n_samples}")
        print(f"  Train: {len(train_idx)} ({len(train_idx)/self.n_samples*100:.1f}%)")
        print(f"  Val:   {len(val_idx)} ({len(val_idx)/self.n_samples*100:.1f}%)")
        print(f"  Test:  {len(test_idx)} ({len(test_idx)/self.n_samples*100:.1f}%)")
    
    def get_scvi_validation_size(self):
        """Get validation size for scVI's internal split"""
        return self.val_size_relative


def train_scvi_models(adata, splitter, n_latent=10, n_epochs=400, batch_size=128):
    """
    Train scVI-architecture models (SCVI, PEAKVI, POISSONVI) with consistent splits.
    
    Parameters
    ----------
    adata : AnnData
        Full dataset
    splitter : DataSplitter
        Data splitter instance
    n_latent : int
        Latent dimension
    n_epochs : int
        Number of training epochs
    
    Returns
    -------
    dict
        Dictionary containing trained models and their test data
    """
    results = {}
    use_cuda = torch.cuda.is_available()
    
    # Prepare data: scVI sees only train+val (85%)
    adata_trainval = adata[splitter.train_val_idx].copy()
    adata_test = adata[splitter.test_idx].copy()
    
    validation_size = splitter.get_scvi_validation_size()
    
    # ==================== SCVI ====================
    print(f"\n{'='*70}")
    print("Training SCVI Model")
    print(f"{'='*70}")
    
    try:
        # Reset GPU memory stats and start timer
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Setup for SCVI (uses raw counts from layers)
        SCVI.setup_anndata(
            adata_trainval,
            layer="counts",
            batch_key=None
        )
        
        scvi_model = SCVI(
            adata_trainval,
            n_latent=n_latent,
            n_hidden=128,
            n_layers=2,
            dropout_rate=0.1,
            gene_likelihood="nb"
        )
        
        scvi_model.train(
            max_epochs=n_epochs,
            train_size=1 - validation_size,
            validation_size=validation_size,
            early_stopping=True,
            early_stopping_patience=20,
            check_val_every_n_epoch=5,
            batch_size=batch_size,
            plan_kwargs={'lr': 1e-4}
        )
        
        # Record time and GPU memory
        train_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0
        _hist = scvi_model.history or {}
        actual_epochs = len(_hist.get('elbo_train', []))  # type: ignore[index]
        
        # Setup test data
        SCVI.setup_anndata(adata_test, layer="counts", batch_key=None)
        
        results['scvi'] = {
            'model': scvi_model,
            'adata_test': adata_test.copy(),
            'history': scvi_model.history,
            'train_time': train_time,
            'peak_memory_gb': peak_memory,
            'actual_epochs': actual_epochs
        }
        
        print("✓ SCVI training completed")
        print(f"  Epochs: {actual_epochs}/{n_epochs}, Time: {train_time:.2f}s, Peak GPU Memory: {peak_memory:.3f} GB")
        
    except Exception as e:
        print(f"✗ SCVI training failed: {str(e)}")
        results['scvi'] = None
    
    # ==================== PEAKVI ====================
    print(f"\n{'='*70}")
    print("Training PEAKVI Model")
    print(f"{'='*70}")
    
    try:
        # Reset GPU memory stats and start timer
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Reset adata for PEAKVI
        adata_trainval = adata[splitter.train_val_idx].copy()
        
        PEAKVI.setup_anndata(
            adata_trainval,
            layer="counts",
            batch_key=None
        )
        
        peakvi_model = PEAKVI(
            adata_trainval,
            n_latent=n_latent,
            n_hidden=128,
        )
        
        peakvi_model.train(
            max_epochs=n_epochs,
            train_size=1 - validation_size,
            validation_size=validation_size,
            early_stopping=True,
            early_stopping_patience=20,
            check_val_every_n_epoch=5,
            batch_size=batch_size,
            plan_kwargs={'lr': 1e-4}
        )
        
        # Record time and GPU memory
        train_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0
        _hist = peakvi_model.history or {}
        actual_epochs = len(_hist.get('elbo_train', []))  # type: ignore[index]
        
        # Setup test data
        adata_test_peakvi = adata[splitter.test_idx].copy()
        PEAKVI.setup_anndata(adata_test_peakvi, layer="counts", batch_key=None)
        
        results['peakvi'] = {
            'model': peakvi_model,
            'adata_test': adata_test_peakvi,
            'history': peakvi_model.history,
            'train_time': train_time,
            'peak_memory_gb': peak_memory,
            'actual_epochs': actual_epochs
        }
        
        print("✓ PEAKVI training completed")
        print(f"  Epochs: {actual_epochs}/{n_epochs}, Time: {train_time:.2f}s, Peak GPU Memory: {peak_memory:.3f} GB")
        
    except Exception as e:
        print(f"✗ PEAKVI training failed: {str(e)}")
        results['peakvi'] = None
    
    # ==================== POISSONVI ====================
    print(f"\n{'='*70}")
    print("Training POISSONVI Model")
    print(f"{'='*70}")
    
    try:
        # Reset GPU memory stats and start timer
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Reset adata for POISSONVI
        adata_trainval = adata[splitter.train_val_idx].copy()
        
        POISSONVI.setup_anndata(
            adata_trainval,
            layer="counts",
            batch_key=None
        )
        
        poissonvi_model = POISSONVI(
            adata_trainval,
            n_latent=n_latent,
            n_hidden=128,
        )
        
        poissonvi_model.train(
            max_epochs=n_epochs,
            train_size=1 - validation_size,
            validation_size=validation_size,
            early_stopping=True,
            early_stopping_patience=20,
            check_val_every_n_epoch=5,
            batch_size=batch_size,
            plan_kwargs={'lr': 1e-4}
        )
        
        # Record time and GPU memory
        train_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0
        _hist = poissonvi_model.history or {}
        actual_epochs = len(_hist.get('elbo_train', []))  # type: ignore[index]
        
        # Setup test data
        adata_test_poissonvi = adata[splitter.test_idx].copy()
        POISSONVI.setup_anndata(adata_test_poissonvi, layer="counts", batch_key=None)
        
        results['poissonvi'] = {
            'model': poissonvi_model,
            'adata_test': adata_test_poissonvi,
            'history': poissonvi_model.history,
            'train_time': train_time,
            'peak_memory_gb': peak_memory,
            'actual_epochs': actual_epochs
        }
        
        print("✓ POISSONVI training completed")
        print(f"  Epochs: {actual_epochs}/{n_epochs}, Time: {train_time:.2f}s, Peak GPU Memory: {peak_memory:.3f} GB")
        
    except Exception as e:
        print(f"✗ POISSONVI training failed: {str(e)}")
        results['poissonvi'] = None
    
    return results
