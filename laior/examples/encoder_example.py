"""
Small example to demonstrate the attention-based encoder.
Run from repository root with:

cd /home/zeyufu/LAB/Liora
python -m laior.examples.encoder_example

"""

import torch
from laior import module

def run_example():
    batch = 8
    n_genes = 200
    hidden_dim = 128
    latent_dim = 16
    i_dim = 4

    # Dummy input
    x = torch.randn(batch, n_genes)

    # MLP encoder (default)
    enc_mlp = module.Encoder(n_genes, hidden_dim, latent_dim)
    qz, qm, qs, dist = enc_mlp(x)
    print('MLP encoder output shapes:', qz.shape, qm.shape, qs.shape)

    # Transformer encoder
    enc_attn = module.Encoder(
        state_dim=n_genes,
        hidden_dim=hidden_dim,
        action_dim=latent_dim,
        encoder_type='transformer',
        attn_embed_dim=64,
        attn_num_heads=4,
        attn_num_layers=2,
        attn_seq_len=32
    )
    qz2, qm2, qs2, dist2 = enc_attn(x)
    print('Transformer encoder output shapes:', qz2.shape, qm2.shape, qs2.shape)

    # VAE end-to-end small forward
    vae = module.VAE(
        state_dim=n_genes,
        hidden_dim=hidden_dim,
        action_dim=latent_dim,
        i_dim=i_dim,
        encoder_type='transformer',
        attn_embed_dim=64,
        attn_num_heads=4,
        attn_num_layers=2,
        attn_seq_len=32
    )
    out = vae(x)
    print('VAE forward returned', len(out), 'items')

if __name__ == '__main__':
    run_example()
