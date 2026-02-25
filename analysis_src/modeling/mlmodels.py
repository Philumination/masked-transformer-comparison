# models definitions
import torch
import torch.nn as nn
import torch.nn.functional as F


# ran fast
class MicrobiomeMaskedAutoencoder(nn.Module):
    def __init__(self, num_microbes, latent_dim=64, hidden_dims=[512, 256]):
        """
        Args:
            num_microbes (int): Input feature size (number of species/OTUs).
            latent_dim (int): Size of the bottleneck layer.
            hidden_dims (list): List of hidden layer sizes for the encoder.
                                The decoder will be the reverse of this.
        """
        super().__init__()

        # --- Encoder ---
        encoder_layers = []
        in_dim = num_microbes

        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim

        # Bottleneck
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder --- # eventuell schwächerer decoder gegen noise lernen
        # Reverse the hidden dims for symmetry
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        in_dim = latent_dim

        for h_dim in hidden_dims_rev:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim

        # Final reconstruction layer
        decoder_layers.append(nn.Linear(in_dim, num_microbes))
        # Note: No activation at the end (output is raw logits/continuous values).
        # If your data is strictly counts, you might apply Softplus later or in the loss.
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, mask=None):
        """
        x: (Batch, Microbes) - The masked input data.
        mask: Ignored in the forward pass of an MLP (unlike Transformer),
              but kept in signature for compatibility.
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction
    

# ran kinda fast 3h
class MicrobiomeLatentTransformer(nn.Module):
    def __init__(self, num_microbes, num_latents=128, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.num_latents = num_latents
        self.d_model = d_model

        # --- 1. Embeddings (Same as your code) ---
        self.microbe_id_embedding = nn.Embedding(num_microbes, d_model)
        self.value_encoder = nn.Linear(1, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- 2. The Bottleneck (Latents) ---
        # These are the "seeds" for your low-dim representation
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))

        # --- 3. Attention Layers ---
        # A. Compression: Cross-Attention (Latents query the Data)
        self.cross_attn_compress = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)

        # B. Processing: Self-Attention (Latents attend to Latents) - Deep & Fast
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.latent_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # C. Reconstruction: Cross-Attention (Data queries the Latents)
        # We query using the microbe positional embeddings to "ask" the latents for values
        self.cross_attn_decode = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_decode = nn.LayerNorm(d_model)

        # --- 4. Output Head ---
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        B, M = x.shape # [Batch, 4000]

        # --- Step 1: Prepare Input (Your embedding logic) ---
        x_expanded = x.unsqueeze(-1)
        # ID embeddings serve as positional anchors
        pos_embeds = self.microbe_id_embedding(torch.arange(M, device=x.device).expand(B, -1))

        embeddings = self.value_encoder(x_expanded) + pos_embeds

        # Apply Masking
        if mask is not None:
            mask_token_expanded = self.mask_token.expand(B, M, -1)
            embeddings = torch.where(mask.unsqueeze(-1), mask_token_expanded, embeddings)

        # --- Step 2: Compress (4000 -> 128) ---
        # Query = Latents (expanded to batch), Key/Value = Microbe Data
        latents = self.latents.unsqueeze(0).expand(B, -1, -1) # [B, 128, D]

        # Latents look at the microbiome data to gather info
        compressed, _ = self.cross_attn_compress(
            query=latents,
            key=embeddings,
            value=embeddings
        )
        compressed = self.norm_cross(compressed + latents) # Residual

        # --- Step 3: Process (Deep reasoning on 128 dims) ---
        # This is the "Low Dimensional Representation" you want!
        latent_representation = self.latent_transformer(compressed) # [B, 128, D]

        # --- Step 4: Reconstruct (128 -> 4000) ---
        # Query = Microbe Positions (What value belongs here?), Key/Value = Latents
        # This asks: "Given the latent summary, what is the abundance of Microbe #42?"
        decoded, _ = self.cross_attn_decode(
            query=pos_embeds, # We query using the ID embeddings
            key=latent_representation,
            value=latent_representation
        )
        decoded = self.norm_decode(decoded + pos_embeds)

        return self.decoder(decoded).squeeze(-1)

    def get_low_dim_repr(self, x):
        """Extract the learned low-dim latent representation for downstream tasks"""
        B, M = x.shape
    
        # Step 1: Prepare Input
        x_expanded = x.unsqueeze(-1)
        pos_embeds = self.microbe_id_embedding(torch.arange(M, device=x.device).expand(B, -1))
        embeddings = self.value_encoder(x_expanded) + pos_embeds
    
        # Step 2: Compress (4000 -> 128)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, 128, D]
        compressed, _ = self.cross_attn_compress(
            query=latents,
            key=embeddings,
            value=embeddings
        )
        compressed = self.norm_cross(compressed + latents)
    
        # Step 3: get lowdim representation 
        latent_representation = self.latent_transformer(compressed)  # [B, 128, D]
        
        # Mean pool [B, D]
        return latent_representation.mean(dim=1)
    

# took too long to run
class MicrobiomeMaskedTransformer(nn.Module):
    def __init__(self, num_microbes, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.microbe_id_embedding = nn.Embedding(num_microbes, d_model)
        self.value_encoder = nn.Linear(1, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        
    def encode(self, x):
        """Extract fixed-size embeddings for downstream tasks"""
        B, M = x.shape
        x_expanded = x.unsqueeze(-1)
        
        embeddings = self.value_encoder(x_expanded) + \
                     self.microbe_id_embedding(torch.arange(M, device=x.device).expand(B, -1))
        
        encoded = self.transformer_encoder(embeddings)  # (B, M, d_model)
        return encoded.mean(dim=1)  # (B, d_model) - average pool across microbes

    def forward(self, x, mask=None):
        B, M = x.shape
        x_expanded = x.unsqueeze(-1)

        # Embed value + position
        embeddings = self.value_encoder(x_expanded) + \
                     self.microbe_id_embedding(torch.arange(M, device=x.device).expand(B, -1))

        # Apply learnable mask token
        if mask is not None:
            mask_token_expanded = self.mask_token.expand(B, M, -1)
            embeddings = torch.where(mask.unsqueeze(-1), mask_token_expanded, embeddings)

        encoded = self.transformer_encoder(embeddings)
        return self.decoder(encoded).squeeze(-1)
    
    
