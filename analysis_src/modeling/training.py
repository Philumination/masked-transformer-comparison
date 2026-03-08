import torch
import torch.nn as nn
import scipy.sparse
from torch.utils.data import Dataset,  DataLoader
import numpy as np


#dataset

class MicrobiomeDataset(Dataset):
    def __init__(self, adata):
        # Convert sparse matrix to dense if necessary, though ideally we handle sparse tensors
        # We now run on CLRs
        if scipy.sparse.issparse(adata.layers["Clrs"]):
            self.data = adata.layers["Clrs"].toarray()
        else:
            self.data = adata.layers["Clrs"]

        # Standardize inputs to Float32
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.n_samples, self.n_microbes = self.data.shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]



class MicrobiomeDataset_counts(Dataset):
    def __init__(self, adata):
        # Convert sparse matrix to dense if necessary, though ideally we handle sparse tensors
        # We now run on raww counts
        if scipy.sparse.issparse(adata.X):
            self.data = adata.X.toarray()
        else:
            self.data = adata.X

        # Standardize inputs to Float32
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.n_samples, self.n_microbes = self.data.shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
 


# simple training loop 
def train_masked_model(model, dataloader, epochs=10, learning_rate=1e-3,
                        device=None, plot_loss=False, loss_weight=0.1,
                          show_progress=True,
):
    
    


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="none")

    losses = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_masked_loss = 0.0

        for masked_input, target, mask in dataloader:
            masked_input = masked_input.to(device)
            target = target.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            reconstruction = model(masked_input, mask)
            loss_matrix = criterion(reconstruction, target)

            masked_loss = (loss_matrix * mask.float()).sum() / (mask.sum() + 1e-6)
            global_loss = loss_matrix.mean()
            final_loss = masked_loss + (loss_weight * global_loss)

            final_loss.backward()
            optimizer.step()

            total_loss += final_loss.item()
            total_masked_loss += masked_loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        avg_masked_loss = total_masked_loss / max(len(dataloader), 1)
        losses.append(avg_loss)

        if plot_loss:
            from IPython.display import clear_output
            import matplotlib.pyplot as plt

            clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(losses) + 1), losses, "b-o", linewidth=2, markersize=8)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training Loss - Epoch {epoch + 1}/{epochs}")
            plt.grid(True, alpha=0.3)
            plt.show()

        if show_progress:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Total Loss: {avg_loss:.4f} | Masked Loss: {avg_masked_loss:.4f}"
            )

    if show_progress:
        print("\nTraining complete.")

    return losses

#multiple mask percentages training loop

def train_multi_mask(
    adata,
    mask_probs,
    use_transformer=False,
    batch_size=32,
    learning_rate=1e-3,
    epochs=10,
    non_zero_bias=0.8,
    loss_weight=0.1,
    device=None,
    csv_dir="../data/interim",
    dataset = MicrobiomeDataset,
):
  
    import pandas as pd
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    from analysis_src.modeling.mlmodels import (
        MicrobiomeMaskedAutoencoder, 
        MicrobiomeLatentTransformer
    )
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = dataset(adata)
    print(f"dataset_name: {dataset.__class__.__name__}")
    trained_models = {}
    training_histories = {}
    
    for mask_prob in mask_probs:
        print(f"\n{'='*60}\nTraining with mask_prob={mask_prob}\n{'='*60}")
        data_type = 'counts' if isinstance(dataset, MicrobiomeDataset_counts) else 'clrs'
        print(f"will be saved as: {csv_dir}/{('transformer' if use_transformer else 'mae')}_nonzero_bias_{non_zero_bias}_data_{data_type}_mask_{str(mask_prob).replace('.', '_')}.csv")
        
        # Data loader with current mask prob
        collator = ZeroBiasedMaskCollator(mask_prob=mask_prob, non_zero_bias=non_zero_bias)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator
        )
        
        # Model init
        if use_transformer:
            model = MicrobiomeLatentTransformer(num_microbes=dataset.n_microbes).to(device)
        else:
            model = MicrobiomeMaskedAutoencoder(
                num_microbes=dataset.n_microbes,
                latent_dim=64,
                hidden_dims=[512, 256]
            ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss(reduction="none")
        
        losses = []
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_masked_loss = 0
            
            for masked_input, target, mask in dataloader:
                masked_input = masked_input.to(device)
                target = target.to(device)
                mask = mask.to(device)
                
                optimizer.zero_grad()
                reconstruction = model(masked_input, mask)
                loss_matrix = criterion(reconstruction, target)
                masked_loss = (loss_matrix * mask.float()).sum() / (mask.sum() + 1e-6)
                global_loss = loss_matrix.mean()
                final_loss = masked_loss + (loss_weight * global_loss)
                
                final_loss.backward()
                optimizer.step()
                
                total_loss += final_loss.item()
                total_masked_loss += masked_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_masked_loss = total_masked_loss / len(dataloader)
            losses.append(avg_loss)
            
            clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(losses)+1), losses, 'b-o', linewidth=2, markersize=6)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title(f"Loss (mask_prob={mask_prob}) - Epoch {epoch+1}/{epochs}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Masked Loss={avg_masked_loss:.4f}")
        
        print("Training complete.")
        trained_models[mask_prob] = model
        training_histories[mask_prob] = losses
        
        # embedding extraction and csv savig
        latent_vectors = extract_embeddings_generic(model, dataset, device)
        model_type = "transformer" if use_transformer else "mae"
        data_type = "counts" if isinstance(dataset, MicrobiomeDataset_counts) else 'clrs'
        mask_prob_str = str(mask_prob).replace(".", "_")
        
        csv_filename = f"{csv_dir}/{model_type}_nonzero_bias_{non_zero_bias}_data_{data_type}_mask_{mask_prob_str}.csv"
        pd.DataFrame(latent_vectors).to_csv(csv_filename, index=False)
        print(f"Saved embeddings to {csv_filename} with shape {latent_vectors.shape}")
    
    return trained_models, training_histories


class ZeroBiasedMaskCollator:
    def __init__(self, mask_prob=0.15, non_zero_bias=0.8):
        """
        Args:
            mask_prob (float): Total percentage of features to mask per sample.
            non_zero_bias (float): Probability that a chosen mask target comes from non-zero entries.
                                   If 0.8, we try to make 80% of our masks cover non-zero values.
        """
        self.mask_prob = mask_prob
        self.non_zero_bias = non_zero_bias
    def __call__(self, batch):
        # batch is a list of tensors, stack them: (Batch, Microbes)
        batch = torch.stack(batch)
        B, M = batch.shape
        # Create the mask container (False = not masked, True = masked)
        mask = torch.zeros_like(batch, dtype=torch.bool)
        # Vectorized implementation of biased masking
        for i in range(B):
            sample = batch[i]
            # Identify indices
            nonzero_indices = torch.nonzero(sample).squeeze(-1)
            zero_indices = torch.nonzero(sample == 0).squeeze(-1)
            num_to_mask = int(M * self.mask_prob)
            # Determine how many non-zeros we want to target
            # We clamp it so we don't try to mask more non-zeros than exist
            target_nz_count = int(num_to_mask * self.non_zero_bias)
            actual_nz_count = min(target_nz_count, len(nonzero_indices))
            # The rest must come from zeros
            actual_zero_count = num_to_mask - actual_nz_count
            # Randomly select indices
            if len(nonzero_indices) > 0:
                perm_nz = torch.randperm(len(nonzero_indices))[:actual_nz_count]
                mask[i, nonzero_indices[perm_nz]] = True
            if len(zero_indices) > 0:
                perm_z = torch.randperm(len(zero_indices))[:actual_zero_count]
                mask[i, zero_indices[perm_z]] = True
        # Create masked input: Copy batch and zero out masked values
        # (In BERT we replace with special token, in continuous data we often replace with 0 or a learnable vector)
        masked_input = batch.clone()
        masked_input[mask] = 0  # Zeroing out the masked values 
        # nochmal schauen wegen 0 
        return masked_input, batch, mask
    



# new generic embedding extraction 
def extract_embeddings_generic(model, dataset, device, batch_size=32):
    model.eval()
    embeddings = []
    simple_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for clean_input in simple_loader:
            clean_input = clean_input.to(device)
            if hasattr(model, 'get_low_dim_repr'):
                z = model.get_low_dim_repr(clean_input)
            else:
                z = model.encoder(clean_input)
            embeddings.append(z.cpu().numpy())
    return np.vstack(embeddings)