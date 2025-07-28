#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
# import seaborn as sb
import matplotlib as mpl
# sb.set_theme(context='poster', 
#              style='ticks',
#              font='sans-serif', 
#              font_scale=1, 
#              color_codes=True, 
#              rc={"lines.linewidth": 1})
mpl.rcParams['savefig.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['savefig.bbox'] = 'tight'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 512*1
LEARNING_RATE = 1e-3
EPOCHS = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
P_TEACHER_FORCING = 0.15  # Probability of using ground truth during training
CONTEXT_LENGTH = 300  # Can be changed to 300 or other values < 784


#%%
class PixelLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=2, dropout=0.2):
        super(PixelLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layers for mean and log variance
        self.mean_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, 1)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Predict mean and log variance
        mean = torch.sigmoid(self.mean_head(lstm_out))  # Sigmoid to keep in [0,1]
        logvar = self.logvar_head(lstm_out)
        std = torch.exp(0.5 * logvar)
        
        return mean, std, hidden

def negative_log_likelihood_loss(mean, std, target):
    """Compute negative log likelihood loss for Gaussian distribution"""
    variance = std ** 2
    nll = 0.5 * torch.log(2 * np.pi * variance) + (target - mean) ** 2 / (2 * variance)
    return nll.mean()

def load_data():
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784 pixels
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                             download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                            download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def train_step(model, data, optimizer, p_teacher_forcing=0.5):
    """Single training step with teacher forcing"""
    model.train()
    batch_size, seq_len = data.shape
    
    # Prepare input and target sequences
    # Input: start with zero, then shifted sequence
    input_seq = torch.zeros(batch_size, seq_len, 1, device=device)
    target_seq = data.unsqueeze(-1)  # Add channel dimension
    
    total_loss = 0
    hidden = None
    
    for t in range(seq_len):
        # Forward pass for current timestep
        if t == 0:
            # Start with zero input
            current_input = input_seq[:, t:t+1, :]
        else:
            # Teacher forcing: use ground truth with probability p
            if np.random.random() < p_teacher_forcing:
                current_input = target_seq[:, t-1:t, :]
            else:
                # Use previous prediction
                current_input = prev_mean.detach()
        
        mean, std, hidden = model(current_input, hidden)
        prev_mean = mean
        
        # Compute loss for current timestep
        current_target = target_seq[:, t:t+1, :]
        loss = negative_log_likelihood_loss(mean, std, current_target)
        total_loss += loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return total_loss.item() / seq_len

def generate_completion(model, context_pixels, context_length=100):
    """Generate pixel completion given context"""
    model.eval()
    with torch.no_grad():
        batch_size = context_pixels.shape[0]
        seq_len = 784
        
        means = torch.zeros(batch_size, seq_len, device=device)
        stds = torch.zeros(batch_size, seq_len, device=device)
        
        hidden = None
        
        for t in range(seq_len):
            if t == 0:
                current_input = torch.zeros(batch_size, 1, 1, device=device)
            elif t <= context_length:
                # Use ground truth in context window
                current_input = context_pixels[:, t-1:t].unsqueeze(-1)
            else:
                # Use previous prediction for autoregressive generation
                current_input = means[:, t-1:t].unsqueeze(-1)
            
            mean, std, hidden = model(current_input, hidden)
            means[:, t] = mean.squeeze(-1).squeeze(-1)
            stds[:, t] = std.squeeze(-1).squeeze(-1)
    
    return means, stds

def train_model():
    """Main training loop"""
    # Load data
    train_loader, test_loader = load_data()
    
    # Initialize model
    model = PixelLSTM(input_size=1, hidden_size=HIDDEN_SIZE, 
                     num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Training history
    train_losses = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            data = data.to(device)
            loss = train_step(model, data, optimizer, P_TEACHER_FORCING)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
    
    return model, train_losses

def visualize_results(model, test_loader, train_losses):
    """Visualize training progress and generate completions"""
    # Plot training loss
    plt.figure(figsize=(15, 12))
    
    # Loss curve
    plt.subplot(3, 4, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.grid(True)
    
    # Generate completions on test set
    model.eval()
    test_data = next(iter(test_loader))[0][:8].to(device)  # Get 8 test samples
    
    # Generate completions with different context lengths
    context_lengths = [50, 100, 200]
    
    for i, context_len in enumerate(context_lengths):
        means, stds = generate_completion(model, test_data, context_len)
        
        # Plot examples
        for j in range(3):  # Show 3 examples per context length
            # Ground Truth
            plt.subplot(3, 4, 2 + i * 4 + j)
            if j == 0:
                plt.title(f'GT (Context: {context_len})')
            gt_img = test_data[j].cpu().numpy().reshape(28, 28)
            plt.imshow(gt_img, cmap='gray')
            plt.axis('off')
            
            # Add context boundary line
            context_pixel = context_len
            context_row = context_pixel // 28
            context_col = context_pixel % 28
            if context_row < 28:
                plt.axhline(y=context_row - 0.5, color='red', linewidth=2)
    
    # Detailed comparison for one example
    plt.figure(figsize=(18, 6))
    
    # Use context length of 100 for detailed visualization
    means, stds = generate_completion(model, test_data, CONTEXT_LENGTH)
    
    for i in range(6):  # Show 6 examples
        # Ground Truth
        plt.subplot(3, 6, i + 1)
        if i == 0:
            plt.title('Ground Truth')
        gt_img = test_data[i].cpu().numpy().reshape(28, 28)
        plt.imshow(gt_img, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        
        # Add context boundary
        context_row = CONTEXT_LENGTH // 28
        if context_row < 28:
            plt.axhline(y=context_row - 0.5, color='red', linewidth=2)
        
        # Reconstruction (Mean)
        plt.subplot(3, 6, i + 7)
        if i == 0:
            plt.title('Reconstruction (Mean)')
        recon_img = means[i].cpu().numpy().reshape(28, 28)
        plt.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        
        # Add context boundary
        if context_row < 28:
            plt.axhline(y=context_row - 0.5, color='red', linewidth=2)
        
        # Uncertainty (Standard Deviation)
        plt.subplot(3, 6, i + 13)
        if i == 0:
            plt.title('Uncertainty (Std Dev)')
        std_img = stds[i].cpu().numpy().reshape(28, 28)
        plt.imshow(std_img, cmap='hot', vmin=0, vmax=std_img.max())
        plt.axis('off')
        
        # Add context boundary
        if context_row < 28:
            plt.axhline(y=context_row - 0.5, color='red', linewidth=2)
    
    plt.suptitle(f'MNIST Pixel Completion (Context Length: {CONTEXT_LENGTH}, Red line shows context boundary)', fontsize=16)
    plt.tight_layout()
    
    # Show pixel-wise comparison for one example
    plt.figure(figsize=(15, 5))
    
    example_idx = 0
    gt_pixels = test_data[example_idx].cpu().numpy()
    pred_pixels = means[example_idx].cpu().numpy()
    std_pixels = stds[example_idx].cpu().numpy()
    
    plt.subplot(1, 3, 1)
    plt.plot(gt_pixels, label='Ground Truth', alpha=0.7)
    plt.plot(pred_pixels, label='Prediction', alpha=0.7)
    plt.axvline(x=CONTEXT_LENGTH, color='red', linestyle='--', label='Context End')
    plt.title('Pixel Values Comparison')
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(std_pixels, color='orange', label='Std Dev')
    plt.axvline(x=CONTEXT_LENGTH, color='red', linestyle='--', label='Context End')
    plt.title('Prediction Uncertainty')
    plt.xlabel('Pixel Index')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    error = np.abs(gt_pixels - pred_pixels)
    plt.plot(error, color='purple', label='Absolute Error')
    plt.axvline(x=CONTEXT_LENGTH, color='red', linestyle='--', label='Context End')
    plt.title('Reconstruction Error')
    plt.xlabel('Pixel Index')
    plt.ylabel('|GT - Pred|')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    context_error = np.mean(error[:CONTEXT_LENGTH])
    generation_error = np.mean(error[CONTEXT_LENGTH:])
    
    print(f"\nResults Summary:")
    print(f"Context Length: {CONTEXT_LENGTH}")
    print(f"Context Reconstruction Error: {context_error:.4f}")
    print(f"Generation Error: {generation_error:.4f}")
    print(f"Mean Uncertainty in Context: {np.mean(std_pixels[:CONTEXT_LENGTH]):.4f}")
    print(f"Mean Uncertainty in Generation: {np.mean(std_pixels[CONTEXT_LENGTH:]):.4f}")

# if __name__ == "__main__":
#%%
# Train the model
model, train_losses = train_model()


#%%
# Load test data for visualization
_, test_loader = load_data()

CONTEXT_LENGTH = 400  # Can be changed to 300 or other values < 784

# Visualize results
visualize_results(model, test_loader, train_losses)

# Save final model
torch.save(model.state_dict(), 'final_pixel_lstm_model.pth')
print("Model saved as 'final_pixel_lstm_model.pth'")
