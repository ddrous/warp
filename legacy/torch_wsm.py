# #%%
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
# import yaml
# from collections import OrderedDict
# import math
# import matplotlib as mpl
# import seaborn as sb
# sb.set_theme(context='poster', 
#              style='ticks',
#              font='sans-serif', 
#              font_scale=1, 
#              color_codes=True, 
#              rc={"lines.linewidth": 1})
# mpl.rcParams['savefig.facecolor'] = 'w'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
# plt.rcParams['savefig.bbox'] = 'tight'

# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Configuration
# config = {
#     'model': {
#         'root_width_size': 10,
#         'root_depth': 5,
#         'root_activation': 'swish',
#         'root_final_activation': 'tanh',
#         'init_state_layers': None,
#         'input_prev_data': False,
#         'model_type': 'wsm',
#         'nb_rnn_layers': 1,
#         'weights_lim': 1.0,
#         'time_as_channel': False,
#         'forcing_prob': 0.15,
#         'noise_theta_init': None,
#         'std_lower_bound': None
#     },
#     'training': {
#         'batch_size': 32,
#         'learning_rate': 1e-3,
#         'epochs': 1,
#         'context_length': 100
#     }
# }

# #%%
# def get_activation(name):
#     """Get activation function by name"""
#     activations = {
#         'relu': nn.ReLU(),
#         'tanh': nn.Tanh(),
#         'swish': nn.SiLU(),
#         'softplus': nn.Softplus(),
#         'identity': nn.Identity()
#     }
#     return activations.get(name, nn.ReLU())

# class RootMLP(nn.Module):
#     """Root network that takes weights as parameters and predicts mean and std"""
#     def __init__(self, data_size, width_size, depth, activation='swish', final_activation='tanh', 
#                  predict_uncertainty=True, input_prev_data=False):
#         super(RootMLP, self).__init__()
        
#         self.data_size = data_size
#         self.width_size = width_size
#         self.depth = depth
#         self.predict_uncertainty = predict_uncertainty
#         self.input_prev_data = input_prev_data
        
#         # Input dimension (time step)
#         input_dim = 1 + data_size if input_prev_data else 1
#         # Output dimension (mean + std if predicting uncertainty)
#         output_dim = 2 * data_size if predict_uncertainty else data_size
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # Build MLP structure to calculate total parameter count
#         layers = []
#         prev_dim = input_dim
        
#         for i in range(depth):
#             layers.append(nn.Linear(prev_dim, width_size))
#             layers.append(get_activation(activation))
#             prev_dim = width_size
            
#         layers.append(nn.Linear(prev_dim, output_dim))
        
#         self.mlp_template = nn.Sequential(*layers)
        
#         # Calculate total number of parameters
#         self.total_params = sum(p.numel() for p in self.mlp_template.parameters())
        
#         # Final activation
#         if final_activation == 'tanh':
#             self.final_activation = nn.Tanh()
#         elif final_activation == 'identity':
#             self.final_activation = nn.Identity()
#         else:
#             self.final_activation = nn.Tanh()
    
#     def unflatten_weights(self, flat_weights):
#         """Convert flat weight vector back to MLP parameters"""
#         params = {}
#         idx = 0
        
#         for name, param in self.mlp_template.named_parameters():
#             param_size = param.numel()
#             param_shape = param.shape
#             params[name] = flat_weights[idx:idx+param_size].view(param_shape)
#             idx += param_size
            
#         return params
    
#     def forward_with_weights(self, x, flat_weights, std_lower_bound=None):
#         """Forward pass using provided weights"""
#         params = self.unflatten_weights(flat_weights)
        
#         # Manual forward pass through the network
#         y = x
#         layer_idx = 0
        
#         for i, layer in enumerate(self.mlp_template):
#             if isinstance(layer, nn.Linear):
#                 weight_name = f"{layer_idx}.weight"
#                 bias_name = f"{layer_idx}.bias"
#                 y = F.linear(y, params[weight_name], params[bias_name])
#                 layer_idx += 2  # weight and bias
#             else:
#                 # Activation function
#                 y = layer(y)
        
#         if not self.predict_uncertainty:
#             return self.final_activation(y)
#         else:
#             # Split into mean and log variance
#             mean, logvar = torch.chunk(y, 2, dim=-1)
#             mean = self.final_activation(mean)
            
#             # Convert logvar to std TODO
#             logvar = torch.clamp(logvar, -4, 4)
#             std = torch.exp(logvar)
            
#             if std_lower_bound is not None:
#                 std = torch.clamp(std, min=std_lower_bound)
            
#             return torch.cat([mean, std], dim=-1)

# class WeightSpaceModel(nn.Module):
#     """Weight Space Model with Linear RNN transition"""
#     def __init__(self, config):
#         super(WeightSpaceModel, self).__init__()
        
#         self.config = config['model']
#         self.data_size = 1  # Single pixel
        
#         # Root MLP
#         self.root_mlp = RootMLP(
#             data_size=self.data_size,
#             width_size=self.config['root_width_size'],
#             depth=self.config['root_depth'],
#             activation=self.config['root_activation'],
#             final_activation=self.config['root_final_activation'],
#             predict_uncertainty=True,
#             input_prev_data=self.config['input_prev_data']
#         )
        
#         self.latent_size = self.root_mlp.total_params
        
#         # Linear RNN matrices
#         self.A = nn.Parameter(torch.eye(self.latent_size))  # Identity initialization
        
#         # B matrix for input integration
#         input_dim = self.data_size + 1 if self.config['time_as_channel'] else self.data_size
#         self.B = nn.Parameter(torch.zeros(self.latent_size, input_dim))
        
#         # Initial state
#         if self.config['init_state_layers'] is None:
#             self.theta_init = nn.Parameter(torch.randn(self.latent_size) * 0.1)
#         else:
#             # Gradual MLP for initial state (simplified version)
#             self.theta_init_mlp = nn.Sequential(
#                 nn.Linear(self.data_size, self.latent_size // 2),
#                 get_activation(self.config['root_activation']),
#                 nn.Linear(self.latent_size // 2, self.latent_size)
#             )
        
#         self.forcing_prob = self.config['forcing_prob']
#         self.weights_lim = self.config['weights_lim']
#         self.time_as_channel = self.config['time_as_channel']
#         self.input_prev_data = self.config['input_prev_data']
#         self.std_lower_bound = self.config['std_lower_bound']
#         self.noise_theta_init = self.config['noise_theta_init']

#     def forward(self, xs, inference_start=None):
#         """
#         Forward pass with stochastic autoregressive generation
#         xs: (batch_size, seq_len) - flattened pixel sequences
#         inference_start: step to switch from context to generation
#         """
#         batch_size, seq_len = xs.shape
#         device = xs.device
        
#         # Initialize outputs
#         means = torch.zeros_like(xs)
#         stds = torch.zeros_like(xs)
        
#         for b in range(batch_size):
#             # Initialize theta for this sequence
#             if self.config['init_state_layers'] is None:
#                 theta = self.theta_init.clone()
#             else:
#                 theta = self.theta_init_mlp(xs[b, 0:1])
#                 theta = theta.squeeze(0)
            
#             if self.noise_theta_init is not None:
#                 theta += torch.randn_like(theta) * self.noise_theta_init
            
#             # Initialize first prediction
#             x_prev = torch.zeros(1, device=device)
#             t_prev = 0.0
            
#             # Initial prediction for first step
#             time_input = torch.tensor([0.0], device=device).unsqueeze(0)
#             if self.input_prev_data:
#                 root_input = torch.cat([time_input, x_prev.unsqueeze(0)], dim=-1)
#             else:
#                 root_input = time_input
            
#             mu_sigma = self.root_mlp.forward_with_weights(root_input, theta, self.std_lower_bound)
#             x_mu_sigma = mu_sigma.squeeze(0)
            
#             for t in range(seq_len):
#                 t_curr = t / seq_len
#                 delta_t = t_curr - t_prev
                
#                 # Split mean and std
#                 x_hat_mean, x_hat_std = torch.chunk(x_mu_sigma, 2, dim=-1)
                
#                 # Sample from distribution
#                 x_hat = torch.randn_like(x_hat_mean) * x_hat_std + x_hat_mean
                
#                 # Determine which input to use
#                 if inference_start is not None:
#                     # Use ground truth up to inference_start, then use predictions
#                     use_ground_truth = t < inference_start
#                     x_t = xs[b, t:t+1] if use_ground_truth else x_hat
#                 else:
#                     # Training: use teacher forcing with probability
#                     use_ground_truth = torch.rand(1) < self.forcing_prob
#                     x_t = xs[b, t:t+1] if use_ground_truth else x_hat
                
#                 # Store predictions
#                 means[b, t] = x_hat_mean.item()
#                 stds[b, t] = x_hat_std.item()
                
#                 # Update hidden state (Linear RNN)
#                 if self.time_as_channel:
#                     x_input = torch.cat([x_t, torch.tensor([t_curr], device=device)], dim=-1)
#                     x_prev_input = torch.cat([x_prev, torch.tensor([t_prev], device=device)], dim=-1)
#                 else:
#                     x_input = x_t
#                     x_prev_input = x_prev
                
#                 # Linear RNN update: θ_{t+1} = A θ_t + B (x_t - x_{t-1})
#                 theta = self.A @ theta + self.B @ (x_input - x_prev_input).unsqueeze(-1).squeeze(-1)
                
#                 if self.weights_lim is not None:
#                     theta = torch.clamp(theta, -self.weights_lim, self.weights_lim)
                
#                 # Prepare for next iteration
#                 x_prev = x_t.clone()
#                 t_prev = t_curr
                
#                 # Get next prediction if not last step
#                 if t < seq_len - 1:
#                     next_time = (t + 1) / seq_len
#                     time_input = torch.tensor([next_time], device=device).unsqueeze(0)
#                     if self.input_prev_data:
#                         root_input = torch.cat([time_input, x_t.unsqueeze(0)], dim=-1)
#                     else:
#                         root_input = time_input
                    
#                     mu_sigma = self.root_mlp.forward_with_weights(root_input, theta, self.std_lower_bound)
#                     x_mu_sigma = mu_sigma.squeeze(0)
        
#         return means, stds

# def negative_log_likelihood_loss(mean, std, target):
#     """Compute negative log likelihood loss for Gaussian distribution"""
#     variance = std ** 2
#     nll = 0.5 * torch.log(2 * np.pi * variance) + (target - mean) ** 2 / (2 * variance)
#     return nll.mean()

# def load_data(batch_size):
#     """Load MNIST dataset"""
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),  # Uncomment if
#         transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784 pixels
#     ])
    
#     train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
#                                              download=True, transform=transform)
#     test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
#                                             download=True, transform=transform)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, test_loader

# def train_model(config):
#     """Main training loop"""
#     # Load data
#     train_loader, test_loader = load_data(config['training']['batch_size'])
    
#     # Initialize model
#     model = WeightSpaceModel(config).to(device)
    
#     # Optimizer
#     optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
#     # Training history
#     train_losses = []
    
#     print("Starting WSM training...")
#     print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
#     print(f"Root MLP has {model.root_mlp.total_params} parameters (latent size)")
    
#     for epoch in range(config['training']['epochs']):
#         model.train()
#         epoch_loss = 0
#         num_batches = 0
        
#         for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
#             data = data.to(device)
            
#             # Forward pass
#             means, stds = model(data)
            
#             # Compute loss
#             loss = negative_log_likelihood_loss(means, stds, data)
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             epoch_loss += loss.item()
#             num_batches += 1
        
#         avg_loss = epoch_loss / num_batches
#         train_losses.append(avg_loss)
#         scheduler.step()
        
#         print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
#         # Save checkpoint every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_loss,
#                 'config': config
#             }, f'wsm_checkpoint_epoch_{epoch+1}.pth')
    
#     return model, train_losses

# def generate_completions(model, test_data, context_length):
#     """Generate completions given context"""
#     model.eval()
#     with torch.no_grad():
#         means, stds = model(test_data, inference_start=context_length)
#     return means, stds

# def visualize_results(model, test_loader, train_losses, config):
#     """Visualize training progress and generate completions"""
#     context_length = config['training']['context_length']
    
#     # Plot training loss
#     plt.figure(figsize=(15, 12))
    
#     # Loss curve
#     plt.subplot(3, 4, 1)
#     plt.plot(train_losses)
#     plt.title('WSM Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Negative Log Likelihood')
#     plt.grid(True)
    
#     # Generate completions on test set
#     model.eval()
#     test_data = next(iter(test_loader))[0][:8].to(device)  # Get 8 test samples
    
#     # Generate completions with different context lengths
#     context_lengths = [50, 100, 200]
    
#     for i, context_len in enumerate(context_lengths):
#         means, stds = generate_completions(model, test_data, context_len)
        
#         # Plot examples
#         for j in range(3):  # Show 3 examples per context length
#             # Ground Truth
#             plt.subplot(3, 4, 2 + i * 4 + j)
#             if j == 0:
#                 plt.title(f'GT (Context: {context_len})')
#             gt_img = test_data[j].cpu().numpy().reshape(28, 28)
#             plt.imshow(gt_img, cmap='gray')
#             plt.axis('off')
            
#             # Add context boundary line
#             context_pixel = context_len
#             context_row = context_pixel // 28
#             context_col = context_pixel % 28
#             if context_row < 28:
#                 plt.axhline(y=context_row - 0.5, color='red', linewidth=2)
    
#     # Detailed comparison for one example
#     plt.figure(figsize=(18, 6))
    
#     # Use specified context length for detailed visualization
#     means, stds = generate_completions(model, test_data, context_length)
    
#     for i in range(6):  # Show 6 examples
#         # Ground Truth
#         plt.subplot(3, 6, i + 1)
#         if i == 0:
#             plt.title('Ground Truth')
#         gt_img = test_data[i].cpu().numpy().reshape(28, 28)
#         plt.imshow(gt_img, cmap='gray', vmin=0, vmax=1)
#         plt.axis('off')
        
#         # Add context boundary
#         context_row = context_length // 28
#         if context_row < 28:
#             plt.axhline(y=context_row - 0.5, color='red', linewidth=2)
        
#         # Reconstruction (Mean)
#         plt.subplot(3, 6, i + 7)
#         if i == 0:
#             plt.title('Reconstruction (Mean)')
#         recon_img = means[i].cpu().numpy().reshape(28, 28)
#         plt.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
#         plt.axis('off')
        
#         # Add context boundary
#         if context_row < 28:
#             plt.axhline(y=context_row - 0.5, color='red', linewidth=2)
        
#         # Uncertainty (Standard Deviation)
#         plt.subplot(3, 6, i + 13)
#         if i == 0:
#             plt.title('Uncertainty (Std Dev)')
#         std_img = stds[i].cpu().numpy().reshape(28, 28)
#         plt.imshow(std_img, cmap='hot', vmin=0, vmax=std_img.max())
#         plt.axis('off')
        
#         # Add context boundary
#         if context_row < 28:
#             plt.axhline(y=context_row - 0.5, color='red', linewidth=2)
    
#     plt.suptitle(f'WSM MNIST Pixel Completion (Context Length: {context_length}, Red line shows context boundary)', fontsize=16)
#     plt.tight_layout()
    
#     # Show pixel-wise comparison for one example
#     plt.figure(figsize=(15, 5))
    
#     example_idx = 0
#     gt_pixels = test_data[example_idx].cpu().numpy()
#     pred_pixels = means[example_idx].cpu().numpy()
#     std_pixels = stds[example_idx].cpu().numpy()
    
#     plt.subplot(1, 3, 1)
#     plt.plot(gt_pixels, label='Ground Truth', alpha=0.7)
#     plt.plot(pred_pixels, label='WSM Prediction', alpha=0.7)
#     plt.axvline(x=context_length, color='red', linestyle='--', label='Context End')
#     plt.title('Pixel Values Comparison')
#     plt.xlabel('Pixel Index')
#     plt.ylabel('Intensity')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(1, 3, 2)
#     plt.plot(std_pixels, color='orange', label='Std Dev')
#     plt.axvline(x=context_length, color='red', linestyle='--', label='Context End')
#     plt.title('Prediction Uncertainty')
#     plt.xlabel('Pixel Index')
#     plt.ylabel('Standard Deviation')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(1, 3, 3)
#     error = np.abs(gt_pixels - pred_pixels)
#     plt.plot(error, color='purple', label='Absolute Error')
#     plt.axvline(x=context_length, color='red', linestyle='--', label='Context End')
#     plt.title('Reconstruction Error')
#     plt.xlabel('Pixel Index')
#     plt.ylabel('|GT - Pred|')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print statistics
#     context_error = np.mean(error[:context_length])
#     generation_error = np.mean(error[context_length:])
    
#     print(f"\nWSM Results Summary:")
#     print(f"Context Length: {context_length}")
#     print(f"Context Reconstruction Error: {context_error:.4f}")
#     print(f"Generation Error: {generation_error:.4f}")
#     print(f"Mean Uncertainty in Context: {np.mean(std_pixels[:context_length]):.4f}")
#     print(f"Mean Uncertainty in Generation: {np.mean(std_pixels[context_length:]):.4f}")
#     print(f"Root MLP Parameters (Latent Size): {model.root_mlp.total_params}")
#     print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())}")

# # if __name__ == "__main__":
# #%%
# # Train the model
# model, train_losses = train_model(config)

# #%%
# # Load test data for visualization
# _, test_loader = load_data(config['training']['batch_size'])

# # Visualize results
# visualize_results(model, test_loader, train_losses, config)

# # Save final model
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'config': config,
#     'train_losses': train_losses
# }, 'final_wsm_model.pth')
# print("Model saved as 'final_wsm_model.pth'")









# #%%
# #%%
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
# import yaml
# from collections import OrderedDict
# import math
# import matplotlib as mpl
# import seaborn as sb
# sb.set_theme(context='poster', 
#              style='ticks',
#              font='sans-serif', 
#              font_scale=1, 
#              color_codes=True, 
#              rc={"lines.linewidth": 1})
# mpl.rcParams['savefig.facecolor'] = 'w'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
# plt.rcParams['savefig.bbox'] = 'tight'

# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Configuration
# config = {
#     'model': {
#         'root_width_size': 10,
#         'root_depth': 5,
#         'root_activation': 'swish',
#         'root_final_activation': 'tanh',
#         'init_state_layers': None,
#         'input_prev_data': False,
#         'model_type': 'wsm',
#         'nb_rnn_layers': 1,
#         'weights_lim': 1.0,
#         'time_as_channel': False,
#         'forcing_prob': 0.15,
#         'noise_theta_init': None,
#         'std_lower_bound': None
#     },
#     'training': {
#         'batch_size': 32,
#         'learning_rate': 1e-3,
#         'epochs': 1, # Set to a higher number for actual training
#         'context_length': 100
#     }
# }

# #%%
# def get_activation(name):
#     """Get activation function by name"""
#     activations = {
#         'relu': nn.ReLU(),
#         'tanh': nn.Tanh(),
#         'swish': nn.SiLU(),
#         'softplus': nn.Softplus(),
#         'identity': nn.Identity()
#     }
#     return activations.get(name, nn.ReLU())

# class RootMLP(nn.Module):
#     """Root network that takes weights as parameters and predicts mean and std"""
#     def __init__(self, data_size, width_size, depth, activation='swish', final_activation='tanh', 
#                  predict_uncertainty=True, input_prev_data=False):
#         super(RootMLP, self).__init__()
        
#         self.data_size = data_size
#         self.width_size = width_size
#         self.depth = depth
#         self.predict_uncertainty = predict_uncertainty
#         self.input_prev_data = input_prev_data
        
#         # Input dimension (time step)
#         input_dim = 1 + data_size if input_prev_data else 1
#         # Output dimension (mean + std if predicting uncertainty)
#         output_dim = 2 * data_size if predict_uncertainty else data_size
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # Build MLP structure to calculate total parameter count
#         layers = []
#         prev_dim = input_dim
        
#         for i in range(depth):
#             layers.append(nn.Linear(prev_dim, width_size))
#             layers.append(get_activation(activation))
#             prev_dim = width_size
            
#         layers.append(nn.Linear(prev_dim, output_dim))
        
#         self.mlp_template = nn.Sequential(*layers)
        
#         # Calculate total number of parameters
#         self.total_params = sum(p.numel() for p in self.mlp_template.parameters())
        
#         # Final activation
#         if final_activation == 'tanh':
#             self.final_activation = nn.Tanh()
#         elif final_activation == 'identity':
#             self.final_activation = nn.Identity()
#         else:
#             self.final_activation = nn.Tanh()
    
#     def unflatten_weights(self, flat_weights):
#         """Convert flat weight vector back to MLP parameters"""
#         params = {}
#         idx = 0
        
#         for name, param in self.mlp_template.named_parameters():
#             param_size = param.numel()
#             param_shape = param.shape
#             params[name] = flat_weights[idx:idx+param_size].view(param_shape)
#             idx += param_size
            
#         return params
    
#     def forward_with_weights(self, x, flat_weights, std_lower_bound=None):
#         """Forward pass using provided weights"""
#         params = self.unflatten_weights(flat_weights)
        
#         # Manual forward pass through the network
#         y = x
        
#         for i, layer in enumerate(self.mlp_template):
#             if isinstance(layer, nn.Linear):
#                 # FIX: Simplified the logic for accessing weights. Using the layer's direct
#                 # index `i` from enumerate is cleaner and more robust than maintaining a separate counter.
#                 weight_name = f"{i}.weight"
#                 bias_name = f"{i}.bias"
#                 y = F.linear(y, params[weight_name], params[bias_name])
#             else:
#                 # Activation function
#                 y = layer(y)
        
#         if not self.predict_uncertainty:
#             return self.final_activation(y)
#         else:
#             # Split into mean and log variance
#             mean, logvar = torch.chunk(y, 2, dim=-1)
#             mean = self.final_activation(mean)
            
#             # Convert logvar to std
#             logvar = torch.clamp(logvar, -4, 4)
#             std = torch.exp(0.5 * logvar) # Use 0.5*logvar for std deviation
            
#             if std_lower_bound is not None:
#                 std = torch.clamp(std, min=std_lower_bound)
            
#             return torch.cat([mean, std], dim=-1)

# class WeightSpaceModel(nn.Module):
#     """Weight Space Model with Linear RNN transition"""
#     def __init__(self, config):
#         super(WeightSpaceModel, self).__init__()
        
#         self.config = config['model']
#         self.data_size = 1  # Single pixel
        
#         # Root MLP
#         self.root_mlp = RootMLP(
#             data_size=self.data_size,
#             width_size=self.config['root_width_size'],
#             depth=self.config['root_depth'],
#             activation=self.config['root_activation'],
#             final_activation=self.config['root_final_activation'],
#             predict_uncertainty=True,
#             input_prev_data=self.config['input_prev_data']
#         )
        
#         self.latent_size = self.root_mlp.total_params
        
#         # Linear RNN matrices
#         self.A = nn.Parameter(torch.eye(self.latent_size))
        
#         input_dim = self.data_size + 1 if self.config['time_as_channel'] else self.data_size
#         self.B = nn.Parameter(torch.zeros(self.latent_size, input_dim))
#         nn.init.xavier_uniform_(self.B) # Better initialization
        
#         # Initial state
#         if self.config['init_state_layers'] is None:
#             self.theta_init = nn.Parameter(torch.randn(self.latent_size) * 0.1)
#         else:
#             self.theta_init_mlp = nn.Sequential(
#                 nn.Linear(self.data_size, self.latent_size // 2),
#                 get_activation(self.config['root_activation']),
#                 nn.Linear(self.latent_size // 2, self.latent_size)
#             )
        
#         self.forcing_prob = self.config['forcing_prob']
#         self.weights_lim = self.config['weights_lim']
#         self.time_as_channel = self.config['time_as_channel']
#         self.input_prev_data = self.config['input_prev_data']
#         self.std_lower_bound = self.config['std_lower_bound']
#         self.noise_theta_init = self.config['noise_theta_init']

#     def forward(self, xs, inference_start=None):
#         batch_size, seq_len = xs.shape
#         device = xs.device
        
#         means_list = []
#         stds_list = []
        
#         for b in range(batch_size):
#             # Initialize theta for this sequence
#             if self.config['init_state_layers'] is None:
#                 theta = self.theta_init.clone()
#             else:
#                 theta = self.theta_init_mlp(xs[b, 0:1]).squeeze(0)
            
#             if self.noise_theta_init is not None:
#                 theta += torch.randn_like(theta) * self.noise_theta_init
            
#             x_prev = torch.zeros(1, device=device)
#             t_prev = 0.0
            
#             # Initial prediction for the first step
#             time_input = torch.tensor([0.0], device=device).unsqueeze(0)
#             if self.input_prev_data:
#                 root_input = torch.cat([time_input, x_prev.unsqueeze(0)], dim=-1)
#             else:
#                 root_input = time_input
            
#             mu_sigma = self.root_mlp.forward_with_weights(root_input, theta, self.std_lower_bound)
            
#             seq_means = []
#             seq_stds = []
            
#             for t in range(seq_len):
#                 t_curr = t / seq_len
                
#                 # Split mean and std from the prediction of the previous step
#                 x_hat_mean, x_hat_std = torch.chunk(mu_sigma.squeeze(0), 2, dim=-1)
                
#                 # FIX: Store the full tensors, not their .item() values.
#                 # This preserves the computation graph for backpropagation.
#                 seq_means.append(x_hat_mean)
#                 seq_stds.append(x_hat_std)
                
#                 # Sample from the predicted distribution
#                 x_hat = torch.randn_like(x_hat_mean) * x_hat_std + x_hat_mean
                
#                 # Determine which input to use (teacher forcing or autoregressive)
#                 if inference_start is not None:
#                     use_ground_truth = t < inference_start
#                 else:
#                     use_ground_truth = torch.rand(1).item() > self.forcing_prob
                
#                 x_t = xs[b, t:t+1] if use_ground_truth else x_hat
                
#                 # Prepare inputs for the RNN update
#                 if self.time_as_channel:
#                     x_input = torch.cat([x_t, torch.tensor([t_curr], device=device)])
#                     x_prev_input = torch.cat([x_prev, torch.tensor([t_prev], device=device)])
#                 else:
#                     x_input = x_t
#                     x_prev_input = x_prev
                
#                 # Update hidden state (Linear RNN)
#                 update_term = self.B @ (x_input - x_prev_input)
#                 theta = self.A @ theta + update_term
                
#                 if self.weights_lim is not None:
#                     theta = torch.clamp(theta, -self.weights_lim, self.weights_lim)
                
#                 # Prepare for the next iteration
#                 x_prev = x_t.clone()
#                 t_prev = t_curr
                
#                 # Generate the prediction for the *next* step
#                 if t < seq_len - 1:
#                     next_time = (t + 1) / seq_len
#                     time_input = torch.tensor([next_time], device=device).unsqueeze(0)
#                     if self.input_prev_data:
#                         root_input = torch.cat([time_input, x_t.unsqueeze(0)], dim=-1)
#                     else:
#                         root_input = time_input
                    
#                     mu_sigma = self.root_mlp.forward_with_weights(root_input, theta, self.std_lower_bound)
            
#             means_list.append(torch.cat(seq_means))
#             stds_list.append(torch.cat(seq_stds))
        
#         # Stack the lists of tensors to create the final batch of outputs
#         means = torch.stack(means_list)
#         stds = torch.stack(stds_list)
        
#         return means, stds

# def negative_log_likelihood_loss(mean, std, target):
#     """Compute negative log likelihood loss for Gaussian distribution"""
#     # Add a small epsilon to std to prevent division by zero and log(0)
#     std = std + 1e-6
#     variance = std.pow(2)
#     nll = 0.5 * torch.log(2 * np.pi * variance) + (target - mean).pow(2) / (2 * variance)
#     return nll.mean()

# def load_data(batch_size):
#     """Load MNIST dataset"""
#     # FIX: Uncommented the normalization transform. The model uses a tanh activation,
#     # which outputs values in [-1, 1]. The input data should be in the same range.
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#         transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784 pixels
#     ])
    
#     train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
#                                              download=True, transform=transform)
#     test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
#                                             download=True, transform=transform)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
#     return train_loader, test_loader

# def train_model(config):
#     """Main training loop"""
#     train_loader, _ = load_data(config['training']['batch_size'])
#     model = WeightSpaceModel(config).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
#     train_losses = []
    
#     print("Starting WSM training...")
#     print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
#     print(f"Root MLP has {model.root_mlp.total_params} parameters (latent size)")
    
#     for epoch in range(config['training']['epochs']):
#         model.train()
#         epoch_loss = 0
        
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
#         for data, _ in progress_bar:
#             data = data.to(device)
#             optimizer.zero_grad()
#             means, stds = model(data)
#             loss = negative_log_likelihood_loss(means, stds, data)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             epoch_loss += loss.item()
#             progress_bar.set_postfix(loss=loss.item())
        
#         avg_loss = epoch_loss / len(train_loader)
#         train_losses.append(avg_loss)
#         scheduler.step()
        
#         print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
#         if (epoch + 1) % 10 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_loss,
#                 'config': config
#             }, f'wsm_checkpoint_epoch_{epoch+1}.pth')
    
#     return model, train_losses

# def generate_completions(model, test_data, context_length):
#     """Generate completions given context"""
#     model.eval()
#     with torch.no_grad():
#         means, stds = model(test_data, inference_start=context_length)
#     return means, stds

# def visualize_results(model, test_loader, train_losses, config):
#     """Visualize training progress and generate completions"""
#     context_length = config['training']['context_length']
    
#     plt.figure(figsize=(15, 5))
#     plt.plot(train_losses)
#     plt.title('WSM Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Negative Log Likelihood')
#     plt.grid(True)
#     plt.show()
    
#     model.eval()
#     test_data = next(iter(test_loader))[0][:6].to(device)
#     means, stds = generate_completions(model, test_data, context_length)
    
#     plt.figure(figsize=(18, 9))
    
#     for i in range(6):
#         # Ground Truth
#         plt.subplot(3, 6, i + 1)
#         if i == 0: plt.title('Ground Truth')
#         gt_img = test_data[i].cpu().numpy().reshape(28, 28)
#         # FIX: Adjusted vmin/vmax for normalized data
#         plt.imshow(gt_img, cmap='gray', vmin=-1, vmax=1)
#         plt.axis('off')
#         context_row = context_length // 28
#         if context_row < 28: plt.axhline(y=context_row - 0.5, color='red', linewidth=1)

#         # Reconstruction (Mean)
#         plt.subplot(3, 6, i + 7)
#         if i == 0: plt.title('Reconstruction (Mean)')
#         recon_img = means[i].cpu().numpy().reshape(28, 28)
#         plt.imshow(recon_img, cmap='gray', vmin=-1, vmax=1)
#         plt.axis('off')
#         if context_row < 28: plt.axhline(y=context_row - 0.5, color='red', linewidth=1)

#         # Uncertainty (Standard Deviation)
#         plt.subplot(3, 6, i + 13)
#         if i == 0: plt.title('Uncertainty (Std Dev)')
#         std_img = stds[i].cpu().numpy().reshape(28, 28)
#         plt.imshow(std_img, cmap='hot')
#         plt.axis('off')
#         if context_row < 28: plt.axhline(y=context_row - 0.5, color='red', linewidth=1)
    
#     plt.suptitle(f'WSM MNIST Pixel Completion (Context: {context_length})', fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()
    
#     # Show pixel-wise comparison for one example
#     plt.figure(figsize=(15, 5))
    
#     example_idx = 0
#     gt_pixels = test_data[example_idx].cpu().numpy()
#     pred_pixels = means[example_idx].cpu().numpy()
#     std_pixels = stds[example_idx].cpu().numpy()
    
#     plt.plot(gt_pixels, label='Ground Truth', alpha=0.7)
#     plt.plot(pred_pixels, label='WSM Prediction', alpha=0.7)
#     plt.fill_between(range(len(pred_pixels)), pred_pixels - std_pixels, pred_pixels + std_pixels, color='orange', alpha=0.3, label='±1 Std Dev')
#     plt.axvline(x=context_length, color='red', linestyle='--', label='Context End')
#     plt.title('Pixel Values Comparison (Example 0)')
#     plt.xlabel('Pixel Index')
#     plt.ylabel('Intensity')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# #%%
# # Train the model
# model, train_losses = train_model(config)

# #%%
# # Load test data for visualization
# _, test_loader = load_data(config['training']['batch_size'])

# # Visualize results
# visualize_results(model, test_loader, train_losses, config)

# # Save final model
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'config': config,
#     'train_losses': train_losses
# }, 'final_wsm_model.pth')
# print("\nModel saved as 'final_wsm_model.pth'")






















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
import yaml
from collections import OrderedDict
import math
import matplotlib as mpl
import seaborn as sb
sb.set_theme(context='poster', 
             style='ticks',
             font='sans-serif', 
             font_scale=1, 
             color_codes=True, 
             rc={"lines.linewidth": 1})
mpl.rcParams['savefig.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['savefig.bbox'] = 'tight'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
config = {
    'model': {
        'root_width_size': 10,
        'root_depth': 5,
        'root_activation': 'swish',
        'root_final_activation': 'tanh',
        'init_state_layers': None,
        'input_prev_data': False,
        'model_type': 'wsm',
        'nb_rnn_layers': 1,
        'weights_lim': 1.0,
        'time_as_channel': False,
        'forcing_prob': 0.15,
        'noise_theta_init': None,
        'std_lower_bound': None
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 1, # Set to a higher number for actual training
        'context_length': 100
    }
}

#%%
def get_activation(name):
    """Get activation function by name"""
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'swish': nn.SiLU(),
        'softplus': nn.Softplus(),
        'identity': nn.Identity()
    }
    return activations.get(name, nn.ReLU())

class RootMLP(nn.Module):
    """Root network that takes weights as parameters and predicts mean and std"""
    def __init__(self, data_size, width_size, depth, activation='swish', final_activation='tanh', 
                 predict_uncertainty=True, input_prev_data=False):
        super(RootMLP, self).__init__()
        
        self.data_size = data_size
        self.width_size = width_size
        self.depth = depth
        self.predict_uncertainty = predict_uncertainty
        self.input_prev_data = input_prev_data
        
        input_dim = 1 + data_size if input_prev_data else 1
        output_dim = 2 * data_size if predict_uncertainty else data_size
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        for i in range(depth):
            layers.append(nn.Linear(prev_dim, width_size))
            layers.append(get_activation(activation))
            prev_dim = width_size
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp_template = nn.Sequential(*layers)
        self.total_params = sum(p.numel() for p in self.mlp_template.parameters())
        self.final_activation = get_activation(final_activation)
    
    def unflatten_weights(self, flat_weights):
        """
        OPTIMIZATION: Convert flat weight vector (or batch of vectors) back to MLP parameters.
        This now handles batches of weights for vectorized computation.
        """
        # flat_weights: (batch_size, total_params) or (total_params,)
        is_batched = flat_weights.dim() == 2
        if not is_batched:
            flat_weights = flat_weights.unsqueeze(0)
        
        batch_size = flat_weights.shape[0]
        params = {}
        idx = 0
        
        for name, param in self.mlp_template.named_parameters():
            param_size = param.numel()
            param_shape = param.shape
            # New shape will be (batch_size, *param_shape)
            params[name] = flat_weights[:, idx:idx+param_size].view(batch_size, *param_shape)
            idx += param_size
            
        return params, is_batched
    
    def forward_with_weights(self, x, flat_weights, std_lower_bound=None):
        """
        OPTIMIZATION: Forward pass using provided weights, vectorized to handle batches.
        x: (batch_size, input_dim)
        flat_weights: (batch_size, total_params)
        """
        params, is_batched = self.unflatten_weights(flat_weights)
        if not is_batched:
            x = x.unsqueeze(0)
        
        y = x
        
        for i, layer in enumerate(self.mlp_template):
            if isinstance(layer, nn.Linear):
                weight_name = f"{i}.weight"
                bias_name = f"{i}.bias"
                W = params[weight_name] # (batch_size, out_features, in_features)
                b = params[bias_name]   # (batch_size, out_features)
                
                # Batched matrix multiplication: y = x @ W.T + b
                y = torch.einsum('bi,boi->bo', y, W) + b
            else:
                y = layer(y)
        
        if not self.predict_uncertainty:
            y = self.final_activation(y)
        else:
            mean, logvar = torch.chunk(y, 2, dim=-1)
            mean = self.final_activation(mean)
            logvar = torch.clamp(logvar, -4, 4)
            std = torch.exp(0.5 * logvar)
            if std_lower_bound is not None:
                std = torch.clamp(std, min=std_lower_bound)
            y = torch.cat([mean, std], dim=-1)

        if not is_batched:
            y = y.squeeze(0)
            
        return y

class WeightSpaceModel(nn.Module):
    """Weight Space Model with Linear RNN transition"""
    def __init__(self, config):
        super(WeightSpaceModel, self).__init__()
        
        self.config = config['model']
        self.data_size = 1
        
        self.root_mlp = RootMLP(
            data_size=self.data_size,
            width_size=self.config['root_width_size'],
            depth=self.config['root_depth'],
            activation=self.config['root_activation'],
            final_activation=self.config['root_final_activation'],
            predict_uncertainty=True,
            input_prev_data=self.config['input_prev_data']
        )
        
        self.latent_size = self.root_mlp.total_params
        self.A = nn.Parameter(torch.eye(self.latent_size))
        input_dim = self.data_size + 1 if self.config['time_as_channel'] else self.data_size
        self.B = nn.Parameter(torch.zeros(self.latent_size, input_dim))
        nn.init.xavier_uniform_(self.B)
        
        if self.config['init_state_layers'] is None:
            self.theta_init = nn.Parameter(torch.randn(self.latent_size) * 0.1)
        else:
            self.theta_init_mlp = nn.Sequential(
                nn.Linear(self.data_size, self.latent_size // 2),
                get_activation(self.config['root_activation']),
                nn.Linear(self.latent_size // 2, self.latent_size)
            )
        
        self.forcing_prob = self.config['forcing_prob']
        self.weights_lim = self.config['weights_lim']
        self.time_as_channel = self.config['time_as_channel']
        self.input_prev_data = self.config['input_prev_data']
        self.std_lower_bound = self.config['std_lower_bound']
        self.noise_theta_init = self.config['noise_theta_init']

    def forward(self, xs, inference_start=None):
        """
        OPTIMIZATION: Vectorized forward pass.
        This version processes the entire batch at once, removing the slow Python loop over batches.
        """
        batch_size, seq_len = xs.shape
        device = xs.device
        
        if self.config['init_state_layers'] is None:
            theta = self.theta_init.unsqueeze(0).expand(batch_size, -1)
        else:
            theta = self.theta_init_mlp(xs[:, 0:1])
        
        if self.noise_theta_init is not None:
            theta += torch.randn_like(theta) * self.noise_theta_init
        
        x_prev = torch.zeros(batch_size, 1, device=device)
        t_prev = 0.0
        
        time_input = torch.full((batch_size, 1), 0.0, device=device)
        root_input = torch.cat([time_input, x_prev], dim=-1) if self.input_prev_data else time_input
        mu_sigma = self.root_mlp.forward_with_weights(root_input, theta, self.std_lower_bound)
        
        means_list, stds_list = [], []
        
        for t in range(seq_len):
            t_curr = t / seq_len
            x_hat_mean, x_hat_std = torch.chunk(mu_sigma, 2, dim=-1)
            
            means_list.append(x_hat_mean)
            stds_list.append(x_hat_std)
            
            x_hat = torch.randn_like(x_hat_mean) * x_hat_std + x_hat_mean
            
            if inference_start is not None:
                x_t = xs[:, t:t+1] if t < inference_start else x_hat
            else:
                use_teacher_forcing = torch.rand(batch_size, 1, device=device) < self.forcing_prob
                x_t = torch.where(use_teacher_forcing, xs[:, t:t+1], x_hat)
            
            if self.time_as_channel:
                t_curr_tensor = torch.full((batch_size, 1), t_curr, device=device)
                t_prev_tensor = torch.full((batch_size, 1), t_prev, device=device)
                x_input = torch.cat([x_t, t_curr_tensor], dim=1)
                x_prev_input = torch.cat([x_prev, t_prev_tensor], dim=1)
            else:
                x_input, x_prev_input = x_t, x_prev
            
            update_term = (x_input - x_prev_input) @ self.B.T
            theta = theta @ self.A.T + update_term
            
            if self.weights_lim is not None:
                theta = torch.clamp(theta, -self.weights_lim, self.weights_lim)
            
            x_prev, t_prev = x_t.clone(), t_curr
            
            if t < seq_len - 1:
                next_time = torch.full((batch_size, 1), (t + 1) / seq_len, device=device)
                root_input = torch.cat([next_time, x_t], dim=-1) if self.input_prev_data else next_time
                mu_sigma = self.root_mlp.forward_with_weights(root_input, theta, self.std_lower_bound)
                
        return torch.cat(means_list, dim=1), torch.cat(stds_list, dim=1)

def negative_log_likelihood_loss(mean, std, target):
    std = std + 1e-6
    variance = std.pow(2)
    nll = 0.5 * torch.log(2 * np.pi * variance) + (target - mean).pow(2) / (2 * variance)
    return nll.mean()

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

def train_model(config):
    """Main training loop"""
    train_loader, _ = load_data(config['training']['batch_size'])
    model = WeightSpaceModel(config).to(device)
    
    # OPTIMIZATION: Apply torch.compile for significant speedup via JIT compilation.
    print("Compiling model with torch.compile... (this may take a moment the first time)")
    model = torch.compile(model)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    train_losses = []
    
    print("Starting WSM training...")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Root MLP has {model.root_mlp.total_params} parameters (latent size)")
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        for data, _ in progress_bar:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            means, stds = model(data)
            loss = negative_log_likelihood_loss(means, stds, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, f'wsm_checkpoint_epoch_{epoch+1}.pth')
    
    return model, train_losses

def generate_completions(model, test_data, context_length):
    model.eval()
    with torch.no_grad():
        # The compiled model can be used directly for inference
        means, stds = model(test_data, inference_start=context_length)
    return means, stds

def visualize_results(model, test_loader, train_losses, config):
    context_length = config['training']['context_length']
    
    plt.figure(figsize=(15, 5))
    plt.plot(train_losses)
    plt.title('WSM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.grid(True)
    plt.show()
    
    test_data = next(iter(test_loader))[0][:6].to(device)
    means, stds = generate_completions(model, test_data, context_length)
    
    plt.figure(figsize=(18, 9))
    for i in range(6):
        plt.subplot(3, 6, i + 1)
        if i == 0: plt.title('Ground Truth')
        gt_img = test_data[i].cpu().numpy().reshape(28, 28)
        plt.imshow(gt_img, cmap='gray', vmin=-1, vmax=1)
        plt.axis('off')
        context_row = context_length // 28
        if context_row < 28: plt.axhline(y=context_row - 0.5, color='red', linewidth=1)

        plt.subplot(3, 6, i + 7)
        if i == 0: plt.title('Reconstruction (Mean)')
        recon_img = means[i].cpu().numpy().reshape(28, 28)
        plt.imshow(recon_img, cmap='gray', vmin=-1, vmax=1)
        plt.axis('off')
        if context_row < 28: plt.axhline(y=context_row - 0.5, color='red', linewidth=1)

        plt.subplot(3, 6, i + 13)
        if i == 0: plt.title('Uncertainty (Std Dev)')
        std_img = stds[i].cpu().numpy().reshape(28, 28)
        plt.imshow(std_img, cmap='hot')
        plt.axis('off')
        if context_row < 28: plt.axhline(y=context_row - 0.5, color='red', linewidth=1)
    
    plt.suptitle(f'WSM MNIST Pixel Completion (Context: {context_length})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    plt.figure(figsize=(15, 5))
    example_idx = 0
    gt_pixels = test_data[example_idx].cpu().numpy()
    pred_pixels = means[example_idx].cpu().numpy()
    std_pixels = stds[example_idx].cpu().numpy()
    
    plt.plot(gt_pixels, label='Ground Truth', alpha=0.7)
    plt.plot(pred_pixels, label='WSM Prediction', alpha=0.7)
    plt.fill_between(range(len(pred_pixels)), pred_pixels - std_pixels, pred_pixels + std_pixels, color='orange', alpha=0.3, label='±1 Std Dev')
    plt.axvline(x=context_length, color='red', linestyle='--', label='Context End')
    plt.title('Pixel Values Comparison (Example 0)')
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%
# Train the model
# Note: The first run after torch.compile will be slow due to the JIT compilation process.
# Subsequent runs will be much faster.
model, train_losses = train_model(config)

#%%
# Load test data for visualization
_, test_loader = load_data(config['training']['batch_size'])

# Visualize results
visualize_results(model, test_loader, train_losses, config)

# Save final model
# Note: To save a compiled model, you might need to save the state_dict from the original model if issues arise.
# For simplicity, we save the state_dict of the compiled model directly.
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'train_losses': train_losses
}, 'final_wsm_model_optimized.pth')
print("\nOptimized model saved as 'final_wsm_model_optimized.pth'")
