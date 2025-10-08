import torch
import torch.nn as nn
from loss_functions import observation_loss, normalization_loss, schrodinger_residual
import os
import numpy as np
import torch.optim as optim
# %%
class PotentialNN(nn.Module):
    """learnable V(x) unrelated to t"""
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # input shape [batch, 1] tensor
        # output shape [batch, 1] tensor
        return self.net(x)
    
# %% ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # input shape [batch, hidden_size]
        identity = x
        out = self.tanh(self.fc1(x))
        out = self.tanh(self.fc2(out))
        out = out + identity
        # input shape [batch, hidden_size]
        return self.tanh(out)

# %%
class WaveFunctionNN(nn.Module):
    """ResNet Wave function"""
    def __init__(self, hidden_size, num_layers):
        super(WaveFunctionNN, self).__init__()
        
        # input: take [t, x] to hidden_size dims
        self.input_layer = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh()
        )
        
        # ResidualBlocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(ResidualBlock(hidden_size))
        
        # output: take hidden_size to real and imag
        self.output_layer = nn.Linear(hidden_size, 2)
        
        # learnabel physcial coef
        self.hbar = nn.Parameter(torch.tensor(0.1))
        self.m = nn.Parameter(torch.tensor(1.0))

        # potential function unrelated to time
        self.V = PotentialNN(hidden_size)

    def forward(self, t, x):
        # t, x shape is [batch, 1]
        inputs = torch.cat([t, x], dim=1)
        # shape: [batch_size, 2]
        
        h = self.input_layer(inputs)
        
        for block in self.res_blocks:
            h = block(h)
        
        out = self.output_layer(h)
        
        real = out[:, 0:1]
        imag = out[:, 1:2]
        return real, imag

    def get_psi(self, t, x):
        """get complex ψ(t,x)"""
        real, imag = self(t, x)
        return torch.complex(real, imag)
    
    def get_probability(self, t, x):
        """get probability |ψ(t,x)|²"""
        psi = self.get_psi(t, x)
        return torch.abs(psi)**2

# %%
def train(model, train_loader, eval_loader, epochs, lr, device,
          save_path='models_pth/best_model.pth', 
          log_interval=100,
          max_grad_norm=5.0,
          eval_method='grid'):  # or 'mcmc'
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20)
    
    best_composite = float('inf')  # ✅ 初始化 best_composite
    best_epoch = 0
    
    print(f"Starting training on {device} for {epochs} epochs...")
    print(f"Gradient clipping enabled with max norm = {max_grad_norm}")
    print(f"{'='*50}")
    
    for epoch in range(1, epochs + 1):
        # ========== Training ==========
        model.train()
        train_losses = []
        schrodinger_losses = []
        obs_losses = []
        norm_losses = []
        grad_norms = []
        
        for t, x in train_loader:
            t, x = t.to(device), x.to(device)  # [B,1], [B,1]
            
            optimizer.zero_grad()

            # Losses (all weighted by 1000)
            schrodinger_loss = 1 * schrodinger_residual(model, t, x)
            obs_loss = 1 * observation_loss(model, t, x)
            norm_loss = 1 * normalization_loss(model, t, x, device)

            loss = schrodinger_loss + obs_loss + norm_loss

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Logging
            train_losses.append(loss.item())
            schrodinger_losses.append(schrodinger_loss.item())
            obs_losses.append(obs_loss.item())
            norm_losses.append(norm_loss.item())
            grad_norms.append(total_norm.item())

        avg_train_loss = np.mean(train_losses)
        avg_schrodinger = np.mean(schrodinger_losses)
        avg_obs = np.mean(obs_losses)
        avg_norm = np.mean(norm_losses)
        avg_grad_norm = np.mean(grad_norms)
        
        scheduler.step(avg_train_loss)

        # ========== Evaluation ==========
        model.eval()
        eval_mae = 0.0
        eval_pde = 0.0
        eval_integral = 0.0
        eval_count = 0

        for t_eval, x_true in eval_loader:  # 注意：移出 with torch.no_grad()
            t_eval = t_eval.to(device)
            x_true = x_true.to(device)
            batch_size = t_eval.shape[0]

            # 1. MAE (不需要梯度)
            with torch.no_grad():
                x_pred = predicted(model, t_eval, device, method=eval_method)
                mae_batch = torch.mean((x_pred - x_true)** 2).item()

            # 2. PDE residual (需要梯度！)
            t_eval.requires_grad_(True)  # ✅ 启用梯度
            x_true.requires_grad_(True)  # ✅ 启用梯度
            pde_batch = schrodinger_residual(model, t_eval, x_true).item()
            
            # 3. Normalization integral (不需要梯度)
            with torch.no_grad():
                integral_batch = compute_integral(model, t_eval, device, x_range=(-4,4), num_samples=1024)

            eval_mae += mae_batch * batch_size
            eval_pde += pde_batch * batch_size
            eval_integral += integral_batch * batch_size
            eval_count += batch_size

        eval_mae /= eval_count
        eval_pde /= eval_count
        avg_integral = eval_integral / eval_count
        norm_error = abs(avg_integral - 1.0)

        # 联合指标（越小越好）
        composite_score = 5* eval_mae + np.sqrt(eval_pde + 1e-8) + norm_error  # ✅ 加 epsilon 防 NaN

        # ========== Logging & Saving ==========
        if composite_score < best_composite:  # ✅ 直接比较
            best_composite = composite_score
            best_epoch = epoch  # ✅ 记录 epoch
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch}: New best composite score: {composite_score:.6f} → Model saved!")

        if epoch % log_interval == 0 or composite_score == best_composite:
            print(f"Epoch [{epoch}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Schrodinger: {avg_schrodinger:.6f} | "
                  f"Obs: {avg_obs:.6f} | "
                  f"Norm: {avg_norm:.6f} | "
                  f"Grad Norm: {avg_grad_norm:.4f} | "
                  f"Eval MAE: {eval_mae:.6f} | "
                  f"Composite: {composite_score:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"{'='*50}")
    print(f"Training completed! Best composite score: {best_composite:.6f} at epoch {best_epoch}")
    print(f"Best model saved to: {save_path}")

    # torch.save(model.state_dict(), save_path)
    
    return best_composite  # ✅ 返回 composite_score

# %%
def compute_integral(model, t, device, x_range=(-4.0, 4.0), num_samples=1024):
    batch_size = t.shape[0]
    low, high = x_range
    volume = high - low

    x_samples = torch.rand(batch_size, num_samples, 1, device=device) * (high - low) + low
    t_exp = t.unsqueeze(1).expand(-1, num_samples, -1)

    probs = model.get_probability(
        t_exp.reshape(-1, 1),
        x_samples.reshape(-1, 1)
    ).view(batch_size, num_samples)

    integral = volume * torch.mean(probs, dim=1)  # [B]
    return torch.mean(integral).item()  # scalar

# %%
def predicted(model, t, device, 
              x_range=(-4.0, 4.0), 
              num_points=512,
              num_mcmc_samples=1000,  # 重命名参数，避免混淆
              method='grid'):
    """
    Always returns tensor of shape [batch_size, 1].
    """
    assert isinstance(t, torch.Tensor), "t must be a torch.Tensor"
    assert t.dim() == 2 and t.shape[1] == 1, "t must be [batch, 1]"
    
    if method == 'grid':
        model.eval()
        batch_size = t.shape[0]
        x_search = torch.linspace(x_range[0], x_range[1], num_points, device=device)
        t_expanded = t.expand(-1, num_points)
        x_expanded = x_search.expand(batch_size, -1)
        
        with torch.no_grad():
            all_probs = model.get_probability(
                t_expanded.reshape(-1, 1),
                x_expanded.reshape(-1, 1)
            ).view(batch_size, num_points)
        
        max_indices = torch.argmax(all_probs, dim=1)  # [B]
        return x_search[max_indices].unsqueeze(1)     # [B, 1]

    elif method == 'mcmc':
        # Get MCMC samples [B, num_mcmc_samples]
        samples = mcmc_sample(
            model, t, device,
            x_range=x_range,
            num_samples=num_mcmc_samples,  # 注意参数名
            burn_in=200,
            thin=1,
            proposal_std=0.3
        )
        # Return mean as point estimate → [B, 1]
        return torch.mean(samples, dim=1, keepdim=True)

    else:
        raise ValueError("method must be 'grid' or 'mcmc'")

# %%
def mcmc_sample(model, t, device, 
                x_range=(-4.0, 4.0), 
                num_samples=1000, 
                burn_in=200, 
                thin=1, 
                proposal_std=0.5):
    """
    Metropolis-Hastings MCMC sampler for |ψ(t; x)|² in 1D.
    
    Args:
        model: WaveFunctionNN
        t: tensor of shape [batch_size, 1] or [1, 1] (must be on `device`)
        device: torch.device
        x_range: (low, high) for sampling domain
        num_samples: number of samples to return **per time point**
        burn_in: number of initial steps to discard
        thin: keep every `thin`-th sample
        proposal_std: std of Gaussian proposal
    
    Returns:
        samples: tensor of shape [batch_size, num_samples] on `device`
    """
    model.eval()
    assert isinstance(t, torch.Tensor), "t must be a torch.Tensor"
    
    batch_size = t.shape[0]
    low, high = x_range

    # Initialize current state: uniform in [low, high]
    current_x = torch.rand(batch_size, device=device) * (high - low) + low  # [B]
    samples = []

    total_steps = burn_in + num_samples * thin

    with torch.no_grad():
        for step in range(total_steps):
            # Propose new x: Gaussian perturbation
            proposal_x = current_x + torch.randn_like(current_x) * proposal_std  # [B]
            
            # Enforce hard boundaries: reject if out of range
            accept = (proposal_x >= low) & (proposal_x <= high)  # [B], bool
            
            # Evaluate probabilities only for valid proposals
            prob_current = model.get_probability(t, current_x.unsqueeze(1)).squeeze(1)  # [B]
            prob_proposal = torch.zeros_like(prob_current)
            if accept.any():
                x_prop_valid = proposal_x[accept].unsqueeze(1)  # [K, 1]
                t_valid = t[accept]                             # [K, 1]
                prob_proposal[accept] = model.get_probability(t_valid, x_prop_valid).squeeze(1)
            
            # Compute acceptance ratio (symmetric proposal)
            ratio = torch.where(
                accept,
                torch.clamp(prob_proposal / (prob_current + 1e-12), max=1.0),
                torch.zeros_like(prob_current)
            )  # [B]
            
            # Accept/reject
            u = torch.rand(batch_size, device=device)
            accept_final = u < ratio
            current_x = torch.where(accept_final, proposal_x, current_x)
            
            # Store sample after burn-in and thinning
            if step >= burn_in and (step - burn_in) % thin == 0:
                samples.append(current_x.clone())
                if len(samples) >= num_samples:
                    break
    
    # Stack samples: [num_samples, B] -> [B, num_samples]
    samples = torch.stack(samples, dim=0).T  # [B, num_samples]
    return samples