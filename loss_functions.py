import torch

# %%
def schrodinger_residual(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    # get psi
    psi = model.get_psi(t, x)

    # ∂ψ/∂t
    dpsi_dt = torch.autograd.grad(
        psi, t,  
        grad_outputs=torch.ones_like(psi),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # ∂²ψ/∂x²
    dpsi_dx = torch.autograd.grad(
        psi, x,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True
    )[0]
    
    d2psi_dx2 = torch.autograd.grad(
        dpsi_dx, x,
        grad_outputs=torch.ones_like(dpsi_dx),
        create_graph=True
    )[0]
    
    # H = -ħ²/2m ∇² + V(x)
    # iħ ∂ψ/∂t = Hψ
    lhs = 1j * model.hbar * dpsi_dt
    rhs = - (model.hbar**2) / (2 * model.m) * d2psi_dx2 + model.V(x) * psi
    
    # return square
    return torch.max(torch.abs(lhs - rhs))

# %%
def observation_loss(model, t_obs, x_obs):
    log_prob = torch.log(model.get_probability(t_obs, x_obs) + 1e-8)
    return -torch.mean(log_prob)

# %%
def normalization_loss(model, t, x, device, x_range=(-4.0, 4.0), num_samples=1024, use_log=False):
    batch_size = t.shape[0]
    low, high = x_range
    volume = high - low  # 1D volume = length

    # Sample x uniformly in [low, high] for each t
    # Shape: [batch_size, num_samples, 1]
    x_samples = torch.rand(batch_size, num_samples, 1, device=device)
    x_samples = x_samples * (high - low) + low

    # Expand t to match x_samples: [batch_size, num_samples, 1]
    t_expanded = t.unsqueeze(1).expand(-1, num_samples, -1)

    # Flatten for batch evaluation
    t_flat = t_expanded.reshape(-1, 1)      # [batch_size * num_samples, 1]
    x_flat = x_samples.reshape(-1, 1)       # [batch_size * num_samples, 1]

    with torch.set_grad_enabled(True):  # Ensure gradients flow
        probs_flat = model.get_probability(t_flat, x_flat)  # [B*S, 1]
        probs = probs_flat.view(batch_size, num_samples)    # [B, S]

    # Monte Carlo estimate of integral: (b - a) * mean(p(x_i))
    integral = volume * torch.mean(probs, dim=1)  # [batch_size]

    if use_log:
        # More stable when integral is close to 0
        loss_per_sample = (torch.log(integral + 1e-8) - torch.log(torch.ones_like(integral))) ** 2
    else:
        loss_per_sample = (integral - 1.0) ** 2

    return torch.mean(loss_per_sample)
