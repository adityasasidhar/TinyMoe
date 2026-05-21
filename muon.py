import torch

@torch.compile
def newton_schulz_iteration(G, steps=5):
    """
    The mathematical heart of Muon.
    This takes a 2D matrix (like a gradient) and iteratively orthogonalizes it
    using a 5th-order polynomial approximation.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # NS iteration is fastest and most stable in bfloat16
    X = G.bfloat16()
    
    # Normalize to ensure the spectral norm is < 1
    X = X / (X.norm() + 1e-7)
    
    # Optimization: We always want to multiply the smaller dimensions together
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if transposed:
        X = X.T
        
    return X


class Muon(torch.optim.Optimizer):
    """
    Standalone, single-GPU Muon Optimizer.
    WARNING: Only pass 2D parameters (like nn.Linear weights) to this optimizer!
    """
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.ndim != 2:
                    raise ValueError(f"Muon only supports 2D tensors, but got {grad.ndim}D tensor.")

                # Initialize momentum state if it doesn't exist
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                # 1. Update Momentum
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                # 2. Orthogonalize the momentum via Newton-Schulz
                update = newton_schulz_iteration(buf)

                # 3. Scale the update
                # Muon authors found that scaling by the square root of the max dimension works best
                scale = max(p.size(0), p.size(1)) ** 0.5
                update = update.to(p.dtype) * scale
                
                # Apply the update
                p.add_(update, alpha=-lr)

        return loss
