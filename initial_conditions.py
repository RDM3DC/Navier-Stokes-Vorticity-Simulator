import torch
import math


def taylor_green(N: int, L: float, device: torch.device, amplitude: float = 1.0) -> torch.Tensor:
    """Return 3D Taylor-Green vortex initial velocity field."""
    x = torch.linspace(0, L, N, device=device, dtype=torch.float32)
    grid = torch.meshgrid(x, x, x, indexing='ij')
    X, Y, Z = grid
    factor = 2 * math.pi / L
    u_x =  amplitude * torch.sin(factor * X) * torch.cos(factor * Y) * torch.cos(factor * Z)
    u_y = -amplitude * torch.cos(factor * X) * torch.sin(factor * Y) * torch.cos(factor * Z)
    u_z = torch.zeros_like(u_x)
    return torch.stack([u_x, u_y, u_z], dim=0)
