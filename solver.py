import torch
import torch.fft as fft


def spectral_laplacian(u: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """Return Laplacian of velocity field using spectral multiplication."""
    u_hat = fft.fftn(u, dim=(-3, -2, -1))
    lap_hat = -k2 * u_hat
    return torch.real(fft.ifftn(lap_hat, dim=(-3, -2, -1)))


def project(u: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """Helmholtz-Hodge projection to enforce divergence-free condition."""
    u_hat = fft.fftn(u, dim=(-3, -2, -1))
    div_hat = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    factor = div_hat / (k2 + 1e-10)
    u_hat[0] -= kx * factor
    u_hat[1] -= ky * factor
    u_hat[2] -= kz * factor
    u = torch.real(fft.ifftn(u_hat, dim=(-3, -2, -1)))
    return u


def derivatives(u: torch.Tensor, dx: float):
    dudx = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
    dudy = (torch.roll(u, -1, dims=2) - torch.roll(u, 1, dims=2)) / (2 * dx)
    dudz = (torch.roll(u, -1, dims=3) - torch.roll(u, 1, dims=3)) / (2 * dx)
    return dudx, dudy, dudz


def curl(u: torch.Tensor, dx: float) -> torch.Tensor:
    dudx, dudy, dudz = derivatives(u, dx)
    omega_x = dudy[2] - dudz[1]
    omega_y = dudz[0] - dudx[2]
    omega_z = dudx[1] - dudy[0]
    return torch.stack([omega_x, omega_y, omega_z], dim=0)


def run_simulation(u: torch.Tensor, nu: float, dt: float, T: float, L: float, device: torch.device):
    """Run 3D Navier-Stokes simulation with RK2."""
    N = u.shape[1]
    dx = L / N
    steps = int(T / dt)

    k = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = kx ** 2 + ky ** 2 + kz ** 2

    diagnostics = {"max_vorticity": [], "enstrophy": [], "energy": []}

    u = project(u, kx, ky, kz, k2)

    for _ in range(steps):
        dudx, dudy, dudz = derivatives(u, dx)
        adv_x = u[0] * dudx[0] + u[1] * dudy[0] + u[2] * dudz[0]
        adv_y = u[0] * dudx[1] + u[1] * dudy[1] + u[2] * dudz[1]
        adv_z = u[0] * dudx[2] + u[1] * dudy[2] + u[2] * dudz[2]
        adv = torch.stack([adv_x, adv_y, adv_z], dim=0)
        lap = spectral_laplacian(u, k2)
        rhs = -adv + nu * lap
        u_star = u + dt * rhs
        u_star = project(u_star, kx, ky, kz, k2)

        dudx_s, dudy_s, dudz_s = derivatives(u_star, dx)
        adv_x = u_star[0] * dudx_s[0] + u_star[1] * dudy_s[0] + u_star[2] * dudz_s[0]
        adv_y = u_star[0] * dudx_s[1] + u_star[1] * dudy_s[1] + u_star[2] * dudz_s[1]
        adv_z = u_star[0] * dudx_s[2] + u_star[1] * dudy_s[2] + u_star[2] * dudz_s[2]
        adv_s = torch.stack([adv_x, adv_y, adv_z], dim=0)
        lap_s = spectral_laplacian(u_star, k2)
        rhs_s = -adv_s + nu * lap_s

        u = u + dt * 0.5 * (rhs + rhs_s)
        u = project(u, kx, ky, kz, k2)

        omega = curl(u, dx)
        max_vort = omega.abs().max().item()
        enstrophy = 0.5 * torch.sum(omega ** 2) * dx ** 3
        energy = 0.5 * torch.sum(u ** 2) * dx ** 3
        diagnostics["max_vorticity"].append(max_vort)
        diagnostics["enstrophy"].append(enstrophy.item())
        diagnostics["energy"].append(energy.item())

    return u, diagnostics
