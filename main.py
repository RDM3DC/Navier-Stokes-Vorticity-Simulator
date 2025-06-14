import argparse
import torch
from solver import run_simulation
from initial_conditions import taylor_green


def main():
    parser = argparse.ArgumentParser(description="3D Navier-Stokes Vorticity Simulator (GPU)")
    parser.add_argument('--N', type=int, default=64, help='Grid points per dimension')
    parser.add_argument('--L', type=float, default=2 * 3.141592653589793, help='Domain size')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity')
    parser.add_argument('--dt', type=float, default=1e-3, help='Time step')
    parser.add_argument('--T', type=float, default=0.1, help='Final time')
    parser.add_argument('--amplitude', type=float, default=1.0, help='Initial velocity amplitude')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    u0 = taylor_green(args.N, args.L, device, amplitude=args.amplitude)
    u0 = u0.to(device)
    u, diagnostics = run_simulation(u0, args.nu, args.dt, args.T, args.L, device)

    print("Final energy: {:.6f}".format(diagnostics['energy'][-1]))
    print("Max vorticity: {:.6f}".format(max(diagnostics['max_vorticity'])))


if __name__ == '__main__':
    main()
