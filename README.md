# Navier-Stokes Vorticity Simulator

A minimal 3D incompressible Navier-Stokes solver using spectral methods and PyTorch.

## Requirements
- Python 3.8+
- PyTorch with CUDA support for GPU acceleration (CPU works but is slower)

## Usage
```
python main.py --N 64 --L 6.283 --nu 0.01 --dt 1e-3 --T 0.1
```
All arguments are optional. The solver starts from a 3D Taylor–Green vortex and prints final diagnostics.
