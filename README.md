# Turbulent Flow Simulation and Visualization

This Python script simulates a 2D turbulent flow and visualizes various physical fields over time using the Fast Fourier Transform (FFT). The results are saved in a NetCDF file and individual plots are saved as PNG images.


[![Watch the video](https://img.youtube.com/vi/qJuOattT-cI/maxresdefault.jpg)](https://youtu.be/qJuOattT-cI?si=4nwLRtPsmo52IwdI)

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- netCDF4

You can install the required packages using pip:

```bash
pip install numpy matplotlib netCDF4
```

## Key Variables
- M, N: Grid dimensions.
- Lx, Ly: Domain size.
- nu: Viscosity.
- Sc: Schmidt number.
- ar: Amplitude of initial perturbation.
- CFLmax: Maximum allowed CFL number.
- tend: End time of the simulation.

## File Structure
- simulate_turb_flow.py: The main simulation and visualization script.
- figs/: Directory where the generated plots are saved.
- turb2d6x8f.nc: NetCDF file containing the simulation data.
