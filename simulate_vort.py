import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from numpy.fft import fft2, ifft2
import os

# Create directory for figures if it doesn't exist
if not os.path.exists('figs'):
    os.makedirs('figs')
    
# Constants
M = 256 #the bigger the better
N = M
Lx = 2 * np.pi
Ly = 2 * np.pi
nu = 5e-4
Sc = 0.7
beta = 0
ar = 0.02
b = 1
CFLmax = 0.8
tend = 2000

# Spatial grid
x = np.linspace(0, Lx, M, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
dx = Lx / M
dy = Ly / N

# Wavenumbers
kx = np.fft.fftfreq(M, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
kx, ky = np.meshgrid(kx, ky, indexing='ij')

# Time variables
time = 10

# Filters
index_kmax = M // 3
kmax = kx[index_kmax, 0]
filter = np.ones((M, N))
filter[index_kmax+1:2*index_kmax+1, index_kmax+1:2*index_kmax+1] = 0

np.random.seed(64)

# Initialize fields
u = np.zeros((M, N))
v = np.zeros((M, N))
omega = np.zeros((M, N))
phi = np.random.rand(M, N)

# Derivatives and inverse Laplacian
ddx = 1j * kx
ddy = 1j * ky
idel2 = -kx**2 - ky**2
idel2[0, 0] = 1
idel2 = 1 / idel2
idel2[0, 0] = 0

kk = kx**2 + ky**2
k2 = kx**2 + ky**2
kk[(kk >= 6**2) & (kk <= 7**2)] *= -1
kk[kk <= 2**2] *= 8

for i in range(M):
    for j in range(N):
        u[i, j] = np.cos(2 * x[i]) * np.sin(2 * y[j]) + ar * np.random.rand()
        v[i, j] = -np.sin(2 * x[i]) * np.cos(2 * y[j]) + ar * np.random.rand()

uhat = fft2(u)
vhat = fft2(v)
omegahat = ddx * vhat - ddy * uhat
phihat = fft2(phi)

# Create netCDF file
ncid = Dataset('turb2d6x8f.nc', 'w', format='NETCDF4')
ncid.createDimension('x', M)
ncid.createDimension('y', N)
ncid.createDimension('time', None)
varid_x = ncid.createVariable('x', 'f4', ('x',))
varid_y = ncid.createVariable('y', 'f4', ('y',))
varid_time = ncid.createVariable('time', 'f4', ('time',))
varid_u = ncid.createVariable('u', 'f4', ('time', 'x', 'y'))
varid_v = ncid.createVariable('v', 'f4', ('time', 'x', 'y'))
varid_omega = ncid.createVariable('vorticity', 'f4', ('time', 'x', 'y'))
varid_phi = ncid.createVariable('scalar', 'f4', ('time', 'x', 'y'))
varid_dissipation = ncid.createVariable('dissipation', 'f4', ('time', 'x', 'y'))
varid_x[:] = x
varid_y[:] = y
varid_time[0] = time
varid_u[0, :, :] = u
varid_v[0, :, :] = v
varid_omega[0, :, :] = omega
varid_phi[0, :, :] = phi
varid_dissipation[0, :, :] = np.zeros((M, N))

dt = 0.5 * min(dx, dy)
nstep = 1

while time < tend:
    # Substep 1
    psihat = -idel2 * omegahat
    uhat = ddy * psihat
    vhat = -ddx * psihat
    u = np.real(ifft2(uhat))
    v = np.real(ifft2(vhat))
    omegadx = np.real(ifft2(ddx * omegahat))
    omegady = np.real(ifft2(ddy * omegahat))
    facto = np.exp(-nu * 8 / 15 * dt * kk)
    factp = np.exp(-nu / Sc * 8 / 15 * dt * k2)
    r0o = -fft2(u * omegadx + v * omegady) + beta * vhat
    r0p = -fft2(u * np.real(ifft2(ddx * phihat)) + v * np.real(ifft2(ddy * phihat))) + b * vhat
    omegahat = facto * (omegahat + dt * 8 / 15 * r0o)
    phihat = factp * (phihat + dt * 8 / 15 * r0p)

    # Substep 2
    psihat = -idel2 * omegahat
    uhat = ddy * psihat
    vhat = -ddx * psihat
    u = np.real(ifft2(uhat))
    v = np.real(ifft2(vhat))
    omegadx = np.real(ifft2(ddx * omegahat))
    omegady = np.real(ifft2(ddy * omegahat))
    r1o = -fft2(u * omegadx + v * omegady) + beta * vhat
    r1p = -fft2(u * np.real(ifft2(ddx * phihat)) + v * np.real(ifft2(ddy * phihat))) + b * vhat
    omegahat = omegahat + dt * (-17 / 60 * facto * r0o + 5 / 12 * r1o)
    phihat = phihat + dt * (-17 / 60 * factp * r0p + 5 / 12 * r1p)
    facto = np.exp(-nu * (-17 / 60 + 5 / 12) * dt * kk)
    factp = np.exp(-nu / Sc * (-17 / 60 + 5 / 12) * dt * k2)
    omegahat *= facto
    phihat *= factp

    # Substep 3
    psihat = -idel2 * omegahat
    uhat = ddy * psihat
    vhat = -ddx * psihat
    u = np.real(ifft2(uhat))
    v = np.real(ifft2(vhat))
    omegadx = np.real(ifft2(ddx * omegahat))
    omegady = np.real(ifft2(ddy * omegahat))
    r2o = -fft2(u * omegadx + v * omegady) + beta * vhat
    r2p = -fft2(u * np.real(ifft2(ddx * phihat)) + v * np.real(ifft2(ddy * phihat))) + b * vhat
    omegahat = omegahat + dt * (-5 / 12 * facto * r1o + 3 / 4 * r2o)
    phihat = phihat + dt * (-5 / 12 * factp * r1p + 3 / 4 * r2p)
    facto = np.exp(-nu * (-5 / 12 + 3 / 4) * dt * kk)
    factp = np.exp(-nu / Sc * (-5 / 12 + 3 / 4) * dt * k2)
    omegahat *= facto
    phihat *= factp

    phihat *= filter
    omegahat *= filter
    time += dt
    nstep += 1

    CFL = max(np.max(np.abs(u)) / dx * dt, np.max(np.abs(v)) / dy * dt)

    if nstep % 20 == 0:
        phi = np.real(ifft2(phihat))
        omega = np.real(ifft2(omegahat))
        dissipation = 2 * nu * (np.real(ifft2(ddx * uhat))**2 + np.real(ifft2(ddy * uhat))**2 + np.real(ifft2(ddx * vhat))**2 + np.real(ifft2(ddy * vhat))**2)
        eta = (nu**3 / np.mean(dissipation))**0.25



        #levels = np.linspace(0, 1, 900+1)
        plt.figure(figsize=(11, 11))
        #plt.pcolormesh(x, y, u.T, shading='auto', cmap='jet', vmin = -1.2, vmax = 1.2)
        plt.pcolormesh(x, y, phi.T, shading='auto', cmap='jet')#, vmin = -3., vmax = 3)

        plt.title(f'Sciencestical 0: {nstep:05d}', fontsize=26)
        plt.xlabel('X (m)', fontsize=26)
        plt.ylabel('Y (m)', fontsize=26)
        plt.axis('equal')
        plt.axis('tight')

        # Customize tick parameters
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #cbar = plt.colorbar()
        #cbar.set_label('Velocity (m/s)', fontsize=26)
        #cbar.ax.tick_params(labelsize=20)
        #cbar.ax.tick_params([])

        # Save the figure and close it without displaying
        plt.savefig(f'figs/step_{nstep:05d}.png')
        plt.close()
        #plt.show()
        print(f'step = {nstep} time = {time:.2g} dt = {dt:.2g} CFL = {CFL:.2g} kmax*eta = {eta * kmax:.2g} {eta * kmax / np.sqrt(Sc):.2g}')

        dim_time_len = len(ncid.dimensions['time'])
        varid_time[dim_time_len] = time
        varid_phi[dim_time_len, :, :] = phi
        varid_u[dim_time_len, :, :] = u
        varid_v[dim_time_len, :, :] = v
        varid_omega[dim_time_len, :, :] = omega
        varid_dissipation[dim_time_len, :, :] = dissipation
        ncid.sync()

    dt = CFLmax / CFL * dt

# Close netCDF file
ncid.close()
print('All done ')
