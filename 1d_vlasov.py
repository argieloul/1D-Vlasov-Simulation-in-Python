# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 04:18:52 2024

@author: arlou
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import solve
from scipy.integrate import trapz
import cv2
import os

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


font = {'family': 'serif',
        'weight': 'normal',
        'size': 16,
        }

def poisson(n, d, f, phi_left, phi_right):
    # Create the right-hand side vector
    b = f  # Scaling the source term for each point

    # Initialize matrix A for the finite difference approximation
    A = -np.diag(2 * np.ones(n)) + np.diag(1*np.ones(n - 1), k=1) + np.diag(1*np.ones(n - 1), k=-1)
    
    # Apply Dirichlet boundary conditions
    # Set left boundary (phi[0] = phi_left)
    
    A[0, :] = 0
    A[0, -1] = 1
    b[0] = phi_left

    # Set right boundary (phi[N-1] = phi_right)
    A[-1, :] = 0
    A[-1, 0] = 1
    b[-1] = phi_right

    
    A /= d**2

    # Solve the linear system A * phi = b
    sol = solve(A, b)

    return sol

def advance_position(f, u, mask):
    f[:,mask] -= 0.5*dt/dx*(f[:,mask] - np.roll(f, 1, axis=0)[:,mask])*u[mask]
    f[:,~mask] -= 0.5*dt/dx*(np.roll(f, -1, axis=0)[:,~mask] - f[:,~mask])*u[~mask]
    return f
    
def advance_velocity(f, E, mask, r):
    f[mask,:] -= 0.5*(dt/du)*r*((f[mask,:] - np.roll(f, 1, axis=1)[mask,:]).T*E[mask]).T
    f[~mask,:] -= 0.5*(dt/du)*r*((np.roll(f, -1, axis=1)[~mask,:] - f[~mask,:]).T*E[~mask]).T
    return f

# Position axis

xmin = 0
xmax = 1
nx = int(input('Points on position axis: ')) 
dx = (xmax - xmin)/nx
x = np.arange(xmin, xmax + dx, dx)

# Velocity axis

umin = float(input('Bottom velocity boundary: ')) 
umax = float(input('Top velocity boundary: ')) 
nu = int(input('Points on velocity axis: ')) 
du = (umax - umin)/nu
u = np.arange(umin, umax + du, du)

dt = 0.1*np.min((dx, du))

uth = 0.1*umax # Particle thermal velocity
q = -1 # Particle charge 
m = 1 # Particle mass
r = q/m # Particle charge-to-mass ratio
n0 = 25 # Distribution function scaling (controls total number of particles)

f = np.zeros((nx+1, nu+1))

for i in range(nu+1):
    f[:, i] = n0*(np.exp(-(u[i] - 2.5)**2/uth**2) + np.exp(-(u[i] + 2.5)**2/uth**2))

np.random.seed(42)

fnoise = np.random.random(f.shape)*0.1*n0
f += fnoise

n_init = trapz(f, u, axis = 1)
rho_ion = -q*np.mean(n_init) # Charge density of static ion background (neutralizes average negative charge density)
rho_init = q*n_init + rho_ion # Charge density after adding ion background 
phi_init = -poisson(nx + 1, dx, 4*np.pi*rho_init, 0, 0)
E_init = -np.gradient(phi_init, x)

omega_plasma = np.sqrt(4*np.pi*q**2/m*trapz(f - fnoise, u, axis = 1)[0])
t_plasma = (2*np.pi)/omega_plasma
t_total = float(input('Total simulation time in plasma period units: '))*t_plasma
Niter = round(t_total/dt) # Number of iterations

time = np.linspace(0, t_total, Niter)

t_save = float(input('Result storing timestep: '))*t_plasma
ms = round((t_save/t_total)*Niter)
nt = (Niter // ms) + 1 # Include initial state
        
u_mask = u > 0

# Allocate memory for the results

result = np.zeros((nt, nx + 1, nu + 1))

p = 0

while p ==0: 
    ans = input('Plot phase space distribution while running simulation? (y/n) ') # Set this to true to create a phase space plot every m-th timestep. False is faster
    if ans == 'y':
        plot = True
        p += 1
    elif ans == 'n':
        plot = False
        p += 1

s = 0

result[s] = f

for it in tqdm(range(Niter)):

    # Advance positions over half a timestep
    
    f = advance_position(f, u, u_mask)
    
    # Solve Gauss' law for electric field
    
    rho = q*trapz(f, u, axis = 1)
    
    rho = rho + rho_ion

    phi = -poisson(nx + 1, dx, 4*np.pi*rho, 0, 0)

    E = -np.gradient(phi, dx)

    # Define mask for E for upwind scheme
    
    E_mask = E > 0 if q > 0 else E < 0

    # Advance velocities over half a timestep
        
    f = advance_velocity(f, E, E_mask, r)
    
    # Advance positions over half a timestep
    
    f = advance_position(f, u, u_mask)
    
    if it != 0 and it%ms == 0:
        s += 1
        result[s] = f
        
    if plot:
        if it == 0:
            plt.imshow(result[s].T, cmap = 'magma', aspect = 0.5*xmax/umax, origin = 'lower', extent = [xmin, xmax, umin, umax])
            plt.xlabel(r'$x$', fontdict = font)
            plt.ylabel(r'$u$', fontdict = font)
            plt.title(f'$t = {round(time[it]/t_plasma, 1)}\, T_{{p}}$')
            plt.show()
        
        if it != 0 and it%ms == 0:
            plt.imshow(result[s].T, cmap = 'magma', aspect = 0.5*xmax/umax, origin = 'lower', extent = [xmin, xmax, umin, umax])
            plt.xlabel(r'$x$', fontdict = font)
            plt.ylabel(r'$u$', fontdict = font)
            plt.title(f'$t = {round(time[it]/t_plasma, 1)}\, T_{{p}}$')
            plt.show()
            
ff = result
    
nf = trapz(ff, u, axis = 2)
rhof = q*nf + rho_ion
phif = np.empty((len(ff), nx + 1))
Ef = np.empty((len(ff), nx + 1))


for i in range(len(ff)):
    phif[i] = -poisson(nx + 1, dx, 4*np.pi*rhof[i], 0, 0) 

Ef = -np.gradient(phif, dx, axis = 1)