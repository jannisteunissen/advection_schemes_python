#!/usr/bin/env python3

import numpy as np
from scipy.optimize import newton_krylov
import scipy.constants as c
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical tests for scalar advection equation')
parser.add_argument('-limiter', type=str, default='vanleer',
                    choices=['gminmod', 'vanleer', 'koren',
                             'dummy'],
                    help='Which limiter to use')
parser.add_argument('-gminmod_theta', type=float, default=1.0,
                    help='Theta for generalized minmod limiter, from 1 to 2')
parser.add_argument('-N', type=int, default=51, help='Number of grid points')
parser.add_argument('-L', type=float, default=1e-3, help='Domain size (m)')
parser.add_argument('-n0', type=float, default=1e18, help='Initial density')
parser.add_argument('-density_normalization', type=float, default=1e18,
                    help='Density normalization factor')
parser.add_argument('-E_bg', type=float, default=1e6,
                    help='Applied field (V/m)')
parser.add_argument('-bc_phi', type=str, choices=['dirichlet', 'neumann'],
                    default='dirichlet',
                    help='Boundary condition for potential')
parser.add_argument('-dt', type=float, default=1e-10, help='Time step (s)')
parser.add_argument('-mu', type=float, default=0.03, help='Mobility (m2/Vs)')
parser.add_argument('-mu_ion', type=float, default=2e-4,
                    help='Ion mobility (m2/Vs)')
parser.add_argument('-D', type=float, default=0.1,
                    help='Diffusion coefficient (m2/s)')
parser.add_argument('-theta', type=float, default=0.5,
                    help='Theta parameter for time integration')
parser.add_argument('-test', type=str, default='shock',
                    help='Type of test (initial condition)')
parser.add_argument('-f_tol', type=float, help='Solver absolute tolerance')
parser.add_argument('-verbose', action='store_true')
parser.add_argument('-T', type=float, default=1.0e-9,
                    help='End time')
args = parser.parse_args()

dx = args.L / args.N
x = np.linspace(0.5*dx, args.L-0.5*dx, args.N)
dt = args.dt

# Avoid division by zero
mu_safe = max(abs(args.mu), 1e-100)
D_safe = max(args.D, 1e-100)

print(f'end time:        {args.T:.2e}')
print(f'dt:              {dt:.2e}')
print(f'dt CFL (t0):     {dx/abs(args.E_bg*mu_safe):.2e}')
print(f'dt diff (t0):    {dx**2/(2*D_safe):.2e}')
print(f'dt drt (t0):     {c.epsilon_0/(c.e*args.n0*mu_safe):.2e}')


def get_alpha(E):
    return 0.025 * E + np.exp(-1e7/E)


def add_ghost_cells(u, g=2):
    N = u.size
    ug = np.zeros(N+2*g)
    ug[g:-g] = u
    ug[:g] = 0
    ug[-g:] = 0
    return ug


def koren_phi(r):
    t = np.minimum(1/3 + 2*r/3, 2.)
    t = np.minimum(2*r, t)
    t = np.maximum(0., t)
    return t


def vanleer_phi(r):
    return (r + np.abs(r))/(1 + np.abs(r))


def gminmod_phi(r, theta):
    return np.maximum(0., np.minimum(np.minimum(theta*r, theta), 0.5 * (1+r)))


def reconstruct_face_muscl(u):
    # Assume u has two layers of ghost cells
    u_diff = u[1:] - u[:-1]

    # Compute ratio of differences, avoiding division by zero
    nom = np.where(np.abs(u_diff[1:]) > 0, u_diff[1:], 1e-16)
    denom = np.where(np.abs(u_diff[:-1]) > 0, u_diff[:-1], 1e-16)
    rp = nom/denom
    symmetric = True

    if args.limiter == 'koren':
        symmetric = False
        phi_L = koren_phi(rp)
        phi_R = koren_phi(1/rp)
    elif args.limiter == 'vanleer':
        phi = vanleer_phi(rp)
    elif args.limiter == 'gminmod':
        phi = gminmod_phi(rp, args.gminmod_theta)
    elif args.limiter == 'dummy':
        phi = rp
    else:
        raise ValueError('Unknown limiter ' + args.limiter)

    if symmetric:
        # Left, right reconstructed values
        u_L = u[1:-2] + 0.5 * phi[:-1] * u_diff[0:-2]
        u_R = u[2:-1] - 0.5 * phi[1:] * u_diff[1:-1]
    else:
        u_L = u[1:-2] + 0.5 * phi_L[:-1] * u_diff[0:-2]
        u_R = u[2:-1] - 0.5 * phi_R[1:] * u_diff[2:]

    return u_L, u_R


def field_at_cell_faces(rho, dx, E_bg):
    n = len(rho)
    E = np.zeros(n+1)

    # Start with a guess E[0] = 0.
    E[1:] = np.cumsum(rho) * args.density_normalization * \
        dx * c.e / c.epsilon_0

    if args.bc_phi == 'neumann':
        E = E + E_bg
    elif args.bc_phi == 'dirichlet':
        # The total potential difference should be -E_bg * L
        current_delta_V = -(np.sum(E[1:-1]) + 0.5 * (E[0] + E[-1])) * dx
        E_correction = (E_bg * args.L + current_delta_V)/args.L
        E = E + E_correction

    return E


def implicit_transport_residual(u_new, u_old, dt):
    residual = np.zeros_like(u_new)

    # Extract electron and ion density
    N = u_new.size//2
    e0, e1 = u_old[:N], u_new[:N]
    i0, i1 = u_old[N:], u_new[N:]

    # Compute fields at cell faces and cell centers
    E0 = field_at_cell_faces(i0 - e0, dx, args.E_bg)
    E1 = field_at_cell_faces(i1 - e1, dx, args.E_bg)
    E0_cc = 0.5 * (E0[1:] + E0[:-1])
    E1_cc = 0.5 * (E1[1:] + E1[:-1])

    # Add ghost cells
    g = 2
    e0, i0 = add_ghost_cells(e0, g), add_ghost_cells(i0, g)
    e1, i1 = add_ghost_cells(e1, g), add_ghost_cells(i1, g)

    ve0, vi0 = -args.mu * E0, args.mu_ion * E0
    ve1, vi1 = -args.mu * E1, args.mu_ion * E1

    # Compute source term
    src0 = args.mu * E0_cc * get_alpha(E0_cc) * e0[g:-g]
    src1 = args.mu * E1_cc * get_alpha(E1_cc) * e1[g:-g]

    theta = args.theta

    # MUSCL scheme

    # Reconstruct left and right values at cell faces
    e0L, e0R = reconstruct_face_muscl(e0)
    e1L, e1R = reconstruct_face_muscl(e1)
    i0L, i0R = reconstruct_face_muscl(i0)
    i1L, i1R = reconstruct_face_muscl(i1)

    # Kurganov-Tadmor / Lax Friedrich approximation
    flux_e0 = 0.5 * (ve0*e0L + ve0*e0R - np.abs(ve0) * (e0R-e0L))
    flux_e1 = 0.5 * (ve1*e1L + ve1*e1R - np.abs(ve1) * (e1R-e1L))
    flux_i0 = 0.5 * (vi0*i0L + vi0*i0R - np.abs(vi0) * (i0R-i0L))
    flux_i1 = 0.5 * (vi1*i1L + vi1*i1R - np.abs(vi1) * (i1R-i1L))

    # Residual due to electron fluxes
    residual[:N] = e1[g:-g] - e0[g:-g] \
        + theta * dt/dx * (flux_e1[1:] - flux_e1[:-1]) \
        + (1-theta) * dt/dx * (flux_e0[1:] - flux_e0[:-1])

    # Add electron diffusion
    fac = args.D * dt/dx**2
    residual[:N] += \
        - theta * fac * (e1[g-1:-g-1] - 2 * e1[g:-g] + e1[g+1:-g+1]) \
        - (1-theta) * fac * (e0[g-1:-g-1] - 2 * e0[g:-g] + e0[g+1:-g+1])

    # Residual due to ion fluxes
    residual[N:] = i1[g:-g] - i0[g:-g] \
        + theta * dt/dx * (flux_i1[1:] - flux_i1[:-1]) \
        + (1-theta) * dt/dx * (flux_i0[1:] - flux_i0[:-1])

    # Add source terms
    residual[:N] += -theta * dt * src0 - (1-theta) * dt * src1
    residual[N:] += -theta * dt * src0 - (1-theta) * dt * src1

    return residual


def shock_solution(x, t):
    u = np.where(np.logical_and(x >= 0.4 * args.L, x <= 0.6 * args.L),
                 args.n0/args.density_normalization, 0.0)
    return u


def get_u0(test, t0):
    if test == 'shock':
        e0 = shock_solution(x, t0)
        i0 = e0
    else:
        raise ValueError('Unknown test ' + test)

    return np.hstack([e0, i0])


# Define helper function for Newton-Krylov method
def my_func(u_new):
    return implicit_transport_residual(u_new, u_old, dt)


t = 0.0
u = get_u0(args.test, t)
u_init = u.copy()

t0 = time.perf_counter()
while (t < args.T):
    u_old = u
    u_guess = u
    sol = newton_krylov(my_func, u_guess, method='lgmres',
                        verbose=args.verbose, f_tol=args.f_tol)
    u = sol
    t += dt
t1 = time.perf_counter()

print(f'CPU-time: {t1 - t0:.2e}')

N = u.size//2
E_face = field_at_cell_faces(u[N:] - u[:N], dx, args.E_bg)
E_cell_center = 0.5 * (E_face[1:] + E_face[:-1])

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(x, sol[:N]*args.density_normalization, label='n_e')
ax[0].plot(x, sol[N:]*args.density_normalization, label='n_i')
ax[0].plot(x, u_init[:N]*args.density_normalization, '--',
           label='initial state')
ax[0].legend()
ax[1].plot(x, E_cell_center, label='field')
ax[1].hlines(args.E_bg, x.min(), x.max(), colors=['gray'],
             ls='--', label='E_bg')
ax[1].legend()
plt.show()
