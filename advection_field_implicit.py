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
parser.add_argument('-dt', type=float, default=1e-10, help='Time step (s)')
parser.add_argument('-mu', type=float, default=0.03, help='Mobility (m2/Vs)')
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

print(f'Time step:       {dt:.2e}')


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
    E[1:] = np.cumsum(rho) * args.density_normalization * dx * c.e / c.epsilon_0

    # The total potential difference should be -E_bg * L
    correction = np.sum(E[1:]) * dx + E_bg * dx * n
    E = E - correction

    return E + E_bg


def implicit_transport_residual(u_new, u_old, dt):
    residual = np.zeros_like(u_new)

    # Assume periodic boundary conditions
    g = 2
    u0 = add_ghost_cells(u_old, g)
    u1 = add_ghost_cells(u_new, g)

    v0 = -args.mu * field_at_cell_faces(u_ion - u_old, dx, args.E_bg)
    v1 = -args.mu * field_at_cell_faces(u_ion - u_new, dx, args.E_bg)

    theta = args.theta

    # MUSCL scheme

    # Reconstruct left and right values at cell faces
    u0L, u0R = reconstruct_face_muscl(u0)
    u1L, u1R = reconstruct_face_muscl(u1)

    # Kurganov-Tadmor / Lax Friedrich approximation
    flux0 = 0.5 * (v0*u0L + v0*u0R - np.abs(v0) * (u0R-u0L))
    flux1 = 0.5 * (v1*u1L + v1*u1R - np.abs(v1) * (u1R-u1L))

    residual = u1[g:-g] - u0[g:-g] \
        + theta * dt/dx * (flux1[1:] - flux1[:-1]) \
        + (1-theta) * dt/dx * (flux0[1:] - flux0[:-1])

    return residual


def shock_solution(x, t):
    u = np.where(np.logical_and(x >= 0.4 * args.L, x <= 0.6 * args.L),
                 args.n0/args.density_normalization, 0.0)
    return u


def get_u0(test, t0):
    if test == 'shock':
        u0 = shock_solution(x, t0)
    else:
        raise ValueError('Unknown test ' + test)

    return u0


# Define helper function for Newton-Krylov method
def my_func(u_new):
    return implicit_transport_residual(u_new, u_old, dt)


t = 0.0
u = get_u0(args.test, t)
u_ion = np.roll(u, 5)

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

E_face = field_at_cell_faces(u_ion - u, dx, args.E_bg)
E_cell_center = 0.5 * (E_face[1:] + E_face[:-1])

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(x, sol*args.density_normalization, label='solution')
ax[0].plot(x, u_ion*args.density_normalization, label='initial state')
ax[0].legend()
ax[1].plot(x, E_cell_center, label='field')
ax[1].hlines(args.E_bg, x.min(), x.max(), colors=['gray'],
             ls='--', label='E_bg')
ax[1].legend()
plt.show()
