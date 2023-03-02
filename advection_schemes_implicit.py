#!/usr/bin/env python3

import numpy as np
from scipy.optimize import newton_krylov
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Numerical tests for scalar advection equation')
parser.add_argument('-scheme', nargs='+', type=str, help='Transport scheme(s)')
parser.add_argument('-limiter', type=str, default='minmod', help='Limiter')
parser.add_argument('-N', type=int, default=51, help='Number of grid points')
parser.add_argument('-cfl', type=float, default=1., help='CFL number')
parser.add_argument('-theta', type=float, default=0.5,
                    help='Theta parameter for time integration')
parser.add_argument('-test', type=str, default='sin-wave', help='Type of test')
parser.add_argument('-f_tol', type=float, help='Solver absolute tolerance')
parser.add_argument('-verbose', action='store_true')
parser.add_argument('-T', type=float, default=1.0,
                    help='End time (1 equals one period)')
parser.add_argument('-v', type=float, default=1.0,
                    help='Velocity')
args = parser.parse_args()

L = 1.0
dx = L / args.N
x = np.linspace(0.5*dx, L-0.5*dx, args.N)
v = args.v
dt = args.cfl * dx/abs(v)
t_end = args.T * L/abs(v)
n_steps = int(np.ceil(t_end/dt))
dt = t_end/n_steps

print(f'Number of steps: {n_steps}')
print(f'Time step:       {dt:.2e}')


def add_periodic_ghost_cells(u, g=2):
    N = u.size
    ug = np.zeros(N+2*g)
    ug[g:-g] = u
    ug[:g], ug[-g:] = u[-g:], u[:g]
    return ug


def koren_phi(r):
    t = np.minimum(1/3 + 2*r/3, 2.)
    t = np.minimum(2*r, t)
    t = np.maximum(0., t)
    return t


def vanleer_phi(r):
    return (r + np.abs(r))/(1 + np.abs(r))


def minmod_phi(r):
    return np.maximum(0., np.minimum(1.0, r))


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
    elif args.limiter == 'minmod':
        phi = minmod_phi(rp)
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


def reconstruct_face_weno3(u):
    # Assume two ghost cells
    g = 2
    N_face = len(u) - 2*g + 1
    eps = 1e-6
    p = 2

    C = np.array([1/3., 2/3.])
    alpha_L = np.zeros([2, N_face])
    alpha_R = np.zeros([2, N_face])
    alpha_L[0] = C[0]/((u[g-2:-g-1] - u[g-1:-g])**2 + eps)**p
    alpha_L[1] = C[1]/((u[g-1:-g] - u[g:-g+1])**2 + eps)**p

    alpha_R[0] = C[0]/((u[g+1:] - u[g:-g+1])**2 + eps)**p
    alpha_R[1] = C[1]/((u[g-1:-g] - u[g:-g+1])**2 + eps)**p
    w_L = alpha_L / alpha_L.sum(axis=0)
    w_R = alpha_R / alpha_R.sum(axis=0)

    # Approximation at cell faces
    s0 = -0.5 * u[g-2:-g-1] + 1.5 * u[g-1:-g]
    s1 = 0.5 * u[g-1:-g] + 0.5 * u[g:-g+1]
    s2 = 1.5 * u[g:-g+1] - 0.5 * u[g+1:]

    u_L = w_L[0] * s0 + w_L[1] * s1
    u_R = w_R[0] * s2 + w_R[1] * s1

    return u_L, u_R


def implicit_transport_residual(u_new, u_old, dt, scheme):
    residual = np.zeros_like(u_new)

    # Assume periodic boundary conditions
    g = 2
    u1 = add_periodic_ghost_cells(u_new, g)
    u0 = add_periodic_ghost_cells(u_old, g)

    theta = args.theta
    k0 = (1-theta) * v * dt/dx
    k1 = theta * v * dt/dx

    if scheme == 'upwind':
        # First order upwind
        # Depending on velocity, shift scheme by one index
        n = 0 if v > 0 else 1

        residual = u1[g:-g] - u0[g:-g] + \
            k1 * (u1[g+n:-g+n] - u1[g-1+n:-g-1+n]) + \
            k0 * (u0[g+n:-g+n] - u0[g-1+n:-g-1+n])
    elif scheme == 'upwind2':
        # Second order upwind
        if v > 0:
            residual = u1[g:-g] - u0[g:-g] + \
                k1 * (1.5*u1[g:-g] - 2*u1[g-1:-g-1] + 0.5*u1[g-2:-g-2]) + \
                k0 * (1.5*u0[g:-g] - 2*u0[g-1:-g-1] + 0.5*u0[g-2:-g-2])
        else:
            residual = u1[g:-g] - u0[g:-g] + \
                k1 * (-1.5*u1[g:-g] + 2*u1[g+1:-g+1] - 0.5*u1[g+2:]) + \
                k0 * (-1.5*u0[g:-g] + 2*u0[g+1:-g+1] - 0.5*u0[g+2:])
    elif scheme == 'central-diff':
        # Central difference scheme
        residual = u1[g:-g] - u0[g:-g] \
            + k1 * 0.5 * (u1[g+1:-g+1] - u1[g-1:-g-1]) \
            + k0 * 0.5 * (u0[g+1:-g+1] - u0[g-1:-g-1])
    elif scheme == 'compact':
        # Abarbanel-Kumar compact scheme
        residual = u1[g:-g] - u0[g:-g] \
            + k1 * 0.5 * (u1[g+1:-g+1] - u1[g-1:-g-1]) \
            + k0 * 0.5 * (u0[g+1:-g+1] - u0[g-1:-g-1]) \
            + 1/6. * (u1[g+1:-g+1] - 2 * u1[g:-g] + u1[g-1:-g-1]) \
            - 1/6. * (u0[g+1:-g+1] - 2 * u0[g:-g] + u0[g-1:-g-1])
    elif scheme == 'limited-upwind':
        # Slope-limited upwind scheme
        # Whether to use the left or right reconstructed values
        LR_index = 0 if v > 0 else 1
        u0f = reconstruct_face_muscl(u0)[LR_index]
        u1f = reconstruct_face_muscl(u1)[LR_index]

        residual = u1[g:-g] - u0[g:-g] \
            + k1 * (u1f[1:] - u1f[:-1]) \
            + k0 * (u0f[1:] - u0f[:-1])
    elif scheme == 'muscl':
        # Reconstruct left and right values at cell faces
        u0L, u0R = reconstruct_face_muscl(u0)
        u1L, u1R = reconstruct_face_muscl(u1)

        # Kurganov-Tadmor / Lax Friedrich approximation
        flux0 = 0.5 * (v*u0L + v*u0R - np.abs(v) * (u0R-u0L))
        flux1 = 0.5 * (v*u1L + v*u1R - np.abs(v) * (u1R-u1L))

        residual = u1[g:-g] - u0[g:-g] \
            + theta * dt/dx * (flux1[1:] - flux1[:-1]) \
            + (1-theta) * dt/dx * (flux0[1:] - flux0[:-1])
    elif scheme == 'weno3':
        # Third order WENO scheme
        # Whether to use the left or right reconstructed values
        LR_index = 0 if v > 0 else 1
        sw0 = reconstruct_face_weno3(u0)[LR_index]
        sw1 = reconstruct_face_weno3(u1)[LR_index]

        residual = u1[g:-g] - u0[g:-g] + \
            k1 * (sw1[1:] - sw1[:-1]) + k0 * (sw0[1:] - sw0[:-1])
    else:
        raise ValueError('Unknown transport scheme ' + args.scheme)

    return residual


def shock_solution(x, t):
    q = np.remainder(x - v*t, L)
    u = np.where(np.logical_and(q >= 0.4 * L, q <= 0.6 * L), 1.0, 0.0)
    return u


def sin_solution(x, t):
    q = np.remainder(x - v*t, L)
    u = np.sin(2 * np.pi * q)
    return u


def sin2_solution(x, t):
    q = np.remainder(x - v*t, L)
    u = np.sin(2 * np.pi * q)**2
    return u


def get_u0_and_solution(test, t0, t1):
    if test == 'shock':
        u0 = shock_solution(x, t0)
        u1 = shock_solution(x, t1)
    elif test == 'sin':
        u0 = sin_solution(x, t0)
        u1 = sin_solution(x, t1)
    elif test == 'sin2':
        u0 = sin2_solution(x, t0)
        u1 = sin2_solution(x, t1)
    else:
        raise ValueError('Unknown test ' + test)

    return u0, u1


for scheme in args.scheme:

    # Define helper function for Newton-Krylov method
    def my_func(u_new):
        return implicit_transport_residual(u_new, u_old, dt, scheme)

    t = 0.0
    u, u_sol = get_u0_and_solution(args.test, t, t_end)

    t0 = time.perf_counter()
    for i in range(n_steps):
        u_old = u
        u_guess = u
        sol = newton_krylov(my_func, u_guess, method='lgmres',
                            verbose=args.verbose, f_tol=args.f_tol)
        u = sol
        t += dt
    t1 = time.perf_counter()

    print(f'{scheme:15} CPU-time: {t1 - t0:.2e}')
    plt.plot(x, sol, label=scheme)

plt.plot(x, u_sol, '--', label='solution')
plt.legend()
plt.show()
