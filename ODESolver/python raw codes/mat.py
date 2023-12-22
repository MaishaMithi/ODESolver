import numpy as np
import matplotlib.pyplot as plt

def deformation_ode(t, u, k, m, F):
    """
    ODE for deformation of materials.
    du/dt = -k/m * u + F/m
    """
    return -k/m * u + F/m

def euler_method(func, y0, t, *args):
    """
    Euler's method for solving ODEs.
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * func(t[i-1], y[i-1], *args)
    return y

def heun_method(func, y0, t, *args):
    """
    Heun's method (2nd order Runge-Kutta) for solving ODEs.
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = func(t[i-1], y[i-1], *args)
        k2 = func(t[i-1] + h, y[i-1] + h * k1, *args)
        y[i] = y[i-1] + (h / 2) * (k1 + k2)
    return y

def runge_kutta_4th_order(func, y0, t, *args):
    """
    4th order Runge-Kutta method for solving ODEs.
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = func(t[i-1], y[i-1], *args)
        k2 = func(t[i-1] + h/2, y[i-1] + h/2 * k1, *args)
        k3 = func(t[i-1] + h/2, y[i-1] + h/2 * k2, *args)
        k4 = func(t[i-1] + h, y[i-1] + h * k3, *args)
        y[i] = y[i-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

# User input for deformation problem
u0 = float(input("Enter initial displacement (u0): "))
k = float(input("Enter stiffness (k): "))
m = float(input("Enter mass (m): "))
F = float(input("Enter external force (F): "))
t_end = float(input("Enter end time (t_end): "))
dt = float(input("Enter time step (dt): "))

# Time points
t = np.arange(0, t_end, dt)

# Solve the ODE using different methods
sol_euler = euler_method(deformation_ode, u0, t, k, m, F)
sol_heun = heun_method(deformation_ode, u0, t, k, m, F)
sol_rk4 = runge_kutta_4th_order(deformation_ode, u0, t, k, m, F)

# Plot the solutions
plt.figure(figsize=(10, 6))

plt.plot(t, sol_euler, label='Euler Method')
plt.plot(t, sol_heun, label='Heun Method')
plt.plot(t, sol_rk4, label='4th Order RK Method')

plt.xlabel('Time')
plt.ylabel('Displacement (u)')
plt.legend()
plt.title('Deformation of Materials - Different ODE Solvers')

plt.show()

# Calculate and plot the error estimation
error_euler = np.abs(sol_euler - sol_rk4)
error_heun = np.abs(sol_heun - sol_rk4)

plt.figure(figsize=(10, 6))

plt.plot(t, error_euler, label='Euler Method Error')
plt.plot(t, error_heun, label='Heun Method Error')

plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.title('Error Estimation - Deformation of Materials')

plt.show()
