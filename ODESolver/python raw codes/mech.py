import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator_ode(t, y, m, c, k):
    """
    ODE for a damped harmonic oscillator.
    dy/dt = v
    dv/dt = (-c*v - k*x) / m
    """
    x, v = y
    dydt = [v, (-c*v - k*x) / m]
    return dydt

def euler_method_mech(func, y0, t):
    """
    Euler's method for solving ODEs.
    """
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * np.array(func(t[i-1], y[i-1]))
    return y

def heun_method_mech(func, y0, t):
    """
    Heun's method (2nd order Runge-Kutta) for solving ODEs.
    """
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = np.array(func(t[i-1], y[i-1]))
        k2 = np.array(func(t[i-1] + h, y[i-1] + h * k1))
        y[i] = y[i-1] + (h / 2) * (k1 + k2)
    return y

def runge_kutta_4th_order_mech(func, y0, t):
    """
    4th order Runge-Kutta method for solving ODEs.
    """
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = np.array(func(t[i-1], y[i-1]))
        k2 = np.array(func(t[i-1] + h/2, y[i-1] + (h/2) * k1))
        k3 = np.array(func(t[i-1] + h/2, y[i-1] + (h/2) * k2))
        k4 = np.array(func(t[i-1] + h, y[i-1] + h * k3))
        y[i] = y[i-1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y

# User input
m = float(input("Enter mass (m): "))
c = float(input("Enter damping coefficient (c): "))
k = float(input("Enter spring constant (k): "))
x0 = float(input("Enter initial displacement (x0): "))
v0 = float(input("Enter initial velocity (v0): "))
t_end = float(input("Enter end time (t_end): "))
dt = float(input("Enter time step (dt): "))

# Define the ODE function
ode_func = lambda t, y: harmonic_oscillator_ode(t, y, m, c, k)

# Time points
t = np.arange(0, t_end, dt)

# Initial conditions
y0 = [x0, v0]

# Solve the ODE using different methods
euler_sol = euler_method_mech(ode_func, y0, t)
heun_sol = heun_method_mech(ode_func, y0, t)
rk4_sol = runge_kutta_4th_order_mech(ode_func, y0, t)

# Plot the solutions
plt.plot(t, euler_sol[:, 0], label='Euler Method')
plt.plot(t, heun_sol[:, 0], label='Heun Method')
plt.plot(t, rk4_sol[:, 0], label='4th Order RK Method')
plt.xlabel('Time')
plt.ylabel('Displacement (x)')
plt.legend()
plt.title('Damped Harmonic Oscillator - Different ODE Solvers')
plt.show()

# Calculate and plot the error estimation
error_euler = np.abs(euler_sol[:, 0] - rk4_sol[:, 0])
error_heun = np.abs(heun_sol[:, 0] - rk4_sol[:, 0])

plt.plot(t, error_euler, label='Euler Method Error')
plt.plot(t, error_heun, label='Heun Method Error')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.title('Error Estimation')
plt.show()
