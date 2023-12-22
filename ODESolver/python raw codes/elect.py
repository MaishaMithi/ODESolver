import numpy as np
import matplotlib.pyplot as plt

def rc_circuit_ode(t, Vc, R, C, Vin):
    dVcdt = (1 / (R * C)) * (Vin - Vc)
    return dVcdt

def euler_method(func, y0, t, *args):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * func(t[i-1], y[i-1], *args)
    return y

def heun_method(func, y0, t, *args):
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
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = func(t[i-1], y[i-1], *args)
        k2 = func(t[i-1] + h/2, y[i-1] + (h/2) * k1, *args)
        k3 = func(t[i-1] + h/2, y[i-1] + (h/2) * k2, *args)
        k4 = func(t[i-1] + h, y[i-1] + h * k3, *args)
        y[i] = y[i-1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y

if __name__ == "__main__":
    # User input for RC circuit transient response
    Vc0 = float(input("Enter initial voltage across the capacitor (Vc0): "))
    R = float(input("Enter resistance (R) of the circuit: "))
    C = float(input("Enter capacitance (C) of the circuit: "))
    Vin = float(input("Enter input voltage (Vin): "))
    t_end_transient = float(input("Enter end time for transient response (t_end): "))
    dt_transient = float(input("Enter time step for transient response (dt): "))

    # Time points for transient response
    t_transient = np.arange(0, t_end_transient, dt_transient)

    # Solve transient response ODE using different methods
    Vc_rk4 = runge_kutta_4th_order(lambda t, Vc, R, C, Vin: rc_circuit_ode(t, Vc, R, C, Vin), Vc0, t_transient, R, C, Vin)
    Vc_euler = euler_method(lambda t, Vc, R, C, Vin: rc_circuit_ode(t, Vc, R, C, Vin), Vc0, t_transient, R, C, Vin)
    Vc_heun = heun_method(lambda t, Vc, R, C, Vin: rc_circuit_ode(t, Vc, R, C, Vin), Vc0, t_transient, R, C, Vin)

    # Calculate errors with respect to RK4
    error_euler = np.abs(Vc_rk4 - Vc_euler)
    error_heun = np.abs(Vc_rk4 - Vc_heun)

    # Plot transient response results and errors
    plt.figure(figsize=(12, 8))

    # Plot numerical solutions
    plt.subplot(2, 1, 1)
    plt.plot(t_transient, Vc_euler, label='Euler Method')
    plt.plot(t_transient, Vc_heun, label='Heun Method')
    plt.plot(t_transient, Vc_rk4, label='4th Order RK Method', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Voltage across Capacitor (Vc)')
    plt.legend()
    plt.title('RC Circuit Transient Response - Different ODE Solvers')

    # Plot errors
    plt.subplot(2, 1, 2)
    plt.plot(t_transient, error_euler, label='Euler Method')
    plt.plot(t_transient, error_heun, label='Heun Method')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.title('Error Estimation (with respect to 4th Order RK)')

    plt.tight_layout()
    plt.show()
