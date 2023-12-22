import numpy as np
import matplotlib.pyplot as plt

def chemical_kinetics_ode(t, A, k):
    dAdt = -k * A
    return dAdt

def euler_method(func, y0, t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * func(t[i-1], y[i-1])
    return y

def heun_method(func, y0, t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = func(t[i-1], y[i-1])
        k2 = func(t[i-1] + h, y[i-1] + h * k1)
        y[i] = y[i-1] + (h / 2) * (k1 + k2)
    return y

def runge_kutta_4th_order(func, y0, t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = func(t[i-1], y[i-1])
        k2 = func(t[i-1] + h/2, y[i-1] + (h/2) * k1)
        k3 = func(t[i-1] + h/2, y[i-1] + (h/2) * k2)
        k4 = func(t[i-1] + h, y[i-1] + h * k3)
        y[i] = y[i-1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y

if __name__ == "__main__":
    # User input for chemical kinetics
    A0 = float(input("Enter initial concentration of A: "))
    k = float(input("Enter rate constant (k) for chemical kinetics: "))
    t_end_chem = float(input("Enter end time for chemical kinetics (t_end): "))
    dt_chem = float(input("Enter time step for chemical kinetics (dt): "))

    # Time points for chemical kinetics
    t_chem = np.arange(0, t_end_chem, dt_chem)

    # Solve chemical kinetics ODE using different methods
    A_rk4 = runge_kutta_4th_order(lambda t, A: chemical_kinetics_ode(t, A, k), A0, t_chem)
    A_euler = euler_method(lambda t, A: chemical_kinetics_ode(t, A, k), A0, t_chem)
    A_heun = heun_method(lambda t, A: chemical_kinetics_ode(t, A, k), A0, t_chem)

    # Calculate errors with respect to RK4
    error_euler = np.abs(A_rk4 - A_euler)
    error_heun = np.abs(A_rk4 - A_heun)

    # Plot chemical kinetics results and errors
    plt.figure(figsize=(12, 8))

    # Plot numerical solutions
    plt.subplot(2, 1, 1)
    plt.plot(t_chem, A_euler, label='Euler Method')
    plt.plot(t_chem, A_heun, label='Heun Method')
    plt.plot(t_chem, A_rk4, label='4th Order RK Method', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('[A] (Concentration)')
    plt.legend()
    plt.title('Chemical Kinetics - Different ODE Solvers')

    # Plot errors
    plt.subplot(2, 1, 2)
    plt.plot(t_chem, error_euler, label='Euler Method')
    plt.plot(t_chem, error_heun, label='Heun Method')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.title('Error Estimation (with respect to 4th Order RK)')

    plt.tight_layout()
    plt.show()
