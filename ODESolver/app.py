import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from datetime import datetime

app = Flask(__name__)

def get_unique_filename():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{hash(str(timestamp))}"

def get_user_input(data):
    expression = data.get("expression")
    function = lambda x, y: eval(expression, {"__builtins__": None}, {"x": x, "y": y})

    try:
        x0 = float(data.get("x0"))
        y0 = float(data.get("y0"))
        h = float(data.get("h"))
        target_x = float(data.get("target_x"))
    except ValueError:
        return None, None, None, None, None, None

    return expression, function, x0, y0, h, target_x

def user_defined_function(x, y, expression):
    return eval(expression, {"__builtins__": None}, {"x": x, "y": y})

def euler_method_app(x0, y0, h, target_x, function):
    x_values = [x0]
    y_values = [y0]

    while x_values[-1] < target_x:
        x_n = x_values[-1]
        y_n = y_values[-1]
        y_n1 = y_n + h * function(x_n, y_n)

        x_values.append(x_n + h)
        y_values.append(y_n1)

    return x_values, y_values

def heun_method_app(f, x0, y0, h, target_x):
    x_values = [x0]
    y_values = [y0]

    while x_values[-1] < target_x:
        x_n = x_values[-1]
        y_n = y_values[-1]

        k1 = f(x_n, y_n)
        k2 = f(x_n + h, y_n + h * k1)

        y_n1 = y_n + 0.5 * h * (k1 + k2)

        x_values.append(x_n + h)
        y_values.append(y_n1)

    return x_values, y_values

def runge_kutta_method_app(f, x0, y0, h, target_x):
    x_values = [x0]
    y_values = [y0]

    while x_values[-1] < target_x:
        x_n = x_values[-1]
        y_n = y_values[-1]

        k1 = h * f(x_n, y_n)
        k2 = h * f(x_n + 0.5 * h, y_n + 0.5 * k1)
        k3 = h * f(x_n + 0.5 * h, y_n + 0.5 * k2)
        k4 = h * f(x_n + h, y_n + k3)

        y_n1 = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        x_values.append(x_n + h)
        y_values.append(y_n1)

    return x_values, y_values

def run_method(method, data):
    expression, function, x0, y0, h, target_x = get_user_input(data)

    if expression is None:
        return {'error': 'Invalid input. Please enter numerical values.'}

    if method == 'Euler':
        x_values, y_values = euler_method_app(x0, y0, h, target_x, function)
    elif method == 'Heun':
        x_values, y_values = heun_method_app(function, x0, y0, h, target_x)
    elif method == 'RK':
        x_values, y_values = runge_kutta_method_app(function, x0, y0, h, target_x)
    else:
        return {'error': 'Invalid method selected.'}

    plot_filename = f'static/images/app/{method}_plot_{get_unique_filename()}.png'
    plot_solution(x_values, y_values, plot_filename, expression, target_x, y_values[-1], method)

    return {
        'expression': expression,
        'h': h,
        'plot_path': plot_filename,
        'tabulated_results': list(zip(x_values, y_values)),
        'target_x': target_x,
        'x0': x0,
        'y0': y0,
        'y_values': y_values
    }

def plot_solution(x_values, y_values, plot_filename, expression, target_x, target_y, method_name):
    plt.plot(x_values, y_values, label=f"y' = {expression}")
    plt.scatter([target_x], [target_y], color='red')  # Add a red dot at the target_x point
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Solution Curve using {method_name}'s Method")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.close()

def deformation_ode(t, u, k, m, F):
    return -k/m * u + F/m

def euler_method_mat(func, y0, t, *args):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * func(t[i-1], y[i-1], *args)
    return y

def heun_method_mat(func, y0, t, *args):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = func(t[i-1], y[i-1], *args)
        k2 = func(t[i-1] + h, y[i-1] + h * k1, *args)
        y[i] = y[i-1] + (h / 2) * (k1 + k2)
    return y

def runge_kutta_4th_order_mat(func, y0, t, *args):
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

def run_material_engineering(data):
    u0 = float(data.get("u0"))
    k = float(data.get("k"))
    m = float(data.get("m"))
    F = float(data.get("F"))
    t_end = float(data.get("t_end"))
    dt = float(data.get("dt"))

    t = np.arange(0, t_end, dt)
    sol_euler = euler_method_mat(deformation_ode, u0, t, k, m, F)
    sol_heun = heun_method_mat(deformation_ode, u0, t, k, m, F)
    sol_rk4 = runge_kutta_4th_order_mat(deformation_ode, u0, t, k, m, F)

    error_euler = np.abs(sol_euler - sol_rk4)
    error_heun = np.abs(sol_heun - sol_rk4)

    plot_filename = f'static/images/mat/mat_plot_{get_unique_filename()}.png'
    plot_material_solutions(t, sol_euler, sol_heun, sol_rk4, plot_filename)

    error_plot_filename = f'static/images/mat/mat_error_plot_{get_unique_filename()}.png'
    plot_error_estimation_mat(t, error_euler, error_heun, error_plot_filename)

    final_answer = f"The final displacement after {t_end} seconds is {sol_rk4[-1]:.4f} units."

    return {
        'plot_path': plot_filename,
        'error_plot_path': error_plot_filename,
        'euler': list(zip(t, sol_euler)),
        'heun': list(zip(t, sol_heun)),
        'rk4': list(zip(t, sol_rk4)),
        'error_euler': list(zip(t, error_euler)),
        'error_heun': list(zip(t, error_heun)),
        'final_answer': final_answer,
    }

def plot_material_solutions(t, sol_euler, sol_heun, sol_rk4, plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(t, sol_euler, label='Euler Method')
    plt.plot(t, sol_heun, label='Heun Method')
    plt.plot(t, sol_rk4, label='4th Order RK Method')
    plt.xlabel('Time')
    plt.ylabel('Displacement (u)')
    plt.legend()
    plt.title('Deformation of Materials - By Different ODE')
    plt.savefig(plot_filename)
    plt.close()

def plot_error_estimation_mat(t, error_euler, error_heun, error_plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(t, error_euler, label='Euler Method Error')
    plt.plot(t, error_heun, label='Heun Method Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Error Estimation - Taking RK methods answer as accurate')
    plt.savefig(error_plot_filename)
    plt.close()

def harmonic_oscillator_ode(t, y, m, c, k):
    x, v = y
    dydt = [v, (-c * v - k * x) / m]
    return dydt

def euler_method_mech(func, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * np.array(func(t[i-1], y[i-1]))
    return y

def heun_method_mech(func, y0, t):
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

def run_mechanical_engineering(data):
    x0 = float(data.get("initial_displacement"))
    v0 = float(data.get("initial_velocity"))
    c = float(data.get("damping"))
    k = float(data.get("spring"))
    m = float(data.get("mass"))
    t_end = float(data.get("end_time"))
    dt = float(data.get("time_step"))

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
    plot_filename = f'static/images/mech/mech_plot_{get_unique_filename()}.png'
    plot_mechanical_solutions(t, euler_sol, heun_sol, rk4_sol, plot_filename)

    # Calculate and plot the error estimation
    error_euler = np.abs(euler_sol[:, 0] - rk4_sol[:, 0])
    error_heun = np.abs(heun_sol[:, 0] - rk4_sol[:, 0])

    error_plot_filename = f'static/images/mech/mech_error_plot_{get_unique_filename()}.png'
    plot_error_estimation_mech(t, error_euler, error_heun, error_plot_filename)

    final_answer = f"The final displacement after {t_end} seconds is {rk4_sol[-1, 0]:.4f} units."

    return {
        'plot_path': plot_filename,
        'error_plot_path': error_plot_filename,
        'euler': list(zip(t, euler_sol[:, 0])),
        'heun': list(zip(t, heun_sol[:, 0])),
        'rk4': list(zip(t, rk4_sol[:, 0])),
        'error_euler': list(zip(t, error_euler)),
        'error_heun': list(zip(t, error_heun)),
        'final_answer': final_answer,
    }

def plot_mechanical_solutions(t, euler_sol, heun_sol, rk4_sol, plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(t, euler_sol[:, 0], label='Euler Method')
    plt.plot(t, heun_sol[:, 0], label='Heun Method')
    plt.plot(t, rk4_sol[:, 0], label='4th Order RK Method')
    plt.xlabel('Time')
    plt.ylabel('Displacement (x)')
    plt.legend()
    plt.title('Damped Harmonic Oscillator - By Different ODE')
    plt.savefig(plot_filename)
    plt.close()

def plot_error_estimation_mech(t, error_euler, error_heun, error_plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(t, error_euler, label='Euler Method Error')
    plt.plot(t, error_heun, label='Heun Method Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Error Estimation - Taking RK methods answer as accurate')
    plt.savefig(error_plot_filename)
    plt.close()

def chemical_kinetics_ode(t, A, k):
    dAdt = -k * A
    return dAdt

def euler_method_chem(func, y0, t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * func(t[i-1], y[i-1])
    return y

def heun_method_chem(func, y0, t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = func(t[i-1], y[i-1])
        k2 = func(t[i-1] + h, y[i-1] + h * k1)
        y[i] = y[i-1] + (h / 2) * (k1 + k2)
    return y

def runge_kutta_4th_order_chem(func, y0, t):
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

def run_chemical_kinetics(data):
    A0 = float(data.get("A0"))
    k = float(data.get("k"))
    t_end_chem = float(data.get("t_end_chem"))
    dt_chem = float(data.get("dt_chem"))

    t_chem = np.arange(0, t_end_chem, dt_chem)
    A_rk4 = runge_kutta_4th_order_chem(lambda t, A: chemical_kinetics_ode(t, A, k), A0, t_chem)
    A_euler = euler_method_chem(lambda t, A: chemical_kinetics_ode(t, A, k), A0, t_chem)
    A_heun = heun_method_chem(lambda t, A: chemical_kinetics_ode(t, A, k), A0, t_chem)

    error_euler = np.abs(A_rk4 - A_euler)
    error_heun = np.abs(A_rk4 - A_heun)

    plot_filename = f'static/images/chem/chem_plot_{get_unique_filename()}.png'
    plot_chemical_kinetics(t_chem, A_euler, A_heun, A_rk4, plot_filename)

    error_plot_filename = f'static/images/chem/chem_error_plot_{get_unique_filename()}.png'
    plot_error_estimation_chem(t_chem, error_euler, error_heun, error_plot_filename)

    final_answer = f"The final concentration after {t_end_chem} seconds is {A_rk4[-1]:.4f} units."

    return {
        'plot_path': plot_filename,
        'error_plot_path': error_plot_filename,
        'euler': list(zip(t_chem, A_euler)),
        'heun': list(zip(t_chem, A_heun)),
        'rk4': list(zip(t_chem, A_rk4)),
        'error_euler': list(zip(t_chem, error_euler)),
        'error_heun': list(zip(t_chem, error_heun)),
        'final_answer': final_answer,
    }

def plot_chemical_kinetics(t, A_euler, A_heun, A_rk4, plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(t, A_euler, label='Euler Method')
    plt.plot(t, A_heun, label='Heun Method')
    plt.plot(t, A_rk4, label='4th Order RK Method')
    plt.xlabel('Time')
    plt.ylabel('[A] (Concentration)')
    plt.legend()
    plt.title('Chemical Kinetics - By Different ODE')
    plt.savefig(plot_filename)
    plt.close()

def plot_error_estimation_chem(t, error_euler, error_heun, error_plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(t, error_euler, label='Euler Method Error')
    plt.plot(t, error_heun, label='Heun Method Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Error Estimation - Taking RK methods answer as accurate')
    plt.savefig(error_plot_filename)
    plt.close()

def rc_circuit_ode(t, Vc, R, C, Vin):
    dVcdt = (1 / (R * C)) * (Vin - Vc)
    return dVcdt

def euler_method_elect(func, y0, t, *args):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * func(t[i-1], y[i-1], *args)
    return y

def heun_method_elect(func, y0, t, *args):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = func(t[i-1], y[i-1], *args)
        k2 = func(t[i-1] + h, y[i-1] + h * k1, *args)
        y[i] = y[i-1] + (h / 2) * (k1 + k2)
    return y

def runge_kutta_4th_order_elect(func, y0, t, *args):
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

def run_electrical_engineering(data):
    Vc0 = float(data.get("Vc0"))
    R = float(data.get("R"))
    C = float(data.get("C"))
    Vin = float(data.get("Vin"))
    t_end_transient = float(data.get("t_end_transient"))
    dt_transient = float(data.get("dt_transient"))

    t_transient = np.arange(0, t_end_transient, dt_transient)

    Vc_rk4 = runge_kutta_4th_order_elect(rc_circuit_ode, Vc0, t_transient, R, C, Vin)
    Vc_euler = euler_method_elect(rc_circuit_ode, Vc0, t_transient, R, C, Vin)
    Vc_heun = heun_method_elect(rc_circuit_ode, Vc0, t_transient, R, C, Vin)

    error_euler = np.abs(Vc_rk4 - Vc_euler)
    error_heun = np.abs(Vc_rk4 - Vc_heun)

    plot_filename = f'static/images/elect/elect_plot_{get_unique_filename()}.png'
    plot_transient_response(t_transient, Vc_euler, Vc_heun, Vc_rk4, plot_filename)

    error_plot_filename = f'static/images/elect/elect_error_plot_{get_unique_filename()}.png'
    plot_error_estimation_elect(t_transient, error_euler, error_heun, error_plot_filename)

    final_answer = f"The final voltage across the capacitor after {t_end_transient} seconds is {Vc_rk4[-1]:.4f} volts."

    return {
        'plot_path': plot_filename,
        'error_plot_path': error_plot_filename,
        'euler': list(zip(t_transient, Vc_euler)),
        'heun': list(zip(t_transient, Vc_heun)),
        'rk4': list(zip(t_transient, Vc_rk4)),
        'error_euler': list(zip(t_transient, error_euler)),
        'error_heun': list(zip(t_transient, error_heun)),
        'final_answer': final_answer,
    }

def plot_transient_response(t, Vc_euler, Vc_heun, Vc_rk4, plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(t, Vc_euler, label='Euler Method')
    plt.plot(t, Vc_heun, label='Heun Method')
    plt.plot(t, Vc_rk4, label='4th Order RK Method')
    plt.xlabel('Time')
    plt.ylabel('Voltage across Capacitor (Vc)')
    plt.legend()
    plt.title('RC Circuit Transient Response - Different ODE Solvers')
    plt.savefig(plot_filename)
    plt.close()

def plot_error_estimation_elect(t, error_euler, error_heun, error_plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(t, error_euler, label='Euler Method Error')
    plt.plot(t, error_heun, label='Heun Method Error')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.title('Error Estimation - Taking RK methods answer as accurate')
    plt.savefig(error_plot_filename)
    plt.close()

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/Euler', methods=['GET', 'POST'])
def euler():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON data
            data = request.get_json()
            results = run_method('Euler', data)
            return jsonify(results)
        else:
            # Handle form data
            data = request.form
            results = run_method('Euler', data)
            if 'error' in results:
                return render_template('Euler.html', error=results['error'])
            return render_template('Euler.html', results=results)

    return render_template('Euler.html', results=None)

@app.route('/Heun', methods=['GET', 'POST'])
def heun():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON data
            data = request.get_json()
            results = run_method('Heun', data)
            return jsonify(results)
        else:
            # Handle form data
            data = request.form
            results = run_method('Heun', data)
            if 'error' in results:
                return render_template('Heun.html', error=results['error'])
            return render_template('Heun.html', results=results)

    return render_template('Heun.html', results=None)

@app.route('/RK', methods=['GET', 'POST'])
def rk():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON data
            data = request.get_json()
            results = run_method('RK', data)
            return jsonify(results)
        else:
            # Handle form data
            data = request.form
            results = run_method('RK', data)
            if 'error' in results:
                return render_template('RK.html', error=results['error'])
            return render_template('RK.html', results=results)

    return render_template('RK.html', results=None)

@app.route('/mat', methods=['GET', 'POST'])
def material_engineering():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON data
            data = request.get_json()
            results = run_material_engineering(data)
            return jsonify(results)
        else:
            # Handle form data
            data = request.form
            results = run_material_engineering(data)
            if 'error' in results:
                return render_template('mat.html', error=results['error'])
            return render_template('mat.html', results=results)

    return render_template('mat.html', results=None)

@app.route('/mech', methods=['GET', 'POST'])
def mechanical_engineering():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON data
            data = request.get_json()
            results = run_mechanical_engineering(data)
            return jsonify(results)
        else:
            # Handle form data
            data = request.form
            results = run_mechanical_engineering(data)
            if 'error' in results:
                return render_template('mech.html', error=results['error'])
            return render_template('mech.html', results=results)

    return render_template('mech.html', results=None)

@app.route('/chem', methods=['GET', 'POST'])
def chemical_kinetics():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON data
            data = request.get_json()
            results = run_chemical_kinetics(data)
            return jsonify(results)
        else:
            # Handle form data
            data = request.form
            results = run_chemical_kinetics(data)
            if 'error' in results:
                return render_template('chem.html', error=results['error'])
            return render_template('chem.html', results=results)

    return render_template('chem.html', results=None)

@app.route('/elect', methods=['GET', 'POST'])
def electrical_engineering():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON data
            data = request.get_json()
            results = run_electrical_engineering(data)
            return jsonify(results)
        else:
            # Handle form data
            data = request.form
            results = run_electrical_engineering(data)
            if 'error' in results:
                return render_template('elect.html', error=results['error'])
            return render_template('elect.html', results=results)

    return render_template('elect.html', results=None)

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
    