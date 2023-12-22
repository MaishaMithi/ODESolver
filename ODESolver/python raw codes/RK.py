import matplotlib.pyplot as plt

def user_defined_function(x, y, expression):
    return eval(expression)

def runge_kutta_method(f, x0, y0, h, target_x):
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

def main():
    # Get user input for the function and initial conditions
    expression = input("Enter the function in terms of x and y (e.g., x - y): ")
    function = lambda x, y: user_defined_function(x, y, expression)

    x0 = float(input("Enter initial x: "))
    y0 = float(input("Enter initial y: "))
    h = float(input("Enter step size, h: "))
    target_x = float(input("Enter target x value: "))

    # Run Runge-Kutta method
    x_values_rk, y_values_rk = runge_kutta_method(function, x0, y0, h, target_x)

    # Display information about the solution
    print("\nBehavior of the Solution Curve:")
    print("--------------------------------------")
    print(f"The solution curve is approximated using the Runge-Kutta Method.")
    print(f"The function is: y' = {expression}")
    print(f"Initial conditions: x0 = {x0}, y0 = {y0}")
    print(f"Step size, h: {h}")
    print(f"Target x value: {target_x}")

    # Display tabulated solutions
    print("\nTabulated Solutions (Runge-Kutta Method):")
    print("--------------------------------------")
    print("  x\t\t  y")
    print("--------------------------------------")
    for x, y in zip(x_values_rk, y_values_rk):
        print(f"{x:.3f}\t\t{y:.3f}")

    # Plot the solution curve
    plt.plot(x_values_rk, y_values_rk, label=f'y\' = {expression}')
    plt.scatter([target_x], [y_values_rk[-1]], color='red')  # Highlight the point at target_x
    plt.title("Solution Curve using Runge-Kutta Method")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display the final result
    print(f"\nWhen x is = {target_x}, the estimated y value is = {y_values_rk[-1]:.3f}")

if __name__ == "__main__":
    main()
