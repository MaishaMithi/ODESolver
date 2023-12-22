import matplotlib.pyplot as plt

def user_defined_function(x, y, expression):
    return eval(expression, {"__builtins__": None}, {"x": x, "y": y})

def euler_method(x0, y0, h, target_x, function):
    x_values = [x0]
    y_values = [y0]

    while x_values[-1] < target_x:
        x_n = x_values[-1]
        y_n = y_values[-1]
        y_n1 = y_n + h * function(x_n, y_n)

        x_values.append(x_n + h)
        y_values.append(y_n1)

    return x_values, y_values

def get_user_input():
    expression = input("Enter the function in terms of x and y (e.g., x - y): ")
    function = lambda x, y: user_defined_function(x, y, expression)

    try:
        x0 = float(input("Enter initial x: "))
        y0 = float(input("Enter initial y: "))
        h = float(input("Enter step size, h: "))
        target_x = float(input("Enter target x value: "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        exit()

    return expression, function, x0, y0, h, target_x

def main():
    expression, function, x0, y0, h, target_x = get_user_input()

    # Perform Euler's Method
    x_values, y_values = euler_method(x0, y0, h, target_x, function)

    # Tabulated results
    tabulated_results = list(zip(x_values, y_values))

    # Display information about the solution
    print("\nBehavior of the Solution Curve:")
    print("--------------------------------------")
    print(f"The solution curve is approximated using Euler's Method.")
    print(f"The function is: y' = {expression}")
    print(f"Initial conditions: x0 = {x0}, y0 = {y0}")
    print(f"Step size, h: {h}")
    print(f"Target x value: {target_x}")

    # Display tabulated solutions
    print("\nTabulated Solutions (Euler's Method):")
    print("--------------------------------------")
    print("  x\t\t  y")
    print("--------------------------------------")
    for x, y in tabulated_results:
        print(f"{x:.3f}\t\t{y:.3f}")

    # Plot the solution curve
    plt.plot(x_values, y_values, label=f'y\' = {expression}')
    plt.scatter([target_x], [y_values[-1]], color='red')  # Highlight the point at target_x
    plt.title("Solution Curve using Euler's Method")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display the final result
    print(f"\nWhen x is = {target_x}, the estimated y value is = {y_values[-1]:.3f}")

    # Pass all the required data to the results dictionary
    results = {
        'expression': expression,
        'x0': x0,
        'y0': y0,
        'h': h,
        'target_x': target_x,
        'tabulated_results': tabulated_results,
        # Add other result information if needed
    }

    return results

if __name__ == "__main__":
    main()
