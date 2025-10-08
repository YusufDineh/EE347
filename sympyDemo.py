import sympy as sp

def test_sympy():
    # Test 1: Create a symbolic variable
    x = sp.Symbol('x')
    assert str(x) == 'x', "Symbol creation failed"

    # Test 2: Create a symbolic expression
    expr = 2 * x + 3
    assert str(expr) == '2*x + 3', "Symbolic expression creation failed"

    # Test 3: Calculate the derivative of the expression
    derivative_expr = sp.diff(expr, x)
    assert str(derivative_expr) == '2', "Derivative calculation failed"

    # Test 4: Solve an equation
    equation = sp.Eq(expr, 10)
    solution = sp.solve(equation, x)
    assert str(solution) == '[7/2]', "Equation solving failed"

    print("All SymPy tests passed!")

# Run the SymPy tests
test_sympy()