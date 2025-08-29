# NumericalMethod.py
import streamlit as st
import numpy as np
import sympy as sp

# ---------------- NUMERICAL METHODS ---------------- #

# 1. Gauss Elimination
def gauss_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    
    for i in range(n):
        # Pivoting
        max_row = np.argmax(abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            raise ValueError("Singular Matrix detected!")
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Elimination
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# 2. Lagrange Interpolation
def lagrange_interpolation(x_vals, y_vals, x):
    total = 0
    n = len(x_vals)
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        total += term
    return total

# 3. Newton's Divided Difference
def newton_divided_difference(x_vals, y_vals, x):
    n = len(x_vals)
    coef = np.zeros([n, n])
    coef[:,0] = y_vals
    for j in range(1, n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_vals[i+j] - x_vals[i])
    result = coef[0,0]
    product = 1.0
    for j in range(1, n):
        product *= (x - x_vals[j-1])
        result += coef[0][j] * product
    return result

# 4. Runge-Kutta 4th Order
def runge_kutta_4(expr, x0, y0, h, n):
    x, y = sp.symbols('x y')
    f = sp.lambdify((x, y), expr, 'numpy')
    results = [(x0, y0)]
    xi, yi = x0, y0
    for _ in range(n):
        k1 = h * f(xi, yi)
        k2 = h * f(xi + h/2, yi + k1/2)
        k3 = h * f(xi + h/2, yi + k2/2)
        k4 = h * f(xi + h, yi + k3)
        yi += (k1 + 2*k2 + 2*k3 + k4) / 6
        xi += h
        results.append((xi, yi))
    return results

# ---------------- STREAMLIT APP ---------------- #
st.title("Numerical Methods Calculator")

method = st.sidebar.selectbox("Select Method", 
                              ["Gauss Elimination", "Lagrange Interpolation", 
                               "Newton Interpolation", "Runge-Kutta 4th Order"])

# ---------------- GAUSS ---------------- #
if method == "Gauss Elimination":
    st.header("Gauss Elimination Method")
    A_str = st.text_area("Enter coefficient matrix A (comma-separated rows):", "2,1\n5,7")
    b_str = st.text_input("Enter constants vector b (comma-separated):", "11,13")
    
    if st.button("Solve"):
        try:
            A = np.array([list(map(float, row.split(','))) for row in A_str.strip().splitlines()])
            b = np.array(list(map(float, b_str.strip().split(','))))
            x = gauss_elimination(A, b)
            st.success(f"Solution: {x}")
        except Exception as e:
            st.error(e)

# ---------------- LAGRANGE ---------------- #
elif method == "Lagrange Interpolation":
    st.header("Lagrange Interpolation")
    x_vals = st.text_input("Enter x values (comma-separated):", "1,2,3,4")
    y_vals = st.text_input("Enter y values (comma-separated):", "1,4,9,16")
    val = st.text_input("Enter x to interpolate:", "2.5")
    
    if st.button("Interpolate"):
        try:
            x_list = list(map(float, x_vals.split(',')))
            y_list = list(map(float, y_vals.split(',')))
            val_num = float(val)
            result = lagrange_interpolation(x_list, y_list, val_num)
            st.success(f"Interpolated Value: {result}")
        except Exception as e:
            st.error(e)

# ---------------- NEWTON ---------------- #
elif method == "Newton Interpolation":
    st.header("Newton's Divided Difference")
    x_vals = st.text_input("Enter x values (comma-separated):", "1,2,3,4")
    y_vals = st.text_input("Enter y values (comma-separated):", "1,4,9,16")
    val = st.text_input("Enter x to interpolate:", "2.5")
    
    if st.button("Interpolate"):
        try:
            x_list = list(map(float, x_vals.split(',')))
            y_list = list(map(float, y_vals.split(',')))
            val_num = float(val)
            result = newton_divided_difference(x_list, y_list, val_num)
            st.success(f"Interpolated Value: {result}")
        except Exception as e:
            st.error(e)

# ---------------- RUNGE-KUTTA ---------------- #
elif method == "Runge-Kutta 4th Order":
    st.header("Runge-Kutta 4th Order Method")
    expr_str = st.text_input("Enter dy/dx = f(x,y):", "x + y")
    x0 = st.text_input("Initial x0:", "0")
    y0 = st.text_input("Initial y0:", "1")
    h = st.text_input("Step size h:", "0.1")
    n = st.text_input("Number of steps n:", "10")
    
    if st.button("Solve"):
        try:
            expr = sp.sympify(expr_str)
            x0_num = float(x0)
            y0_num = float(y0)
            h_num = float(h)
            n_num = int(n)
            results = runge_kutta_4(expr, x0_num, y0_num, h_num, n_num)
            for xi, yi in results:
                st.write(f"x = {xi:.4f}, y = {yi:.4f}")
        except Exception as e:
            st.error(e)
