import tkinter as tk
from tkinter import messagebox
import numpy as np
import sympy as sp

# ---------------- Gauss Elimination ----------------
def gauss_elimination(a, b):
    n = len(b)
    a = a.astype(float)
    b = b.astype(float)

    for i in range(n):
        if a[i][i] == 0.0:
            raise ValueError("Divide by zero detected in Gauss elimination!")

        for j in range(i+1, n):
            ratio = a[j][i]/a[i][i]
            for k in range(n):
                a[j][k] = a[j][k] - ratio * a[i][k]
            b[j] = b[j] - ratio * b[i]

    x = np.zeros(n)
    x[n-1] = b[n-1]/a[n-1][n-1]

    for i in range(n-2, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= a[i][j]*x[j]
        x[i] /= a[i][i]
    return x

# ---------------- Lagrange Interpolation ----------------
def lagrange_interpolation(x_points, y_points, x_val):
    result = 0
    n = len(x_points)
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if j != i:
                term *= (x_val - x_points[j])/(x_points[i] - x_points[j])
        result += term
    return result

# ---------------- Newton’s Divided Difference ----------------
def newton_divided_diff(x_points, y_points, x_val):
    n = len(x_points)
    coef = np.zeros([n, n])
    coef[:,0] = y_points

    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1])/(x_points[i+j]-x_points[i])

    result = coef[0][0]
    product = 1.0
    for i in range(1,n):
        product *= (x_val - x_points[i-1])
        result += coef[0][i]*product
    return result

# ---------------- Runge–Kutta 4th Order ----------------
def runge_kutta(f, x0, y0, xn, n):
    h = (xn - x0) / n
    x, y = x0, y0
    for i in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
    return y

# ---------------- GUI ----------------
class NumericalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods Calculator")
        self.root.geometry("600x500")

        self.label = tk.Label(root, text="Choose a Method", font=("Arial", 14, "bold"))
        self.label.pack(pady=10)

        # Buttons
        tk.Button(root, text="Gauss Elimination", command=self.gauss_ui, width=25).pack(pady=5)
        tk.Button(root, text="Lagrange Interpolation", command=self.lagrange_ui, width=25).pack(pady=5)
        tk.Button(root, text="Newton Interpolation", command=self.newton_ui, width=25).pack(pady=5)
        tk.Button(root, text="Runge-Kutta Method", command=self.runge_kutta_ui, width=25).pack(pady=5)

    # ---------------- UI Functions ----------------
    def gauss_ui(self):
        win = tk.Toplevel(self.root)
        win.title("Gauss Elimination")

        tk.Label(win, text="Enter coefficient matrix A (comma separated rows):").pack()
        ent_a = tk.Entry(win, width=50)
        ent_a.pack()

        tk.Label(win, text="Enter constant matrix B (comma separated):").pack()
        ent_b = tk.Entry(win, width=50)
        ent_b.pack()

        def solve():
            try:
                A = np.array([list(map(float, row.split())) for row in ent_a.get().split(',')])
                B = np.array(list(map(float, ent_b.get().split())))
                x = gauss_elimination(A, B)
                messagebox.showinfo("Solution", f"X = {x}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(win, text="Solve", command=solve).pack(pady=10)

    def lagrange_ui(self):
        win = tk.Toplevel(self.root)
        win.title("Lagrange Interpolation")

        tk.Label(win, text="Enter x points (comma separated):").pack()
        ent_x = tk.Entry(win, width=50)
        ent_x.pack()

        tk.Label(win, text="Enter y points (comma separated):").pack()
        ent_y = tk.Entry(win, width=50)
        ent_y.pack()

        tk.Label(win, text="Enter x value to interpolate:").pack()
        ent_val = tk.Entry(win, width=20)
        ent_val.pack()

        def solve():
            try:
                x_points = list(map(float, ent_x.get().split(',')))
                y_points = list(map(float, ent_y.get().split(',')))
                x_val = float(ent_val.get())
                result = lagrange_interpolation(x_points, y_points, x_val)
                messagebox.showinfo("Result", f"P({x_val}) = {result}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(win, text="Interpolate", command=solve).pack(pady=10)

    def newton_ui(self):
        win = tk.Toplevel(self.root)
        win.title("Newton Interpolation")

        tk.Label(win, text="Enter x points (comma separated):").pack()
        ent_x = tk.Entry(win, width=50)
        ent_x.pack()

        tk.Label(win, text="Enter y points (comma separated):").pack()
        ent_y = tk.Entry(win, width=50)
        ent_y.pack()

        tk.Label(win, text="Enter x value to interpolate:").pack()
        ent_val = tk.Entry(win, width=20)
        ent_val.pack()

        def solve():
            try:
                x_points = list(map(float, ent_x.get().split(',')))
                y_points = list(map(float, ent_y.get().split(',')))
                x_val = float(ent_val.get())
                result = newton_divided_diff(x_points, y_points, x_val)
                messagebox.showinfo("Result", f"P({x_val}) = {result}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(win, text="Interpolate", command=solve).pack(pady=10)

    def runge_kutta_ui(self):
        win = tk.Toplevel(self.root)
        win.title("Runge-Kutta Method")

        tk.Label(win, text="Enter function f(x,y):").pack()
        ent_f = tk.Entry(win, width=50)
        ent_f.pack()

        tk.Label(win, text="Enter x0, y0, xn, n (comma separated):").pack()
        ent_vals = tk.Entry(win, width=50)
        ent_vals.pack()

        def solve():
            try:
                f_expr = ent_f.get()
                x, y = sp.symbols('x y')
                f = sp.lambdify((x,y), sp.sympify(f_expr), "numpy")

                vals = list(map(float, ent_vals.get().split(',')))
                x0, y0, xn, n = vals
                result = runge_kutta(f, x0, y0, xn, int(n))
                messagebox.showinfo("Result", f"y({xn}) = {result}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(win, text="Solve", command=solve).pack(pady=10)

# ---------------- Run ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = NumericalApp(root)
    root.mainloop()
