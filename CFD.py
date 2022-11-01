# --------------------------- #
# --CFD Assignment - VP16HK-- #
# --------------------------- #
# ---IMPORT LIB & PACKAGES--- #
# --------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.sparse import coo_matrix, linalg
# --------------------------- #
# --------END IMPORT--------- #
# --------------------------- #
# --------------------------- #
# -----------DEF------------- #
# --------------------------- #
def create_mesh(show_mesh=False):
    x_mesh, y_mesh = np.meshgrid(x, y)
    if show_mesh:
        plt.grid(color='k', linestyle='-', linewidth=1)
        plt.xticks(x)
        plt.yticks(y)
        plt.scatter(x_mesh, y_mesh, marker='.', color='r')
        plt.show()
def solve(S, D):
    return (linalg.inv(S)@D).reshape(ny, nx)
def plot_contour(z):
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    x, y = np.meshgrid(x, y)
    contourf = plt.contourf(x, y, z, 64, cmap='gist_rainbow')
    plt.contour(contourf, z)
    plt.colorbar(contourf, shrink = 1.25, orientation = 'horizontal')
    plt.show()
def cell_contour(z):
    contourf = plt.imshow(z[::-1, :] , extent = [x_min , x_max, y_min, y_max], cmap='gist_rainbow')
    plt.colorbar(contourf, shrink = 1.25, orientation = 'horizontal')
    plt.show()
# --------------------------- #
# ---------END DEF----------- #
# --------------------------- #
# --------------------------- #
# ------------MAIN----------- #
# --------------------------- #
def main():
    create_mesh(show_mesh=False)
    Center = A * ((Te - 2 * T + Tw) / dx) + B * ((Tn - 2 * T + Ts) /dy) - C * ((Te - Tw) / 2) # 1
    West = A * ((Te - 3 * T + 2 * Tw) / dx) + B * ((Tn - 2 * T + Ts) /dy) - C * ((Te + T) / 2 - Tw) # 2
    South = A * ((Te - 2 * T + Tw) / dx) + B * ((Tn - 2 * T + Ts) /dy) - C * ((Te + T) / 2 - Tw) # 3
    North = A * ((Te - 2 * T + Tw) / dx) + B * ((2 * Tn - 3 * T + Ts)/ dy) - C * ((Te - Tw) / 2) # 4
    East = -A * ((T - Tw) / dx) - C * (T - (T + Tw) / 2) + B * (Tn - 2* T + Ts) # 5
    SouthWest = South + West - Center # 6
    NorthWest = North + West - Center # 7
    SouthEast = -A * ((T - Tw) / dx) - C * (T - (T + Tw) / 2) + B *(Tn - T) - 2 * B * (T - Ts) # 8
    NorthEast = -A * ((T - Tw) / dx) - C * (T - (T + Tw) / 2) + 2 * B* (Tn - T) - B * (T - Ts) # 9
    S_data = []
    S_row = []
    S_col = []
# For cell 1 from the southwest
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], SouthWest, 'numpy')
    S_data.extend([f(1, 0, 0, 0, 0, U[C1]), f(0, 0, 0, 1, 0, U[C1]),f(0, 0, 1, 0, 0, U[C1])])
    S_row.extend([C1]*3)
    S_col.extend([C1, C1 + 1, C1 + nx])
    D[C1] = -f(0, TWw, 0, 0, TWw, U[C1])
# For cell 50 from the southeast
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], SouthEast, 'numpy')
    S_data.extend([f(1, 0, 0, 0, 0, U[C2]), f(0, 1, 0, 0, 0, U[C2]),
    f(0, 0, 1, 0, 0, U[C2])])
    S_row.extend([C2]*3)
    S_col.extend([C2, C2 - 1, C2 + nx])
    D[C2] = -f(0, 0, 0, 0, TWs, U[C2])
# For cell 50*19+1 from the northwest
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], NorthWest, 'numpy')
    S_data.extend([f(1, 0, 0, 0, 0, U[C3]), f(0, 0, 0, 1, 0, U[C3]),f(0, 0, 0, 0, 1, U[C3])])
    S_row.extend([C3]*3)
    S_col.extend([C3, C3 + 1, C3 - nx])
    D[C3] = -f(0, TWw, TWn, 0, 0, U[C3])
# For cell 50*20 from the northeast
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], NorthEast, 'numpy')
    S_data.extend([f(1, 0, 0, 0, 0, U[C4]), f(0, 1, 0, 0, 0, U[C4]),f(0, 0, 0, 0, 1, U[C4])])
    S_row.extend([C4]*3)
    S_col.extend([C4, C4 - 1, C4 - nx])
    D[C4] = -f(0, 0, TWn, 0, 0, U[C4])
# For cell from 2 to 50-1 from the south
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], South, 'numpy')
    list_s = list(range(1, nx - 1))
    for i, j in zip(list_s, list_s):
        S_data.extend([f(0, 1, 0, 0, 0, U[i]), f(1, 0, 0, 0, 0, U[i]),f(0, 0, 0, 1, 0, U[i]), f(0, 0, 1, 0, 0,U[i])])
        S_row.extend([i]*4)
        S_col.extend([j - 1, j, j + 1, j + nx])
        D[i] = -f(0, 0, 0, 0, TWs, U[i])
# For cell from 1+50j (j=1,18) from the west
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], West, 'numpy')
    list_w = [i for i in range(1, nx * ny - nx) if i % nx == 0]
    for i, j in zip(list_w, list_w):
        S_data.extend([f(1, 0, 0, 0, 0, U[i]), f(0, 0, 0, 1, 0, U[i]),f(0, 0, 1, 0, 0, U[i]), f(0, 0, 0, 0, 1,U[i])])
        S_row.extend([i] * 4)
        S_col.extend([j, j + 1, j + nx, j - nx])
        D[i] = -f(0, TWw, 0, 0, 0, U[i])
# For cell 50+50*j (j=1,18) from the east
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], East, 'numpy')
    list_e = np.array([i for i in range(2 * nx, nx * ny) if i % nx ==
0]) - 1
    for i, j in zip(list_e, list_e):
        S_data.extend([f(1, 0, 0, 0, 0, U[i]), f(0, 1, 0, 0, 0, U[i]),f(0, 0, 1, 0, 0, U[i]), f(0, 0, 0, 0, 1,U[i])])
        S_row.extend([i] * 4)
        S_col.extend([j, j - 1, j + nx, j - nx])
        D[i] = -f(0, 0, 0, 0, 0, U[i])
# For cell from 1+50*19 to 50*20 from the north
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], North, 'numpy')
    list_n = list(range(nx * (ny - 1) + 1, nx * ny - 1))
    for i, j in zip(list_n, list_n):
        S_data.extend([f(1, 0, 0, 0, 0, U[i]), f(0, 1, 0, 0, 0, U[i]),f(0, 0, 0, 1, 0, U[i]), f(0, 0, 0, 0, 1,U[i])])
        S_row.extend([i] * 4)
        S_col.extend([j, j - 1, j + 1, j - nx])
        D[i] = -f(0, 0, TWn, 0, 0, U[i])
# For the entire cells excluding the cells along the boundary
    list_cell = np.arange(nx * ny).reshape(ny, nx)[1:-1, 1:-1].reshape(-1, )
    f = sym.lambdify([T, Tw, Tn, Te, Ts, u], Center, 'numpy')
    for i, j in zip(list_cell, list_cell):
        S_data.extend([f(1, 0, 0, 0, 0, U[i]), f(0, 1, 0, 0, 0, U[i]),f(0, 0, 0, 1, 0, U[i]), f(0, 0, 1, 0, 0, U[i]),f(0, 0, 0, 0, 1, U[i])])
        S_row.extend([i] * 5)
        S_col.extend([j, j - 1, j + 1, j + nx, j - nx])
        D[i] = -f(0, 0, 0, 0, 0, U[i])
    data = np.array(S_data).reshape(-1, )
    row = np.array(S_row).reshape(-1, )
    col = np.array(S_col).reshape(-1, )
    S = coo_matrix((data, (row, col))).tocsc(True)
    z = solve(S, D)
    plot_contour(z)
    cell_contour(z)
# Plot temperature along the centerline versus x
    plt.plot(x[:-1]+dx/2,z[ny//2,:])
    plt.xlabel('x')
    plt.ylabel('Temperature distribution on centerline')
    plt.show()
# --------------------------- #
# ---------END MAIN---------- #
# --------------------------- #
if __name__ == "__main__":
    num_iter = 1
    rho = 1000
    Cp = 4181
    k = 0.6
    u_max = 0.1
    TWn = 300
    TWs = 200
    TWw = 450
    x_min, x_max, nx = (0, 100e-6, 100)
    y_min, y_max, ny = (-10e-6, 10e-6, 40)
    L = x_max - x_min
    d = y_max - y_min
    x, dx = np.linspace(x_min, x_max, nx + 1, retstep=True)
    y, dy = np.linspace(y_min, y_max, ny + 1, retstep=True)
    U = u_max * (1 - ((y[:-1] + y[1:]) / 2 / (d / 2)) ** 2)
    U = np.broadcast_to(U, (nx, ny)).reshape(-1, 1)
    D = np.empty((nx * ny, 1))
    T, Tw, Tn, Te, Ts, u = sym.symbols('T, Tw, Tn, Te, Ts, u')
    A = k / dx
    B = k / dy
    C = rho * Cp * u
    A1 = dx * 1
    A2 = dy * 1
    C1 = 0
    C2 = nx - 1
    C3 = nx * (ny - 1)
    C4 = nx * ny - 1
    main()
# --Copyright: Quach Gia Huy-- #