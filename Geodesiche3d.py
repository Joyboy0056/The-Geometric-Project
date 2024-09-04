import sympy as sp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def christoffel_symbols(metric, coords):
    """
    Calcola i simboli di Christoffel per una metrica data in dimensione n.

    :param metric: Matrice della metrica (Matrix di SymPy).
    :param coords: Coordinate simboliche (lista di Symbol di SymPy).
    :return: Lista di matrici SymPy dei simboli di Christoffel.
    """
    n = metric.shape[0]
    g_inv = metric.inv()

    christoffel_matrices = []
    for k in range(n):
        christoffel_matrix = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                term_sum = sum(
                    g_inv[k, sigma] * (
                            sp.diff(metric[i, sigma], coords[j]) +
                            sp.diff(metric[sigma, j], coords[i]) -
                            sp.diff(metric[i, j], coords[sigma])
                    ) / 2
                    for sigma in range(n)
                )
                christoffel_matrix[i, j] = term_sum
        christoffel_matrices.append(christoffel_matrix)

    return christoffel_matrices


def geodesic_eqns(Y, tau, christoffel_numeric):
    n = len(Y) // 2
    derivatives = [0] * len(Y)

    for k in range(n):
        d2x_dtau2 = -sum(
            christoffel_numeric[k][i][j](*Y[n:]) * Y[i] * Y[j]
            for i in range(n) for j in range(n)
        )
        derivatives[k] = d2x_dtau2

    derivatives[n:] = Y[:n]
    return derivatives


def solve_geodesics(metric, coords, Y0, tau_range):
    # Calcola i simboli di Christoffel
    christoffel_matrices = christoffel_symbols(metric, coords)

    # Converti le espressioni simboliche in funzioni numeriche
    christoffel_numeric = [[[sp.lambdify(coords, christoffel_matrices[k][i, j], "numpy")
                             for j in range(len(coords))] for i in range(len(coords))]
                           for k in range(len(coords))]

    # Risoluzione numerica delle equazioni delle geodetiche
    sol = odeint(geodesic_eqns, Y0, tau_range, args=(christoffel_numeric,))
    return sol


def plot_geodesic_3d(sol, coords):
    if len(coords) == 3:
        x, y, z = sol[:, 2], sol[:, 3], sol[:, 4]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='Geodesic')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()
    elif len(coords) == 2:
        theta_sol, phi_sol = sol[:, 2], sol[:, 3]
        x = np.sin(theta_sol) * np.cos(phi_sol)
        y = np.sin(theta_sol) * np.sin(phi_sol)
        z = np.cos(theta_sol)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='Geodesic on Surface')
        plt.legend()
        plt.show()
    else:
        print("Plotting is only supported for 2D or 3D spaces.")


# Esempio di utilizzo: spazio 3D in coordinate sferiche
def main():
    # Definire le coordinate sferiche
    r, theta, phi = sp.symbols('r theta phi')
    coords = [r, theta, phi]

    # Metrica round per coordinate sferiche
    g = sp.Matrix([
        [1, 0, 0],
        [0, r ** 2, 0],
        [0, 0, r ** 2 * sp.sin(theta) ** 2]
    ])

    # Condizioni iniziali [dr/dtau, dtheta/dtau, dphi/dtau, r, theta, phi]
    Y0 = [0, 0.1, 0.1, 1, np.pi / 4, 0]

    # Parametro affine tau
    tau_range = np.linspace(0, 10, 1000)

    # Risolvi le geodetiche
    sol = solve_geodesics(g, coords, Y0, tau_range)

    # Plot della geodetica
    plot_geodesic_3d(sol, coords)


if __name__ == "__main__":
    main()
