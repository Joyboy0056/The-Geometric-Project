import numpy as np
import sympy as sp
from Christoffels import christoffel_symbols
from scipy.integrate import odeint
from Geodesics import print_geodesic_equations

# Definire le coordinate sferiche
theta, phi = sp.symbols('theta phi')
coords = [theta, phi]

# Metrica round della sfera S^2
g = sp.Matrix([
    [1, 0],
    [0, sp.sin(theta)**2]
])
christoffel_matrices = christoffel_symbols(g, coords)

print_geodesic_equations(christoffel_matrices, coords)

# Funzione per trasformare SymPy in NumPy
christoffel_numeric = [[[sp.lambdify(coords, christoffel_matrices[k][i, j], "numpy")
                         for j in range(2)] for i in range(2)] for k in range(2)]


def geodesic_eqns(Y, tau, christoffel_numeric):
    dtheta, dphi, theta, phi = Y

    d2theta_dtau2 = -sum(
        christoffel_numeric[0][i][j](theta, phi) * Y[i] * Y[j] for i in range(2) for j in range(2)
    )
    d2phi_dtau2 = -sum(
        christoffel_numeric[1][i][j](theta, phi) * Y[i] * Y[j] for i in range(2) for j in range(2)
    )

    return [d2theta_dtau2, d2phi_dtau2, dtheta, dphi]


# Condizioni iniziali: [dtheta/dtau, dphi/dtau, theta, phi]
Y0 = [0.0, 1.0, np.pi / 4, 0.0]

# Parametro affine tau
tau = np.linspace(0, 10, 1000)

# Risoluzione numerica delle equazioni
sol = odeint(geodesic_eqns, Y0, tau, args=(christoffel_numeric,))

import matplotlib.pyplot as plt

# Estrai le soluzioni per theta e phi
theta_sol = sol[:, 2]
phi_sol = sol[:, 3]

# Converte in coordinate cartesiane per il plot 3D
x = np.sin(theta_sol) * np.cos(phi_sol)
y = np.sin(theta_sol) * np.sin(phi_sol)
z = np.cos(theta_sol)

# Plot della geodetica sulla sfera
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Geodesic on S^2')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Aggiungi la superficie della sfera
u = np.linspace(0, np.pi, 100)
v = np.linspace(0, 2 * np.pi, 100)
x_sphere = np.outer(np.sin(u), np.cos(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.cos(u), np.ones_like(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.3)

plt.legend()
plt.show()
