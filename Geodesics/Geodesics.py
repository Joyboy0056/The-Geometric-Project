import sympy as sp
from Christoffels import christoffel_symbols, pretty_print_christoffel_matrices
from scipy.integrate import odeint

def print_geodesic_equations(christoffel_matrices, coords):
    """
    Stampa le equazioni geodetiche in una forma leggibile.

    :param christoffel_matrices: Lista di matrici dei simboli di Christoffel.
    :param coords: Coordinate simboliche (lista di Symbol di SymPy).
    """
    n = len(coords)
    geodesic_eqns = []

    # Definire le derivate rispetto al parametro affine (tipicamente tau)
    tau = sp.symbols('tau')
    x = [sp.Function(f'x^{i+1}')(tau) for i in range(n)]
    dxdt = [sp.diff(x[i], tau) for i in range(n)]
    d2xdt2 = [sp.diff(dxdt[i], tau) for i in range(n)]

    # Costruire le equazioni geodetiche
    for i in range(n):
        equation = d2xdt2[i]
        for j in range(n):
            for k in range(n):
                equation += -christoffel_matrices[i][j, k] * dxdt[j] * dxdt[k]
        geodesic_eqns.append(sp.Eq(equation, 0))

    # Stampa le equazioni geodetiche
    for i, eqn in enumerate(geodesic_eqns):
        print(f"Equazione geodetica per {x[i]}:")
        sp.pretty_print(eqn)
        print("\n")


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


# Esempio: Disco di Poincaré
r, theta = sp.symbols('r theta')
g = sp.Matrix([[1/(1+r**2), 0], [0, r**2]])
coords = [r, theta]

# Calcolo dei simboli di Christoffel con la funzione già definita
christoffel_matrices = christoffel_symbols(g, coords)

# Stampa delle equazioni geodetiche
print_geodesic_equations(christoffel_matrices, coords)
