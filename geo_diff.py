import sympy as sp
from sympy import symbols, diff, Matrix, MutableDenseNDimArray

class Manifold:
    def __init__(self, metric, coordinates):
        """
        Inizializza una varietà con la sua metrica e le coordinate.

        :param metric: Matrice della metrica (Matrix di SymPy).
        :param coordinates: Coordinate simboliche (lista di Symbol di SymPy).
        """
        self.metric = metric
        self.coords = coordinates
        self.dimension = len(coordinates)
        self.christoffel_symbols = None
        self.riemann_tensor = None
        self.ricci_tensor = None
        self.scalar_curvature = None

    def compute_christoffel_symbols(self):
        """
        Calcola i simboli di Christoffel.
        """
        g_inv = self.metric.inv()
        christoffel_matrices = []

        for k in range(self.dimension):
            christoffel_matrix = sp.zeros(self.dimension, self.dimension)
            for i in range(self.dimension):
                for j in range(self.dimension):
                    term_sum = sum(
                        g_inv[k, sigma] * (
                            sp.diff(self.metric[i, sigma], self.coords[j]) +
                            sp.diff(self.metric[sigma, j], self.coords[i]) -
                            sp.diff(self.metric[i, j], self.coords[sigma])
                        ) / 2
                        for sigma in range(self.dimension)
                    )
                    christoffel_matrix[i, j] = term_sum
            christoffel_matrices.append(christoffel_matrix)

        self.christoffel_symbols = christoffel_matrices

    def compute_riemann_tensor(self):
        """
        Calcola il tensore di curvatura di Riemann.
        """
        if self.christoffel_symbols is None:
            self.compute_christoffel_symbols()

        Riemann_tensor = MutableDenseNDimArray.zeros(
            self.dimension, self.dimension, self.dimension, self.dimension)

        for rho in range(self.dimension):
            for sigma in range(self.dimension):
                for mu in range(self.dimension):
                    for nu in range(self.dimension):
                        partial_mu = diff(self.christoffel_symbols[rho][nu, sigma], self.coords[mu])
                        partial_nu = diff(self.christoffel_symbols[rho][mu, sigma], self.coords[nu])

                        gamma_mu = sum(
                            self.christoffel_symbols[rho][mu, lamb] * self.christoffel_symbols[lamb][nu, sigma]
                            for lamb in range(self.dimension)
                        )
                        gamma_nu = sum(
                            self.christoffel_symbols[rho][nu, lamb] * self.christoffel_symbols[lamb][mu, sigma]
                            for lamb in range(self.dimension)
                        )

                        Riemann_tensor[rho, sigma, mu, nu] = partial_mu - partial_nu + gamma_mu - gamma_nu

        self.riemann_tensor = sp.simplify(Riemann_tensor)

    def compute_ricci_tensor(self):
        """
        Calcola il tensore di Ricci.
        """
        if self.riemann_tensor is None:
            self.compute_riemann_tensor()

        ricci_tensor = sp.MutableDenseNDimArray.zeros(self.dimension, self.dimension)
        for mu in range(self.dimension):
            for nu in range(self.dimension):
                ricci_tensor[mu, nu] = sum(self.riemann_tensor[rho, mu, rho, nu] for rho in range(self.dimension))

        self.ricci_tensor = sp.simplify(sp.Matrix(ricci_tensor))

    def compute_scalar_curvature(self):
        """
        Calcola la curvatura scalare.
        """
        if self.ricci_tensor is None:
            self.compute_ricci_tensor()

        metric_inv = self.metric.inv()
        scalar_curvature = sum(
            metric_inv[mu, nu] * self.ricci_tensor[mu, nu]
            for mu in range(self.dimension) for nu in range(self.dimension)
        )

        self.scalar_curvature = sp.simplify(scalar_curvature)

    def pretty_print_matrix(self, matrix, name):
        """
        Stampa una matrice SymPy in modo leggibile.
        """
        print(f"{name}:")
        sp.pprint(matrix, use_unicode=True)
