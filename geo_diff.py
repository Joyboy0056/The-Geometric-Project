import sympy as sp
from sympy import symbols, diff, MutableDenseNDimArray

def print_pretty_matrix(matrix, name="Matrix"):
    """
    Stampa una matrice SymPy in modo leggibile.

    Args:
        matrix: matrice SymPy da stampare.
        name: nome della matrice da visualizzare.
    """
    print(f"{name}:")
    sp.pprint(matrix, use_unicode=True)

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
        self.einstein_tensor = None
        
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
        return self.christoffel_symbols

    def compute_riemann_tensor(self):
        """
        Calcola le componenti del tensore di curvatura di Riemann, nella forma
        R^ρ_σμν=∂μΓ^ρ_σν−∂νΓ^ρ_σμ+∑_λ(Γ^ρ_μλ Γ^λ_σν−Γ^ρ_νλ Γ^λ_σμ)
        """
        self.compute_christoffel_symbols()

        Riemann_tensor = MutableDenseNDimArray.zeros(
            self.dimension, self.dimension, self.dimension, self.dimension)

        for rho in range(self.dimension):
            for sigma in range(self.dimension):
                for mu in range(self.dimension):
                    for nu in range(self.dimension):
                        # Derivata parziale dei simboli di Christoffel
                        partial_mu = diff(self.christoffel_symbols[rho][sigma, nu], self.coords[mu])
                        partial_nu = diff(self.christoffel_symbols[rho][sigma, mu], self.coords[nu])

                        # Somma dei termini gamma
                        gamma_mu = sum(
                            self.christoffel_symbols[rho][mu, lamb] * self.christoffel_symbols[lamb][sigma, nu]
                            for lamb in range(self.dimension)
                        )
                        gamma_nu = sum(
                            self.christoffel_symbols[rho][nu, lamb] * self.christoffel_symbols[lamb][sigma, mu]
                            for lamb in range(self.dimension)
                        )

                        # Combinazione finale del tensore di Riemann
                        Riemann_tensor[rho, sigma, mu, nu] = partial_mu - partial_nu + gamma_mu - gamma_nu

        self.riemann_tensor = sp.simplify(Riemann_tensor)
        return self.riemann_tensor

    def compute_ricci_tensor(self):
        """
        Calcola il tensore di Ricci.
        """
        self.compute_christoffel_symbols()
        self.compute_riemann_tensor()

        ricci_tensor = sp.MutableDenseNDimArray.zeros(self.dimension, self.dimension)
        for mu in range(self.dimension):
            for nu in range(self.dimension):
                ricci_tensor[mu, nu] = sum(self.riemann_tensor[rho, mu, rho, nu] for rho in range(self.dimension))

        self.ricci_tensor = sp.simplify(sp.Matrix(ricci_tensor))
        return self.ricci_tensor

    def compute_scalar_curvature(self):
        """
        Calcola la curvatura scalare.
        """
        self.compute_christoffel_symbols()
        self.compute_riemann_tensor()
        self.compute_ricci_tensor()

        metric_inv = self.metric.inv()
        scalar_curvature = sum(
            metric_inv[mu, nu] * self.ricci_tensor[mu, nu]
            for mu in range(self.dimension) for nu in range(self.dimension)
        )

        self.scalar_curvature = sp.simplify(scalar_curvature)
        return self.scalar_curvature


    def create_spacetime_metric(self, scale_factor):
        """
        Costruisce una metrica lorentziana dallo spazio delle foglie spaziali.
        Assume che la metrica attuale rappresenti solo le foglie spaziali.
        """
        if self.metric.shape[0] + 1 != len(self.coords) + 1:
            raise ValueError("La metrica delle foglie deve essere n-dimensionale rispetto a n+1 coordinate totali.")

        lorentzian_metric = sp.eye(self.dimension + 1)
        lorentzian_metric[0, 0] = -1  # Aggiunge il termine dt^2
        lorentzian_metric[1:, 1:] = scale_factor**2*self.metric

        return lorentzian_metric


    def compute_einstein_tensor(self):
        """
        Calcola il tensore di Einstein.
        """
        self.compute_christoffel_symbols()
        self.compute_riemann_tensor()
        self.compute_ricci_tensor()
        self.compute_scalar_curvature()

        self.einstein_tensor = self.ricci_tensor - (1 / 2) * self.metric * self.scalar_curvature
        return self.einstein_tensor



    def einstein_constant(self):
        """
        Calcola la costante di Einstein lambda:
            Ric=lambda*g.
        """
        self.compute_christoffel_symbols()
        self.compute_riemann_tensor()
        self.compute_ricci_tensor()
        self.compute_scalar_curvature()

        return self.scalar_curvature/self.dimension 


    def is_einstein_mfd(self):
        """
            Verifica se è una varietà di Einstein
        """
        self.compute_christoffel_symbols()
        self.compute_riemann_tensor()
        self.compute_ricci_tensor()
        self.compute_scalar_curvature()
        l = self.einstein_constant()
        return self.ricci_tensor == l*self.metric

    
    def vacuum_einstein_eqs(self, Lambda):
        """
        Verifica True or False se una Manifold soddisfa le equazioni di Einstein.
        
        :param: Lambda è la costante cosmologica; può essere assegnata in sympy
        sia come Lambda = sympy.symbols('Lambda') che come vera e propria funzione
        sympy, o semplicemente come funzione costante.
        """
        self.compute_christoffel_symbols()
        self.compute_riemann_tensor()
        self.compute_ricci_tensor()
        self.compute_scalar_curvature()
        #self.compute_einstein_tensor()
        
        self.einstein_tensor = sp.simplify(self.compute_einstein_tensor())

        return self.einstein_tensor + Lambda*self.metric == sp.zeros(self.dimension, self.dimension)

    def vacuum_einstein_eqs_without_cosm_const(self):
        self.einstein_tensor = sp.simplify(self.compute_einstein_tensor())
        return self.einstein_tensor == sp.zeros(self.dimension, self.dimension)

