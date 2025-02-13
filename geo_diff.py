import sympy as sp
from sympy import Matrix, diff, MutableDenseNDimArray
import numpy as np

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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
        self.covariant_riemann = None
        self.ricci_tensor = None
        self.scalar_curvature = None
        self.einstein_tensor = None
        self.sectional_curvatures = None
        self.geodesics = None
        self.kretschmann_scalar = None

    def get_christoffel_symbols(self):
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

    def get_riemann_tensor(self):
        """
        Calcola le componenti del tensore di curvatura di Riemann, nella forma
        R^ρ_σμν=∂μΓ^ρ_σν−∂νΓ^ρ_σμ+∑_λ(Γ^ρ_μλ Γ^λ_σν−Γ^ρ_νλ Γ^λ_σμ)
        """
        self.get_christoffel_symbols()

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

    def is_flat(self):
        self.get_riemann_tensor()
        n = self.dimension
        return self.riemann_tensor == sp.MutableDenseNDimArray.zeros(n, n, n, n)

    def get_covariant_riemann(self):
        """
                Calcola il Riemann completamente covariante.
                R_ρσμν = g_λρ R^λ_σμν
                """
        self.get_riemann_tensor()

        cov_riemann = MutableDenseNDimArray.zeros(
            self.dimension, self.dimension, self.dimension, self.dimension)

        for rho in range(self.dimension):
            for sigma in range(self.dimension):
                for mu in range(self.dimension):
                    for nu in range(self.dimension):
                        cov_riemann[rho, sigma, mu, nu] = sum(
                            self.metric[rho, lamb] * self.riemann_tensor[lamb, sigma, mu, nu]
                            for lamb in range(self.dimension)
                        )
        self.covariant_riemann = sp.simplify(cov_riemann)
        return self.covariant_riemann

    def get_sectional_curvature_matrix(self):
        """
        Calcola la matrice S_ij = R_ijij/(g_ii g_jj-g_ij^2)
            delle curvature sezionali rispetto ai piani coordinati (i,j)
        Il caso della diagonale i=j non individua nessun piano ed è
            gestito dalla funzione ponendo S_ii = 0.
        """
        self.get_covariant_riemann()
        R = self.covariant_riemann
        g = self.metric
        n = self.dimension
        S = MutableDenseNDimArray.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    S[i, j] += R[i, j, i, j] / (g[i, i] * g[j, j] - g[i, j] ** 2)

        self.sectional_curvatures = sp.simplify(S)
        return self.sectional_curvatures

    def print_sectional_curvatures(self):
        """Prints the sectional curvatures of the Manifold. 
            It needs that they have been computed before 
             by self.get_sectional_curvatures_matrix()"""
        #self.get_sectional_curvatures_matrix()
        for i, coord1 in enumerate(self.coords):
            for j, coord2 in enumerate(self.coords):
                if i < j:
                    print(f'Sectional curvature in the plane (∂{coord1},∂{coord2}): K_{coord1}{coord2} =',
                          self.sectional_curvatures[i, j])

    def get_ricci_tensor(self):
        """
        Calcola il tensore di Ricci.
        R_μν = R^ρ_μρν
        """
        self.get_riemann_tensor()

        ricci_tensor = sp.MutableDenseNDimArray.zeros(self.dimension, self.dimension)
        for mu in range(self.dimension):
            for nu in range(self.dimension):
                ricci_tensor[mu, nu] = sum(self.riemann_tensor[rho, mu, rho, nu] for rho in range(self.dimension))

        self.ricci_tensor = sp.simplify(sp.Matrix(ricci_tensor))
        return self.ricci_tensor

    def get_ricci_tensor2(self):
        """
        Calcola il tensore di Ricci.
        R_μν = g^ab R_aμbν
        """
        # self.get_christoffel_symbols()
        self.get_covariant_riemann()
        g_inv = self.metric.inv()

        ricci_tensor = sp.MutableDenseNDimArray.zeros(self.dimension, self.dimension)
        for mu in range(self.dimension):
            for nu in range(self.dimension):
                ricci_tensor[mu, nu] = sum(
                    g_inv[a, b] * self.covariant_riemann[a, mu, b, nu]
                    for a in range(self.dimension) for b in range(self.dimension)
                )

        self.ricci_tensor = sp.simplify(sp.Matrix(ricci_tensor))
        return self.ricci_tensor

    def get_scalar_curvature(self):
        """
        Calcola la curvatura scalare.
        """
        self.get_ricci_tensor()

        metric_inv = self.metric.inv()
        scalar_curvature = sum(
            metric_inv[mu, nu] * self.ricci_tensor[mu, nu]
            for mu in range(self.dimension) for nu in range(self.dimension)
        )

        self.scalar_curvature = sp.simplify(scalar_curvature)
        return self.scalar_curvature

    def get_kretschmann_scalar(self):
        """
        Calcola l'invariante di Kretschmann R_ρσμν*R^ρσμν
        """
        self.get_riemann_tensor()
        g_inv = self.metric.inv()
        #self.get_covariant_riemann()
        self.get_sectional_curvature_matrix() #questo calcola anche il covariant Riemann

        # Costruisco il Riemann completamente controvariante
        contra_riemann = MutableDenseNDimArray.zeros(
            self.dimension, self.dimension, self.dimension, self.dimension)
        for rho in range(self.dimension):
            for sigma in range(self.dimension):
                for mu in range(self.dimension):
                    for nu in range(self.dimension):
                        contra_riemann[rho, sigma, mu, nu] = sum(
                            g_inv[a, sigma] * g_inv[b, mu] * g_inv[c, nu] * self.riemann_tensor[rho, a, b, c]
                            for a in range(self.dimension)
                            for b in range(self.dimension)
                            for c in range(self.dimension)
                        )
        contra_riemann = sp.simplify(contra_riemann)

        # Costruisco l'invariante di Kretschmann
        K = sum(self.covariant_riemann[rho, sigma, mu, nu] * contra_riemann[rho, sigma, mu, nu]
                for rho in range(self.dimension)
                for sigma in range(self.dimension)
                for mu in range(self.dimension)
                for nu in range(self.dimension)
                )
        self.kretschmann_scalar = sp.simplify(K)
        return self.kretschmann_scalar

    def create_spacetime_metric(self, scale_factor):
        """
        Costruisce una metrica lorentziana dallo spazio delle foglie spaziali.
        Assume che la metrica attuale rappresenti solo le foglie spaziali.
        :param scale_factor: è il fattore conforme di scala. Può essere una funzione sympy, una costante...
        """
        if self.metric.shape[0] + 1 != len(self.coords) + 1:
            raise ValueError("La metrica delle foglie deve essere n-dimensionale rispetto a n+1 coordinate totali.")

        lorentzian_metric = sp.eye(self.dimension + 1)
        lorentzian_metric[0, 0] = -1  # Aggiunge il termine dt^2
        lorentzian_metric[1:, 1:] = scale_factor ** 2 * self.metric

        return lorentzian_metric

    def get_einstein_tensor(self):
        """
        Calcola il tensore di Einstein.
        """
        self.get_scalar_curvature()
        self.einstein_tensor = sp.simplify(self.ricci_tensor - (1 / 2) * self.metric * self.scalar_curvature)
        return self.einstein_tensor

    def einstein_constant(self):
        """
        Calcola la costante di Einstein lambda:
            Ric=lambda*g.
        """
        self.get_scalar_curvature()
        return self.scalar_curvature / self.dimension

    def is_einstein_mfd(self):
        """
            Verifica se è una varietà di Einstein
        """
        self.get_scalar_curvature()
        l = self.einstein_constant()
        return self.ricci_tensor == l * self.metric

    def vacuum_einstein_eqs(self, Lambda):
        """
        Verifica True or False se una Manifold soddisfa le equazioni di Einstein.

        :param: Lambda è la costante cosmologica; può essere assegnata in sympy
        sia come Lambda = sympy.symbols('Lambda') che come vera e propria funzione
        sympy, o semplicemente come funzione costante.
        """
        self.get_einstein_tensor()
        return self.einstein_tensor + Lambda * self.metric == sp.zeros(self.dimension, self.dimension)

    def inner_product(self, X, Y):
        """
        Calcola il prodotto scalare g(X, Y) tra due vettori X e Y utilizzando la metrica g.

        :param X: Vettore X (lista o matrice 1D di dimensione n)
        :param Y: Vettore Y (lista o matrice 1D di dimensione n)
        :return: Il prodotto scalare g(X, Y)
        """
        # Assicuriamoci che X e Y siano nel formato giusto
        # if len(X) != self.dimension or len(Y) != self.dimension:
        #   raise ValueError("I vettori X e Y devono essere di dimensione n.")

        # Calcolare il prodotto scalare g(X, Y) = X^T * g * Y
        inner_product = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                inner_product += self.metric[i, j] * X[i] * Y[j]

        return inner_product

    def get_geodesic_equations(self):
        """
        Calcola simbolicamente le equazioni geodetiche per la varietà.

        Le equazioni hanno la forma:
            d^2 x^mu / dτ^2 + Γ^mu_{νρ} (dx^ν / dτ) (dx^ρ / dτ) = 0

        :return: Lista delle equazioni geodetiche, una per ogni coordinata, in forma sp.Equality.
        """
        self.get_christoffel_symbols()

        # Variabili per le derivate delle coordinate
        t = sp.symbols('τ')  # Parametro della curva
        coords = self.coords
        coord_funcs = [sp.Function(str(coord))(t) for coord in coords]

        # Velocità (prime derivate rispetto a t)
        velocities = [sp.diff(func, t) for func in coord_funcs]

        # Accelerazioni (seconde derivate rispetto a t)
        accelerations = [sp.diff(vel, t) for vel in velocities]

        # Equazioni geodetiche
        geodesic_equations = []
        for i in range(self.dimension):
            equation = accelerations[i]
            for j in range(self.dimension):
                for k in range(self.dimension):
                    equation += self.christoffel_symbols[i][j, k] * velocities[j] * velocities[k]
            geodesic_equations.append(sp.Eq(equation, 0))

        self.geodesics = geodesic_equations
        return self.geodesics

    def display_geodesic_equations(self):
        """Prints the geodesics equations of the Manifold. 
            It needs that they have been computed before 
             by self.get_geodesic_equations()"""
        #self.get_geodesic_equations()
        eqs_list = []
        for i, coord in enumerate(self.coords):
            eqs_list.append(self.geodesics[i])
            print(f"\nGeodesic equation along {coord}:")
            sp.pprint(eqs_list[i])  # Stampa leggibile in console
            print("\nLaTeX format:")
            print(f'{sp.printing.latex(eqs_list[i])}')  # Output LaTeX-friendly

    def geodesic_system(self, lambda_, Y):
        """
        Sistema differenziale del primo ordine per le equazioni geodetiche.

        - lambda_: parametro affine
        - Y: array contenente [x^i, v^i]
        - manifold: istanza di Manifold o Submanifold
        """
        dim = self.dimension
        coords = self.coords  # variabili x^i simboliche
        christoffels = self.get_christoffel_symbols()  # array di simboli di Christoffel

        # Separiamo le coordinate e le velocità
        x_vals = Y[:dim]
        v_vals = Y[dim:]

        # Convertiamo le espressioni simboliche in funzioni numeriche
        subs_dict = dict(zip(coords, x_vals))  # sostituzione delle coordinate attuali
        christoffels_numeric = np.array([
            [[float(sp.N(christoffels[i][j, k].subs(subs_dict))) for k in range(dim)]
             for j in range(dim)] for i in range(dim)
        ])

        # Calcoliamo dv^i/dlambda = -Γ^i_{jk} v^j v^k
        dv_vals = np.zeros(dim)
        for i in range(dim):
            dv_vals[i] = -sum(
                christoffels_numeric[i][j, k] * v_vals[j] * v_vals[k] for j in range(dim) for k in range(dim))

        return np.concatenate([v_vals, dv_vals])

    # Funzione per integrare le geodetiche
    def solve_geodesic(self, initial_position, initial_velocity, lambda_span, num_points=100):
        """
        Risolve numericamente le equazioni geodetiche.

        - manifold: istanza di Manifold o Submanifold
        - initial_position: condizioni iniziali sulle coordinate
        - initial_velocity: condizioni iniziali sulle derivate
        - lambda_span: tuple (lambda_iniziale, lambda_finale)
        - num_points: numero di punti per la soluzione
        """
        y0 = np.concatenate([initial_position, initial_velocity])

        sol = solve_ivp(
            fun=lambda l, Y: self.geodesic_system(l, Y),
            t_span=lambda_span,
            y0=y0,
            method='RK45',  # Runge-Kutta di ordine 4-5
            t_eval=np.linspace(lambda_span[0], lambda_span[1], num_points)
        )

        return sol


    

    def kulkarni_nomizu(self, A, B):
        """
        Compute Kulkarni-Nomizu product of two symmetric (0,2)-tensors as
            (A\owedge B)_ijkl = A_ik B_jl + A_jl B_ik - A_il B_jk - A_jk B_il
        """
        n = self.dimension
        AnomB = MutableDenseNDimArray.zeros(n, n, n, n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        AnomB[i, j, k, l] += A[i, k] * B[j, l] + A[j, l] * B[i, k] - A[i, l] * B[j, k] - A[j, k] * B[
                            i, l]

        return sp.simplify(AnomB)
        

    def covariant_derivative(self, X, T, ind):
        """
        Compute the covariant derivative nabla_X T, where T is a generic (h,k) tensor
            X = X^l ∂_l and e.g. T = T^k_ij ∂_k dx^i dx^j
            ∇_X T = X^l(∂_l(T^k_ij) - Γ^m_li T^k_mj - Γ^m_lj T^k_im + Γ^k_lm T^m_ij)dx^i dx^j ∂_k
                i.e. (∇_l T)^k_ij = ∂_l(T^k_ij) - Γ^m_li T^k_mj - Γ^m_lj T^k_im + Γ^k_lm T^m_ij

        :param X: vector field given the direction where the derivation is computed
        :param T: tensor field on which the derivation acts
        :param ind: tuple (h, k) for the type of T
        """
        self.get_christoffel_symbols()
        Gamma = self.christoffel_symbols
        n = self.dimension
        h, k = ind[0], ind[1]

        if h == 1 and k == 0:
            nabla_XT = sp.Matrix.zeros(n, 1)
            for k in range(n):
                for i in range(n):
                    nabla_XT[k] = sum(
                        X[i] * sp.diff(T[k], self.coords[i]) + Gamma[k][i, j] * T[j]
                        for j in range(n)
                    )

        elif h == 0 and k == 1:
            nabla_XT = sp.Matrix.zeros(n, 1)
            for j in range(n):
                for i in range(n):
                    nabla_XT[j] = sum(
                        X[i] * sp.diff(T[j], self.coords[i]) - Gamma[k][i, j] * T[k]
                        for k in range(n)
                    )

        # elif h == 2 and k == 0:
        # elif h == 0 and k == 2:
        elif h == 1 and k == 2:
            # ∇_X T = X^l(∂_l(T^k_ij) - Γ^m_li T^k_mj - Γ^m_lj T^k_im + Γ^k_lm T^m_ij)dx^i dx^j ∂_k
            nabla_XT = MutableDenseNDimArray.zeros(n, n, n)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            nabla_XT[k, i, j] = sum(
                                X[l] * (sp.diff((T), self.coords[l]) - Gamma[m][l, i] * T[k, m, j] - Gamma[m][l, j] * T[
                                    k, i, m] + Gamma[k][l, m] * T[m, i, j])
                                for m in range(n)
                            )

        # elif h == 2 and k == 1:
        else:
            raise f'This method has not been yet extended to this higher-order tensor field T = {sp.pprint(T)}'

        return sp.simplify(nabla_XT)
        

    def gradient(self, f):
        """
        Compute the gradient "X = grad_g(f)" of a scalar function f:(M,g) --> R
        being X = g^ij ∂_i(f) ∂_j

        :param f: sympy function sp.Function('f')('x^1 ... x^n'),
                     where x^j = sp.symbols('x^j')
                     and self.coords = [x^1 ... x^n].
        """
        g_inv = self.metric.inv()
        n = self.dimension
        X = sp.Matrix.zeros(n, 1)
        for j in range(n):
            X[j] = sum(
                g_inv[i, j] * sp.diff(f, self.coords[i])
                for i in range(n)
            )

        return sp.simplify(X)

    def divergence(self, X):
        """
        Compute the divergence "div_g(X)" of a vector field X:(M,g) --> TM
        """
        n = self.dimension
        delta = sp.Matrix.eye(n)
        Gamma = self.get_christoffel_symbols()

        divX = sum(
            delta[mu, nu] * (sp.diff(X[nu], self.coords[mu]) + Gamma[nu][mu, lamb] * X[lamb])
            for mu in range(n) for nu in range(n) for lamb in range(n)
        )
        return sp.simplify(divX)

    def divergence2(self, X):
        n = self.dimension
        g = sp.det(self.metric)
        div = sum(
            1 / sp.sqrt(g) * sp.diff(sp.sqrt(g) * X[i], self.coords[i])
            for i in range(n)
        )
        return sp.simplify(div)

    def laplacian(self, f):
        """Compute the laplacian of f as the divergence of the gradient
            i.e. Δ_g(f) = div_g(grad_g(f)) """
        return self.divergence(self.gradient(f))

    def hessian(self, f):
        """Compute the Hessian matrix of a function f as ∇df """
        n = self.dimension
        Gamma = self.get_christoffel_symbols()
        Hess = MutableDenseNDimArray.zeros(n, n)
        for i in range(n):
            for j in range(n):
                Hess[i, j] = sum(
                    sp.diff(f, self.coords[i], self.coords[j]) - Gamma[k][i, j] * sp.diff(f, self.coords[k])
                    for k in range(n)
                )
        return sp.simplify(Hess)

    def laplacian2(self, f):
        n = self.dimension
        Hess = self.hessian(f)
        g_inv = self.metric.inv()
        lapl = sum(
            g_inv[i, j] * Hess[i, j]
            for i in range(n) for j in range(n)
        )
        return sp.simplify(lapl)



    def get_geometrics(self):
        """Compute the main geometric objects of a (sub)manifold"""
        self.get_einstein_tensor() #computed: christoffels, riemann, ricci, scalar, einstein
        self.get_kretschmann_scalar() #computed: covariant riemann, sectional curvatures, kretschmann
        self.get_geodesic_equations()



class Submanifold(Manifold):
    def __init__(self, ambient_manifold, sub_coords, embedding):
        """
        Inizializza una sottovarietà.
        :param ambient_manifold: Istanza della classe Manifold della varietà originale.
        :param sub_coords: Coordinate simboliche della sottovarietà.
        :param embedding: Funzione di immersione che esprime le coordinate globali in termini di quelle della sottovarietà.
        """
        self.ambient_manifold = ambient_manifold
        self.sub_coords = sub_coords
        self.embedding = embedding  # Lista di espressioni simboliche: x_i = f(sub_coords)
        self.dimension = len(sub_coords)

        self.embedding_jacobian = None
        self.induced_metric = None
        self.normal_field = None
        self.second_fundamental_form = None
        self.mean_curvature = None

        self.metric = None  # questo serve per gestire bene l'ereditarietà di certi metodi di Manifold
        self.coords = self.sub_coords  # come ad esempio self.get_christoffels_symbols()

    def get_embedding_jacobian(self):
        self.embedding_jacobian = Matrix([
            [diff(f, coord) for coord in self.sub_coords]
            for f in self.embedding
        ])

        return self.embedding_jacobian

    def get_induced_metric(self):
        """
        Calcola la metrica indotta sulla sottovarietà.
        :return: Matrice simbolica della metrica indotta.
        """
        g = self.ambient_manifold.metric  # Metrica della varietà ambiente
        self.get_embedding_jacobian()  # Jacobiano dell'immersione

        # Metrica indotta: G_ab = (Jacobian)^T * g * (Jacobian)
        self.induced_metric = sp.simplify(self.embedding_jacobian.T * g * self.embedding_jacobian)
        self.metric = self.induced_metric  # serve per poterci agire con metodi di Manifold, e.g. get_christoffel_symbols()
        # self.coords = self.sub_coords

        return self.induced_metric

    def get_normal_field(self):
        """
        Calcola il campo normale della submanifold nell'ambiente.
        :return: Lista di vettori normali simbolici.
        """
        jacobian = self.get_embedding_jacobian()
        ambient_metric = self.ambient_manifold.metric

        tangent_vectors = [jacobian[:, i] for i in range(self.dimension)]
        d = len(self.embedding)

        normal_vectors = [sp.symbols(f'n{i + 1}') for i in range(d)]  # Lista di simboli normali
        # initial_symbols = [sp.symbols(f'n{i+1}') for i in range(d)] #ci serve per dopo nel caso in cui n-k>1

        equations = []
        # Condizioni di ortogonalità rispetto alla metrica
        for tangent in tangent_vectors:
            eq = sum(ambient_metric[i, j] * tangent[i] * normal_vectors[j]
                     for i in range(d) for j in range(d))
            equations.append(eq)

        # Normalizzazione: g(N, N) = 1
        norm_eq = sum(ambient_metric[i, j] * normal_vectors[i] * normal_vectors[j]
                      for i in range(d) for j in range(d)) - 1
        equations.append(norm_eq)

        # Risolve il sistema e seleziona il verso giusto
        solutions = sp.solve(equations, normal_vectors)
        # self.compute_scalar_curvature()
        # if self.scalar_curvature >= 0:
        #     self.normal_field = sp.Matrix([solutions[0]])
        # else:
        #      self.normal_field = sp.Matrix([solutions[1]])

        # gestione del caso n-k>1: mancante

        self.normal_field = sp.Matrix([solutions[0]])

        self.normal_field = self.normal_field.subs(sp.I, 1)  # normalizza a reali eventuali vettori complessi
        # questo punto è poco chiaro, non dovrebbe succedere
        return sp.simplify(self.normal_field)

    def get_IInd_fundamental_form(self):
        """
        Calcola la seconda forma fondamentale per la sottovarietà in un ambiente con connessione in generale non piatta.
        :return: Matrice simbolica della seconda forma fondamentale.
        """
        self.get_embedding_jacobian()
        self.ambient_manifold.get_christoffel_symbols()
        Gamma = self.ambient_manifold.christoffel_symbols
        self.get_normal_field()

        II = sp.zeros(self.dimension, self.dimension)
        tangent_vectors = [self.embedding_jacobian[:, i] for i in range(self.dimension)]
        coords = self.sub_coords

        num_vectors = len(tangent_vectors)  # è la dimensione dell'immagine dell'embedding
        num_coords = len(coords)

        # Matrice di derivate, dove ogni elemento è un vettore (colonna)
        derivative_matrix = [[None for _ in range(num_vectors)] for _ in range(num_coords)]

        # Calcolo delle derivate dirette dei vettori tangenti
        for j, tangent_vector in enumerate(tangent_vectors):  # Itera sui vettori tangenti
            for i, coord in enumerate(coords):  # Itera sulle coordinate
                # Calcola la derivata del j-esimo vettore tangente rispetto alla i-esima coordinata
                derivative_matrix[i][j] = tangent_vector.diff(coord)

        # Correzione della connessione
        for i in range(self.dimension):  # indice di derivazione
            for j in range(self.dimension):  # Indice del vettore tangente da derivare
                # Inizializziamo la derivata covariante
                nabla_XY = derivative_matrix[i][j]
                # Inizializzo la derivata covariante come la derivata diretta precedentemente costruita

                # Aggiungo la correzione dei Christoffel
                christoffel_correction = sp.zeros(len(tangent_vectors[0]), 1)  # Vettore colonna
                for k in range(len(tangent_vectors[0])):  # Componente del vettore
                    for m in range(len(tangent_vectors[0])):  # Somma sui vettori tangenti
                        christoffel_correction[k] += Gamma[k][i, m] * tangent_vectors[j][m]

                # Aggiorna la derivata covariante con la correzione
                nabla_XY += christoffel_correction

                # Calcola la proiezione su normal_field per la seconda forma fondamentale
                # II[i, j] = self.normal_field.dot(nabla_XY)
                II[i, j] = self.ambient_manifold.inner_product(nabla_XY, self.normal_field)

        # Salva e restituisce la seconda forma fondamentale
        self.second_fundamental_form = sp.simplify(II)
        return self.second_fundamental_form

    def get_mean_curvatureII(self):
        """
        Calcola la curvatura media della varietà immersa.
        :param: Normal vector field in forma di vettore sympy
        :return: Scalare in forma di sympy function o costante
                Traccia della matrice II
        """
        self.get_IInd_fundamental_form()
        self.get_induced_metric()

        I = self.induced_metric.inv()
        II = self.second_fundamental_form
        # H = 0
        # for a in range(self.dimension):
        #    H += I[a, a] * II[a, a]
        H = sum(
            I[a, a] * II[a, a] for a in range(self.dimension)
        )
        self.mean_curvature = sp.simplify(H)
        return self.mean_curvature

    def is_minimal(self):
        return self.mean_curvature == 0

    def is_totally_geodesic(self):
        n = self.dimension
        self.get_IInd_fundamental_form()
        return self.second_fundamental_form == sp.Matrix.zeros(n, n)

    # di seguito dei doppioni con inserimento manuale del normal vector field
    def get_second_fundamental_form(self, normal_field):
        """
        Calcola la seconda forma fondamentale per la sottovarietà in un ambiente con connessione in generale non piatta.
        :param: normal_field: Campo normale in forma di vettore SymPy.
        :return: Matrice simbolica della seconda forma fondamentale.
        """
        self.get_embedding_jacobian()
        self.ambient_manifold.get_christoffel_symbols()
        Gamma = self.ambient_manifold.christoffel_symbols

        II = sp.zeros(self.dimension, self.dimension)
        tangent_vectors = [self.embedding_jacobian[:, i] for i in range(self.dimension)]
        coords = self.sub_coords

        num_vectors = len(tangent_vectors)  # è la dimensione dell'immagine dell'embedding
        num_coords = len(coords)

        # Matrice di derivate, dove ogni elemento è un vettore (colonna)
        derivative_matrix = [[None for _ in range(num_vectors)] for _ in range(num_coords)]

        # Calcolo delle derivate dirette dei vettori tangenti
        for j, tangent_vector in enumerate(tangent_vectors):  # Itera sui vettori tangenti
            for i, coord in enumerate(coords):  # Itera sulle coordinate
                # Calcola la derivata del j-esimo vettore tangente rispetto alla i-esima coordinata
                derivative_matrix[i][j] = tangent_vector.diff(coord)

        # Correzione della connessione
        for i in range(self.dimension):  # indice di derivazione
            for j in range(self.dimension):  # Indice del vettore tangente da derivare
                # Inizializziamo la derivata covariante
                nabla_XY = derivative_matrix[i][j]
                # Inizializzo la derivata covariante come la derivata diretta precedentemente costruita

                # Aggiungo la correzione dei Christoffel
                christoffel_correction = sp.zeros(len(tangent_vectors[0]), 1)  # Vettore colonna
                for k in range(len(tangent_vectors[0])):  # Componente del vettore
                    for m in range(len(tangent_vectors[0])):  # Somma sui vettori tangenti
                        christoffel_correction[k] += Gamma[k][i, m] * tangent_vectors[j][m]

                # Aggiorna la derivata covariante con la correzione
                nabla_XY += christoffel_correction

                # Calcola la proiezione su normal_field per la seconda forma fondamentale
                II[i, j] = normal_field.dot(nabla_XY)

        # Salva e restituisci la seconda forma fondamentale
        self.second_fundamental_form = sp.simplify(II)
        return self.second_fundamental_form

    def get_mean_curvature(self, normal_field):
        """
        Calcola la curvatura media della varietà immersa.
        :param: Normal vector field in forma di vettore sympy
        :return: Scalare in forma di sympy function o costante
                Traccia della matrice II
        """
        self.get_second_fundamental_form(normal_field)
        self.get_induced_metric()

        I = self.induced_metric.inv()
        II = self.second_fundamental_form
        H = 0
        for a in range(self.dimension):
            for b in range(self.dimension):
                if a == b:
                    H += I[a, b] * II[a, b]

        self.mean_curvature = sp.simplify(H)
        return self.mean_curvature


    
    def plot_surface(self, domain, fig_title='Surface'):
        """:param domain: it's a list made of 2 tuples giving the intervals of the variables parametrizing the surface"""

        coords = self.sub_coords
        x, y, z = self.embedding[0], self.embedding[1], self.embedding[2]

        flg_null = None
        for i, c in enumerate([x, y, z]):
            if c == 0:
                flg_null = i

        func = [sp.lambdify((coords[0], coords[1]), coord, 'numpy') for coord in [x, y, z]]
        # questo mi produce e.g. [x(u,v), y(u,v), z(u,v)]

        # Creiamo la meshgrid per le coordinate
        a1, b1, a2, b2 = domain[0][0], domain[0][1], domain[1][0], domain[1][1]
        u = np.linspace(a1, b1, 100)
        v = np.linspace(a2, b2, 100)
        U, V = np.meshgrid(u, v)

        # Valutiamo le coordinate cartesiane
        Func = [c(U, V) for c in func]

        if flg_null != None: #gestisce i casi con una coordinata nulla
            if flg_null == 0:
                Func[flg_null] = np.zeros_like(Func[flg_null+1])
            else: #elif flg_null == 1 or flg_null == 2:
                Func[flg_null] = np.zeros_like(Func[flg_null-1])

        # Plot
        fig = plt.figure(figsize=(6, 6))
        plt.title(fig_title)
        fig.canvas.manager.set_window_title('Surface plot')

        ax = fig.add_subplot(111, projection='3d')

        if flg_null != None:
            ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='cividis', edgecolor='none', alpha=0.9, shade=True)
        else:
            ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='Blues', edgecolor='none', alpha=0.9, shade=True)

        # Etichette degli assi
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.set_title('abcsda', fontsize=14)

        # Rimuoviamo i gridlines per un aspetto più pulito
        #ax.grid(False)
        #plt.axis('off')

        return plt.show()

    def plot_geodesics_on_surface(self, domain, geodesics, fig_title='Geodesics'):
        """
        Plotta la superficie immersa in R3 e le geodetiche sopra di essa.

        :param domain: Lista con due tuple che danno gli intervalli delle coordinate parametrizzanti la submanifold.
        :param geodesics: Lista di soluzioni numeriche delle equazioni geodetiche.
        """

        coords = self.sub_coords
        x, y, z = self.embedding[0], self.embedding[1], self.embedding[2]

        flg_null = None
        for i, c in enumerate([x, y, z]):
            if c == 0:
                flg_null = i

        # Funzioni di embedding lambda per valutare le coordinate 3D
        func = [sp.lambdify((coords[0], coords[1]), coord, 'numpy') for coord in [x, y, z]]

        # Creiamo la meshgrid per la superficie
        a1, b1, a2, b2 = domain[0][0], domain[0][1], domain[1][0], domain[1][1]
        u = np.linspace(a1, b1, 100)
        v = np.linspace(a2, b2, 100)
        U, V = np.meshgrid(u, v)

        # Valutiamo l'immersione nello spazio 3D
        Func = [c(U, V) for c in func]

        if flg_null != None: #gestisce i casi con una coordinata nulla
            if flg_null == 0:
                Func[flg_null] = np.zeros_like(Func[flg_null+1])
            else: #elif flg_null == 1 or flg_null == 2:
                Func[flg_null] = np.zeros_like(Func[flg_null-1])
                

        # Creazione della figura
        fig = plt.figure(figsize=(6, 6))
        plt.title(fig_title)
        fig.canvas.manager.set_window_title('Geodesic plot')

        ax = fig.add_subplot(111, projection='3d')

        # Plot della superficie
        
        if flg_null != None:
            ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='cividis', edgecolor='none', alpha=0.75, shade=True)
        else:
            ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='Blues', edgecolor='none', alpha=0.75, shade=True)
        
        # Plot delle geodetiche
        for sol in geodesics:
            u_vals, v_vals = sol.y[0], sol.y[1]  # Coordinate sulla submanifold

            # Mappiamo le coordinate della geodetica nell'immersione 3D
            x_vals = func[0](u_vals, v_vals)
            y_vals = func[1](u_vals, v_vals)
            z_vals = func[2](u_vals, v_vals)

            ax.plot(x_vals, y_vals, z_vals, color='r', linewidth=2)

        # Etichette degli assi
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        return plt.show()



    def get_geometrics_sub(self):
        """Compute the main geometric objects of a (sub)manifold"""
        self.get_induced_metric()
        self.get_einstein_tensor()
        self.get_kretschmann_scalar()
        self.get_sectional_curvature_matrix()
        self.get_geodesic_equations()
        self.get_mean_curvatureII()
