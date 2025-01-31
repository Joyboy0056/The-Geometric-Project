import sympy as sp
from sympy import Matrix, diff, MutableDenseNDimArray

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
        self.sectional_curvature = None
        self.geodesics = None
        self.kretschmann_scalar = None

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


    def is_flat(self):
        self.compute_riemann_tensor()
        n = self.dimension
        return self.riemann_tensor == sp.MutableDenseNDimArray.zeros(n, n, n, n)

    def compute_covariant_riemann(self):
        """
                Calcola il Riemann completamente covariante.
                R_ρσμν = g_λρ R^λ_σμν
                """
        self.compute_riemann_tensor()

        cov_riemann = MutableDenseNDimArray.zeros(
            self.dimension, self.dimension, self.dimension, self.dimension)

        for rho in range(self.dimension):
            for sigma in range(self.dimension):
                for mu in range(self.dimension):
                    for nu in range(self.dimension):
                        cov_riemann[rho, sigma, mu, nu] = sum(
                            self.metric[rho,lamb] * self.riemann_tensor[lamb, sigma, mu, nu]
                            for lamb in range(self.dimension)
                        )
        self.covariant_riemann = sp.simplify(cov_riemann)
        return self.covariant_riemann


    def compute_sectional_curvature_matrix(self):
        """
        Calcola la matrice S_ij = R_ijij/(g_ii g_jj-g_ij^2)
            delle curvature sezionali rispetto ai piani coordinati (i,j)
        Il caso della diagonale i=j non individua nessun piano ed è 
            gestito dalla funzione ponendo S_ii = 0. 
        """
        self.compute_covariant_riemann()
        R = self.covariant_riemann
        g = self.metric
        n = self.dimension
        S = MutableDenseNDimArray.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    S[i, j] += R[i, j, i, j] / (g[i, i] * g[j, j] - g[i, j]**2)

        return S

    def print_sectional_curvatures(self):
        S = self.compute_sectional_curvature_matrix()
        for i, coord1 in enumerate(self.coords):
            for j, coord2 in enumerate(self.coords):
                if i < j:
                    print(f'Sectional curvature in the plane (∂{coord1},∂{coord2}): K_{coord1}{coord2}:', S[i, j])
    

    def compute_ricci_tensor(self):
        """
        Calcola il tensore di Ricci.
        R_μν = R^ρ_μρν
        """
        self.compute_riemann_tensor()

        ricci_tensor = sp.MutableDenseNDimArray.zeros(self.dimension, self.dimension)
        for mu in range(self.dimension):
            for nu in range(self.dimension):
                ricci_tensor[mu, nu] = sum(self.riemann_tensor[rho, mu, rho, nu] for rho in range(self.dimension))

        self.ricci_tensor = sp.simplify(sp.Matrix(ricci_tensor))
        return self.ricci_tensor


    def compute_ricci_tensor2(self):
        """
        Calcola il tensore di Ricci.
        R_μν = g^ab R_aμbν
        """
        #self.compute_christoffel_symbols()
        self.compute_covariant_riemann()
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
        

    def compute_scalar_curvature(self):
        """
        Calcola la curvatura scalare.
        """
        self.compute_ricci_tensor()

        metric_inv = self.metric.inv()
        scalar_curvature = sum(
            metric_inv[mu, nu] * self.ricci_tensor[mu, nu]
            for mu in range(self.dimension) for nu in range(self.dimension)
        )

        self.scalar_curvature = sp.simplify(scalar_curvature)
        return self.scalar_curvature 


    def compute_kretschmann_scalar(self):
        """
        Calcola l'invariante di Kretschmann R_ρσμν*R^ρσμν
        """
        self.compute_riemann_tensor()
        g_inv = self.metric.inv()
        self.compute_covariant_riemann()

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
        :param: scale_factor è il fattore conforme di scala. Può essere una funzione sympy, una costante...
        """
        if self.metric.shape[0] + 1 != len(self.coords) + 1:
            raise ValueError("La metrica delle foglie deve essere n-dimensionale rispetto a n+1 coordinate totali.")

        lorentzian_metric = sp.eye(self.dimension + 1)
        lorentzian_metric[0, 0] = -1  # Aggiunge il termine dt^2
        lorentzian_metric[1:, 1:] = scale_factor ** 2 * self.metric

        return lorentzian_metric
    

    def compute_einstein_tensor(self):
        """
        Calcola il tensore di Einstein.
        """
        self.compute_scalar_curvature()
        self.einstein_tensor = sp.simplify(self.ricci_tensor - (1 / 2) * self.metric * self.scalar_curvature)
        return self.einstein_tensor

    def einstein_constant(self):
        """
        Calcola la costante di Einstein lambda:
            Ric=lambda*g.
        """
        self.compute_scalar_curvature()
        return self.scalar_curvature / self.dimension

    def is_einstein_mfd(self):
        """
            Verifica se è una varietà di Einstein
        """
        self.compute_scalar_curvature()
        l = self.einstein_constant()
        return self.ricci_tensor == l * self.metric

    def vacuum_einstein_eqs(self, Lambda):
        """
        Verifica True or False se una Manifold soddisfa le equazioni di Einstein.

        :param: Lambda è la costante cosmologica; può essere assegnata in sympy
        sia come Lambda = sympy.symbols('Lambda') che come vera e propria funzione
        sympy, o semplicemente come funzione costante.
        """
        self.compute_einstein_tensor()
        return self.einstein_tensor + Lambda * self.metric == sp.zeros(self.dimension, self.dimension)



    def inner_product(self, X, Y):
        """
        Calcola il prodotto scalare g(X, Y) tra due vettori X e Y utilizzando la metrica g.

        :param X: Vettore X (lista o matrice 1D di dimensione n)
        :param Y: Vettore Y (lista o matrice 1D di dimensione n)
        :return: Il prodotto scalare g(X, Y)
        """
        # Assicuriamoci che X e Y siano nel formato giusto
        #if len(X) != self.dimension or len(Y) != self.dimension:
         #   raise ValueError("I vettori X e Y devono essere di dimensione n.")

        # Calcolare il prodotto scalare g(X, Y) = X^T * g * Y
        inner_product = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                inner_product += self.metric[i, j] * X[i] * Y[j]

        return inner_product
    
    

    def compute_sectional_curvature(self, X, Y):
        """
        Calcola la curvatura sezionale per un piano definito dai vettori tangenti X e Y.

        :param X: Vettore tangente X (lista o matrice 1D di dimensione n).
        :param Y: Vettore tangente Y (lista o matrice 1D di dimensione n).
        :return: Valore della curvatura sezionale K(X, Y).
        """
        if len(X) != self.dimension or len(Y) != self.dimension:
            raise ValueError("I vettori X e Y devono avere la stessa dimensione delle coordinate della varietà.")

        self.compute_riemann_tensor()

        # Calcola il numeratore: R(X, Y, X, Y)
        R_XYXY = 0
        for mu in range(self.dimension):
            for nu in range(self.dimension):
                for sigma in range(self.dimension):
                    for rho in range(self.dimension):
                        R_XYXY += (
                                self.riemann_tensor[mu, nu, sigma, rho] * X[mu] * Y[nu] * X[sigma] * Y[rho]
                        )

        # Calcola il denominatore: ||X ∧ Y||^2 = g(X, X)g(Y, Y) - g(X, Y)^2
        g_XX = self.inner_product(X, X)
        g_YY = self.inner_product(Y, Y)
        g_XY = self.inner_product(X, Y)

        wedge_norm_squared = g_XX * g_YY - g_XY ** 2

        if wedge_norm_squared == 0:
            raise ValueError("I vettori X e Y non definiscono un piano valido (sono linearmente dipendenti).")

        # Calcola la curvatura sezionale
        sectional_curvature = R_XYXY / wedge_norm_squared
        self.sectional_curvature = sp.simplify(sectional_curvature)

        return self.sectional_curvature
    


    def compute_geodesic_equations(self):
        """
        Calcola simbolicamente le equazioni geodetiche per la varietà.

        Le equazioni hanno la forma:
            d^2 x^mu / dτ^2 + Γ^mu_{νρ} (dx^ν / dτ) (dx^ρ / dτ) = 0

        :return: Lista delle equazioni geodetiche, una per ogni coordinata, in forma sp.Equality.
        """
        self.compute_christoffel_symbols()

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
        self.compute_geodesic_equations()
        eqs_list = []
        for i, coord in enumerate(self.coords):
            eqs_list.append(self.geodesics[i])
            print(f"\nGeodesic equation along {coord}:")
            sp.pprint(eqs_list[i])  # Stampa leggibile in console
            print("\nLaTeX format:")
            print(f'{sp.printing.latex(eqs_list[i])}')  # Output LaTeX-friendly


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

        self.metric = None #questo serve per gestire bene l'ereditarietà di certi metodi di Manifold
        self.coords = self.sub_coords #come ad esempio self.compute_christoffels_symbols()

    def compute_embedding_jacobian(self):
        self.embedding_jacobian = Matrix([
            [diff(f, coord) for coord in self.sub_coords]
            for f in self.embedding
            ])

        return self.embedding_jacobian

    def compute_induced_metric(self):
        """
        Calcola la metrica indotta sulla sottovarietà.
        :return: Matrice simbolica della metrica indotta.
        """
        g = self.ambient_manifold.metric  # Metrica della varietà ambiente
        self.compute_embedding_jacobian() # Jacobiano dell'immersione

        # Metrica indotta: G_ab = (Jacobian)^T * g * (Jacobian)
        self.induced_metric = sp.simplify(self.embedding_jacobian.T * g * self.embedding_jacobian)
        self.metric = self.induced_metric #serve per poterci agire con metodi di Manifold, e.g. compute_christoffel_symbols()
        #self.coords = self.sub_coords

        return self.induced_metric


    def compute_normal_field(self):
        """
        Calcola il campo normale della submanifold nell'ambiente.
        :return: Lista di vettori normali simbolici.
        """
        jacobian = self.compute_embedding_jacobian()
        ambient_metric = self.ambient_manifold.metric
       
        tangent_vectors = [jacobian[:, i] for i in range(self.dimension)]
        d = len(self.embedding)

        normal_vectors = [sp.symbols(f'n{i+1}') for i in range(d)]  # Lista di simboli normali
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
        self.compute_scalar_curvature()
        if self.scalar_curvature >= 0:
            self.normal_field = sp.Matrix([solutions[0]])
        else:
             self.normal_field = sp.Matrix([solutions[1]]) 
            
        #gestione del caso n-k>1: mancante
        
        self.normal_field = self.normal_field.subs(sp.I, 1) #normalizza a reali eventuali vettori complessi
        # questo punto è poco chiaro, non dovrebbe succedere
        return sp.simplify(self.normal_field)


    def compute_IInd_fundamental_form(self):
        """
        Calcola la seconda forma fondamentale per la sottovarietà in un ambiente con connessione in generale non piatta.
        :return: Matrice simbolica della seconda forma fondamentale.
        """
        self.compute_embedding_jacobian()
        self.ambient_manifold.compute_christoffel_symbols()
        Gamma = self.ambient_manifold.christoffel_symbols  
        self.compute_normal_field()

        II = sp.zeros(self.dimension, self.dimension)  
        tangent_vectors = [self.embedding_jacobian[:, i] for i in range(self.dimension)]
        coords = self.sub_coords

        num_vectors = len(tangent_vectors) #è la dimensione dell'immagine dell'embedding
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

    def compute_mean_curvatureII(self):
        """
        Calcola la curvatura media della varietà immersa.
        :param: Normal vector field in forma di vettore sympy
        :return: Scalare in forma di sympy function o costante
                Traccia della matrice II
        """
        self.compute_IInd_fundamental_form()
        self.compute_induced_metric()

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
        
    #di seguito dei doppioni con inserimento manuale del normal vector field
    def compute_second_fundamental_form(self, normal_field):
        """
        Calcola la seconda forma fondamentale per la sottovarietà in un ambiente con connessione in generale non piatta.
        :param: normal_field: Campo normale in forma di vettore SymPy.
        :return: Matrice simbolica della seconda forma fondamentale.
        """
        self.compute_embedding_jacobian()
        self.ambient_manifold.compute_christoffel_symbols()
        Gamma = self.ambient_manifold.christoffel_symbols  

        II = sp.zeros(self.dimension, self.dimension)  
        tangent_vectors = [self.embedding_jacobian[:, i] for i in range(self.dimension)]
        coords = self.sub_coords

        num_vectors = len(tangent_vectors) #è la dimensione dell'immagine dell'embedding
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


    def compute_mean_curvature(self, normal_field):
        """
        Calcola la curvatura media della varietà immersa.
        :param: Normal vector field in forma di vettore sympy
        :return: Scalare in forma di sympy function o costante
                Traccia della matrice II
        """
        self.compute_second_fundamental_form(normal_field)
        self.compute_induced_metric()

        I = self.induced_metric.inv()
        II = self.second_fundamental_form
        H = 0
        for a in range(self.dimension):
            for b in range(self.dimension):
                if a == b:
                    H += I[a, b] * II[a, b]

        self.mean_curvature = sp.simplify(H)
        return self.mean_curvature





