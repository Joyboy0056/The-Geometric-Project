import sympy as sp

def christoffel_symbols(metric, coords):
    """
    Calcola i simboli di Christoffel per una metrica data in dimensione n.

    :param metric: Matrice della metrica (Matrix di SymPy).
    :param coords: Coordinate simboliche (lista di Symbol di SymPy).
    :return: Lista di matrici SymPy dei simboli di Christoffel.
    """
    n = metric.shape[0]  # Dimensione dello spazio
    g_inv = metric.inv()  # Calcola la matrice inversa della metrica

    # Lista per le matrici dei simboli di Christoffel
    christoffel_matrices = []

    # Calcola i simboli di Christoffel
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


def pretty_print_christoffel_matrices(christoffel_matrices):
    """
    Stampa le matrici dei simboli di Christoffel in un formato leggibile.
    """
    for k, matrix in enumerate(christoffel_matrices):
        print(f'\nGamma^{k+1}:')
        matrix_list = [[matrix[i, j] for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]
        for row in matrix_list:
            print(row)