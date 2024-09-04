import sympy as sp
from Christoffels import christoffel_symbols, pretty_print_christoffel_matrices

print('\nChristoffels di R2 standard (x,y):')
# Definizione della metrica per R2
# n = 2  # Dimensione dello spazio
x, y = sp.symbols('x y')  # Coordinate simboliche
coords = [x, y]

metric = sp.Matrix([[1, 0], [0, 1]])  # Metrica identità per R2

# Calcola i simboli di Christoffel
christoffel_matrices = christoffel_symbols(metric, coords)
pretty_print_christoffel_matrices(christoffel_matrices)

print('\nChristoffels del disco euclideo D2 (r, theta):')
# Definizione della metrica per il disco unitario in coordinate polari

r, theta = sp.symbols('r theta')  # Coordinate simboliche
polar_coords = [r, theta]

# Metrica del disco unitario
polar_metric = sp.Matrix([
    [1, 0],
    [0, r**2]
])

# Calcola i simboli di Christoffel
christoffel_matrices = christoffel_symbols(polar_metric, polar_coords)
pretty_print_christoffel_matrices(christoffel_matrices)


print('\nChristoffels della sfera S2 (t, theta):')
t, theta = sp.symbols('t theta')
round_coords = [t, theta]

round_metric = sp.Matrix([
    [1, 0],
    [0, sp.sin(t)**2]
])

# Calcola i simboli di Christoffel
christoffel_matrices = christoffel_symbols(round_metric, round_coords)
pretty_print_christoffel_matrices(christoffel_matrices)

print('\nChristoffels dello spazio sferico (r, theta, phi):')
r, theta, phi = sp.symbols('r theta phi')
coords = [r, theta, phi]

# Definizione della metrica per lo spazio R^3 in coordinate sferiche
metric = sp.Matrix([[1, 0, 0],
                    [0, r ** 2, 0],
                    [0, 0, r ** 2 * sp.cos(theta) ** 2]])

# Calcolo dei simboli di Christoffel
christoffel_matrices = christoffel_symbols(metric, coords)
pretty_print_christoffel_matrices(christoffel_matrices)


print('\nChristoffels dello spazio cilindrico (r, theta, z):')
r, theta, z = sp.symbols('r theta z')
coords = [r, theta, z]

metric = sp.Matrix([[1, 0, 0],
                    [0, r ** 2, 0],
                    [0, 0, 1]])

# Calcolo dei simboli di Christoffel
christoffel_matrices = christoffel_symbols(metric, coords)
pretty_print_christoffel_matrices(christoffel_matrices)

print('\nChristoffels della sfera (deSitter?) S3 (t, phi, theta):')
t, phi, theta = sp.symbols('t, phi, theta')
coords = [t, phi, theta]
# recall g_{Sn}=dt^2+sin^2(t)g_{Sn-1}
round_metric = sp.Matrix([
    [1, 0, 0],
    [0, sp.sin(t)**2, 0],
    [0, 0, sp.sin(t)**2 * sp.cos(phi)**2]
])

# Calcolo dei simboli di Christoffel
christoffel_matrices = christoffel_symbols(round_metric, coords)
pretty_print_christoffel_matrices(christoffel_matrices)

print('\nChristoffels di R3 ristretta a S2 (phi, theta):')
phi, theta = sp.symbols('phi theta')
coords = [phi, theta]

metric = sp.Matrix([[1, 0],
                    [0, sp.cos(phi)**2],
                    ])

# Calcolo dei simboli di Christoffel
christoffel_matrices = christoffel_symbols(metric, coords)
pretty_print_christoffel_matrices(christoffel_matrices)


print('\nChristoffels della palla (AntideSitter?) H3 (t, phi, theta):')
r, phi, theta = sp.symbols('r, phi, theta')
coords = [r, phi, theta]
# recall g_{Sn}=dt^2+sin^2(t)g_{Sn-1}
hyp_metric = sp.Matrix([
    [1/(1+r**2), 0, 0],
    [0, r**2, 0],
    [0, 0, r**2 * sp.cos(phi)**2]
])

# Calcolo dei simboli di Christoffel
christoffel_matrices = christoffel_symbols(hyp_metric, coords)
pretty_print_christoffel_matrices(christoffel_matrices)