from geo_diff import Manifold
import sympy as sp
from sympy import symbols, Matrix

# Poincaré half plane
t, phi = symbols('t phi')
g_hyperbolic = Matrix([[1, 0], [0, sp.sinh(t)**2]])
manifold_hyperbolic = Manifold(g_hyperbolic, [t, phi])

manifold_hyperbolic.compute_christoffel_symbols()
manifold_hyperbolic.compute_riemann_tensor()
manifold_hyperbolic.compute_ricci_tensor()
manifold_hyperbolic.compute_scalar_curvature()

manifold_hyperbolic.pretty_print_matrix(manifold_hyperbolic.ricci_tensor, "Ricci Tensor")
print("Scalar Curvature:", manifold_hyperbolic.scalar_curvature)

# Poincaré half space
t, theta, phi = symbols('t theta phi')
g_3d_hyp = Matrix([
    [1, 0, 0],
    [0, sp.sinh(t)**2, 0],
    [0, 0, sp.sinh(t)**2 * sp.sin(theta)**2]
])
manifold_3d_hyp = Manifold(g_3d_hyp, [t, theta, phi])

manifold_3d_hyp.compute_christoffel_symbols()
manifold_3d_hyp.compute_riemann_tensor()
manifold_3d_hyp.compute_ricci_tensor()
manifold_3d_hyp.compute_scalar_curvature()

manifold_3d_hyp.pretty_print_matrix(manifold_3d_hyp.ricci_tensor, "3D Ricci Tensor")
print("3D Scalar Curvature:", manifold_3d_hyp.scalar_curvature)


# Poincaré half hyperspace

t, theta, psi, phi = symbols('t theta psi phi')
g_4d_hyp = Matrix([
    [1, 0, 0, 0],
    [0, sp.sinh(t)**2, 0, 0],
    [0, 0, sp.sinh(t)**2 * sp.sin(theta)**2, 0],
    [0, 0, 0, sp.sinh(t)**2 * sp.sin(theta)**2 * sp.sin(psi)**2]
])
manifold_4d_hyp = Manifold(g_4d_hyp, [t, theta, psi, phi])

manifold_4d_hyp.compute_christoffel_symbols()
manifold_4d_hyp.compute_riemann_tensor()
manifold_4d_hyp.compute_ricci_tensor()
manifold_4d_hyp.compute_scalar_curvature()

manifold_4d_hyp.pretty_print_matrix(manifold_4d_hyp.ricci_tensor, "4D Ricci Tensor")
print("4D Scalar Curvature:", manifold_4d_hyp.scalar_curvature)