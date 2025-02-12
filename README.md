# Differential Geometry with Python
A Python library for differential geometry, mainly based on `SymPy`, includes tensor analysis on manifolds, designed for computing fundamental geometric objects such as intrinsic and extrinsic curvatures, geodesics, covariant derivatives, divergences, gradients, Laplacians, and for verifying solutions to Einstein's equations.

## Overview
In this repository you can find my [Manifold and Submanifold class](https://github.com/Joyboy0056/The-Geometric-Project/blob/main/geo_diff.py) to handle computations within the framework of Differential Geometry. Throughout my academic education, despite attending several classes on General Relativity and Differential Geometry, I have never had the opportunity to directly compute Einstein's equations, which always felt quite strange to me. For this reason, I decided to develop my own library (based on `SymPy`) to handle tensorial calculations on manifolds. Additionally, you can find a [small notebook](https://github.com/Joyboy0056/The-Geometric-Project/blob/main/hyperbolic_egs.ipynb) where I explicitly verify, step by step, that constant curvature geometries satisfy Einstein's vacuum equations, with a particular focus on the hyperbolic solution. Moreover, a brief insight into the topic of geodesics has also been provided in the [Geodesics](https://github.com/Joyboy0056/The-Geometric-Project/blob/main/Geodesics.ipynb) notebook, where they are first computed symbolically, in the fashion of this library, and then solved numerically using `NumPy` methods.

---

### Briefs geo-diff notes
Geodesics of a manifold $(M^n,g)$ are curves of the form $\tau\mapsto\left(x^1(\tau),...,x^n(\tau)\right),$  which are solutions of the following differential equation: 

$\sum_{\nu,\lambda=1}^n\frac{d^2x^\mu}{d\tau^2}+\Gamma^\mu_{\nu\lambda}\frac{d x^\nu}{d\tau}\frac{d x^\lambda}{d\tau}=0,$ per ogni $\mu=1,...,n$

Where:

- $x^\mu$ are local coordinates on the manifold.
- $\Gamma^\mu_{\nu\lambda}$ are the Christoffel symbols of the Levi-Civita connection on the manifold.
- $\tau$ is the affine parameter (arc length) along a curve on the manifold.

Geodesics are intrinsic in a certain sense, meaning that they depend on the metric $g\in T^*M\odot T^*M$, which is independent of the coordinates in which it is expressed, given by $g=g_{\mu\nu}dx^\mu\odot dx^\nu$, in Einstein notation.

In general, the components of a metric $g$ on $M^n$ are computed via pullback from an ambient space, usually Euclidean. For instance, if there exists an embedding $F:M^n\hookrightarrow\mathbb{R}^m$, with $\mathbf{\delta}=\delta_{ab}dx^a\odot dx^b$ being the standard Euclidean metric, then $g="J_F^T\cdot J_F"=\delta_{ab}\frac{\partial F^a}{\partial x^\mu}\frac{\partial F^b}{\partial x^\nu} dx^\mu\odot dx^\nu$. In other words, $g$ is the pullback of $\delta$ through the parameterization $F$.

The Christoffel symbols are computed using the formula $\Gamma^\rho_{\mu\nu}=\frac{1}{2}\sum_{\lambda}g^{\rho\lambda}\left(\partial_\nu g_{\mu\lambda}+\partial_\mu g_{\nu\lambda}-\partial_\lambda g_{\mu\nu}\right)$, from which the Riemann curvature tensor can be determined as $\mathbf{Riem}=\sum_{\rho,\sigma,\mu,\nu}R^\rho_{\sigma\mu\nu}\partial_\rho\otimes dx^\sigma\otimes dx^\mu\otimes dx^\nu$ where its components are given by
$R^\rho_{\sigma\mu\nu}=\sum_\lambda \partial_\mu\Gamma^\rho_{\nu\sigma}-\partial_\nu\Gamma^\rho_{\mu\sigma}+\Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma}+\Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$. This is the most general object describing the curvature of a metric $g$, and it possesses various nontrivial traces, including the Ricci tensor $\mathbf{Ric}=\sum_{\rho,\mu,\nu}\overbrace{R^\rho_{\mu\rho\nu}}^{=:R_{\mu\nu}} dx^\mu\odot dx^\nu$ and the Ricci scalar (or scalar curvature) $\mathbf{R}=\sum_{\mu,\nu}g^{\mu\nu}R_{\mu\nu}$. In the case $n=2$, the Riemann tensor $\mathbf{Riem}$ is entirely determined by the scalar curvature $\mathbf{R}$, while for $n=3$ it is determined by the Ricci tensor $\mathbf{Ric}$. There are also other nontrivial traces of the Riemann tensor that are significant in higher dimensions, one of the most fundamental being the Kretschmann invariant $\mathbf{K}=\sum_{\mu,\nu,\rho,\sigma}R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}$.



