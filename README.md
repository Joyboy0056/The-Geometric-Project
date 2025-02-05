# Differential Geometry with Python
A Python library for differential geometry, including tensor analysis on manifolds, designed for computing fundamental geometric objects such as intrinsic and extrinsic curvatures, geodesics, covariant derivatives, divergences, gradients, Laplacians, and for verifying solutions to Einstein's equations.

## Overview
In this repository you can find my [Manifold and Submanifold class](https://github.com/Joyboy0056/The-Geometric-Project/blob/main/geo_diff.py) to handle computations within the framework of Differential Geometry. Throughout my academic education, despite attending several classes on General Relativity and Differential Geometry, I have never had the opportunity to directly compute Einstein's equations, which always felt quite strange to me. For this reason, I decided to develop my own library based on SymPy to handle tensorial calculations on manifolds. Additionally, you can find a [small notebook](https://github.com/Joyboy0056/The-Geometric-Project/blob/main/hyperbolic_egs.ipynb) where I explicitly verify, step by step, that constant curvature geometries satisfy Einstein's vacuum equations, with a particular focus on the hyperbolic solution.

---

### Briefs geo-diff notes
Le geodetiche di una varietà $(M^n,g)$ sono curve della forma $\tau\mapsto\left(x^1(\tau),...,x^n(\tau)\right),$ soluzioni della seguente equazione differenziale:

$\sum_{\nu,\lambda=1}^n\frac{d^2x^\mu}{d\tau^2}+\Gamma^\mu_{\nu\lambda}\frac{d x^\nu}{d\tau}\frac{d x^\lambda}{d\tau}=0,$ per ogni $\mu=1,...,n$

Dove:

- $x^\mu$ sono coordinate locali sulla varietà.
- $\Gamma^\mu_{\nu\lambda}$ sono i simboli di Christoffel della! connessione di Levi-Civita sulla varietà.
- $\tau$ è il parametro affine (lunghezza d'arco) lungo una curva sulla varietà.

Esse sono in un certo senso intrinseche, nel senso che dipendono dalla metrica $g\in T^*M\odot T^*M$, la quale è indipendente dalle coordinate rispetto alla quale viene descritta come $g=g_{\mu\nu}dx^\mu\odot dx^\nu$, in notazione di Einstein.

Generalmente, le componenti di una metrica $g$ su $M^n$ sono calcolate attraverso il pull-back da un ambiente, di solito Euclideo: ad esempio, se esiste un embedding $F:M^n\hookrightarrow\mathbb{R}^m$, con $\mathbf{\delta}=\delta_{ab}dx^a\odot dx^b$ la metrica Euclidea standard, allora $g="J_F^T\cdot J_F"=\delta_{ab}\frac{\partial F^a}{\partial x^\mu}\frac{\partial F^b}{\partial x^\nu} dx^\mu\odot dx^\nu$. In altre parole, $g$ è il pull-back di $\mathbf{\delta}$ tramite la parametrizzazione $F$.

I simboli di Christoffel si calcolano attraverso la formula $\Gamma^\rho_{\mu\nu}=\frac{1}{2}\sum_{\lambda}g^{\rho\lambda}\left(\partial_\nu g_{\mu\lambda}+\partial_\mu g_{\nu\lambda}-\partial_\lambda g_{\mu\nu}\right)$, dai quali si può calcolare il tensore di curvatura di Riemann $\mathbf{Riem}=\sum_{\rho,\sigma,\mu,\nu}R^\rho_{\sigma\mu\nu}\partial_\rho\otimes dx^\sigma\otimes dx^\mu\otimes dx^\nu$ di componenti 
$R^\rho_{\sigma\mu\nu}=\sum_\lambda \partial_\mu\Gamma^\rho_{\nu\sigma}-\partial_\nu\Gamma^\rho_{\mu\sigma}+\Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma}+\Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$. Esso è l'oggetto più generale che descrive la curvatura di una metrica $g$ e possiede varie tracce non banali, tra cui il Ricci $\mathbf{Ric}=\sum_{\rho,\mu,\nu}\overbrace{R^\rho_{\mu\rho\nu}}^{=:R_{\mu\nu}} dx^\mu\odot dx^\nu$ e la curvatura scalare $\mathbf{R}=\sum_{\mu,\nu}g^{\mu\nu}R_{\mu\nu}$. Nel caso $n=2$, $\mathbf{Riem}$ è completamente determinato dalla curvatura scalare $\mathbf{R}$, mentre per $n=3$ dal Ricci $\mathbf{Ric}$.



