Le curve geodetiche di una varietà $(M^n,g)$ sono soluzioni della seguente equazione differenziale:

$\sum_{\nu,\lambda=1}^n\frac{d^2x^\mu}{d\tau^2}+\Gamma^\mu_{\nu\lambda}\frac{d x^\nu}{d\tau}\frac{d x^\lambda}{d\tau}=0,$ per ogni $\mu=1,...,n$

Dove:

- $x^\mu$ sono coordinate locali sulla varietà.
- $\Gamma^\mu_{\nu\lambda}$ sono i simboli di Christoffel della! connessione di Levi-Civita sulla varietà.
- $\tau$ è il parametro affine (lunghezza d'arco) lungo una curva sulla varietà.

Esse sono in un certo senso intrinseche, nel senso che dipendono dalla metrica $g\in T^*M\odot T^*M$, la quale è indipendente dalle coordinate rispetto alla quale viene descritta come $g=g_{\mu\nu}dx^\mu\odot dx^\nu$.
Generalmente, le componenti della metrica sono calcolate attraverso il pull-back da un ambiente come $g_{\mu\nu}=$
$$\vdots$$

I simboli di Christoffel si calcolano attraverso la formula $\Gamma^\rho_{\mu\nu}=\frac{1}{2}\sum_{\lambda}g^{\rho\lambda}\left(\partial_\nu g_{\mu\lambda}+\partial_\mu g_{\nu\lambda}-\partial_\lambda g_{\mu\nu}\right)$, dai quali si può calcolare il tensore di curvatura di Riemann $\mathbf{Riem}=R^\rho_{\sigma\mu\nu}\partial_\rho\otimes dx^\sigma\otimes dx^\mu\otimes dx^\nu$ di componenti 
$R^\rho_{\sigma\mu\nu}=\sum_\lambda \partial_\mu\Gamma^\rho_{\nu\sigma}-\partial_\nu\Gamma^\rho_{\mu\sigma}+\Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma}+\Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$.
