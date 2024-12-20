Le curve geodetiche di una varietà $(M^n,g)$ sono soluzioni della seguente equazione differenziale:

$\sum_{\nu,\lambda=1}^n\frac{d^2x^\mu}{d\tau^2}+\Gamma^\mu_{\nu\lambda}\frac{d x^\nu}{d\tau}\frac{d x^\lambda}{d\tau}=0,$ per ogni $\mu=1,...,n$

Dove:

- $x^\mu$ sono coordinate locali sulla varietà.
- $\Gamma^\mu_{\nu\lambda}$ sono i simboli di Christoffel della! connessione di Levi-Civita sulla varietà.
- $\tau$ è il parametro affine (lunghezza d'arco) lungo una curva sulla varietà.

Esse sono in un certo senso intrinseche, nel senso che dipendono dalla metrica $g\in T^*M\odot T^*M$, la quale è indipendente dalle coordinate rispetto alla quale viene descritta come $g=g_{\mu\nu}dx^\mu\odot dx^\nu$, in notazione di Einstein.

Generalmente, le componenti della metrica sono calcolate attraverso il pull-back da un ambiente, di solito Euclideo: ad esempio, se esiste un embedding $F:(M^n,g)\hookrightarrow(\mathbb{R}^m,\mathbf{\delta})$, dove $\mathbf{\delta}=\delta_{ab}dx^a\odot dx^b$ è la metrica Euclidea standard, allora $g=\nabla F^T\cdot\nabla F=\sum_{a,b}\delta_{ab}(\nabla F^T)^a_\mu(\nabla F)^b_\nu dx^\mu\odot dx^\nu$. In altre parole, $g$ è il pull-back di $\mathbf{\delta}$ tramite la parametrizzazione $F$.

I simboli di Christoffel si calcolano attraverso la formula $\Gamma^\rho_{\mu\nu}=\frac{1}{2}\sum_{\lambda}g^{\rho\lambda}\left(\partial_\nu g_{\mu\lambda}+\partial_\mu g_{\nu\lambda}-\partial_\lambda g_{\mu\nu}\right)$, dai quali si può calcolare il tensore di curvatura di Riemann $\mathbf{Riem}=R^\rho_{\sigma\mu\nu}\partial_\rho\otimes dx^\sigma\otimes dx^\mu\otimes dx^\nu$ di componenti 
$R^\rho_{\sigma\mu\nu}=\sum_\lambda \partial_\mu\Gamma^\rho_{\nu\sigma}-\partial_\nu\Gamma^\rho_{\mu\sigma}+\Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma}+\Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$. Esso è l'oggetto più generale che descrive la curvatura di una metrica $g$ e possiede varie tracce non banali, tra cui il Ricci $\mathbf{Ric}=\sum_{\rho}\overbrace{R^\rho_{\mu\rho\nu}}^{=:R_{\mu\nu}} dx^\mu\odot dx^\nu$ e la curvatura scalare $\mathbf{R}=\sum_{\mu,\nu}g^{\mu\nu}R_{\mu\nu}$. Nel caso $n=2$, $\mathbf{Riem}$ è completamente determinato dalla curvatura scalare $\mathbf{R}$, mentre per $n=3$ dal Ricci $\mathbf{Ric}$.
