# %% [markdown]
# # Processos Aleatorios em Tempo Discreto -- Parte 4: Estacionariedade e WSS
#
# - Estacionario de 1a ordem: $f_{x(n)}(\alpha) = f_{x(n+k)}(\alpha)$ para
#   todo $k$ $\Rightarrow$ $m_x(n)=m_x$ constante.
# - Estacionario de 2a ordem: densidade conjunta de $(x(n_1),x(n_2))$
#   invariante a deslocamento $\Rightarrow$ $r_x(k,l)$ depende so de $k-l$.
# - Estacionario em sentido amplo (WSS): (1) $m_x(n)=m_x$ constante, (2)
#   $r_x(k,l)$ depende so de $k-l$, (3) $c_x(0)<\infty$.
#
# Este script compara numericamente dois processos muito parecidos na
# forma, mas um WSS e outro nao.
#
# Convertido do MATLAB Live Script (`.mlx`) para Python. Use um editor com
# suporte a celulas Jupytext (VS Code, Spyder, PyCharm) para rodar como
# notebook interativo, ou execute como script comum.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Caso 1 (NAO WSS): amplitude aleatoria, fase fixa
#
# $$x(n) = A\cos(n\omega_0), \qquad A \sim \mathcal U(-1,1) \text{ (fase fixa, nao aleatoria)}$$
#
# A media de conjunto e $E\{A\}\cos(n\omega_0)$. Como $E\{A\}=0$ aqui, a
# media da certo (fica em 0), mas a autocorrelacao
# $r_x(k,l) = E\{A^2\}\cos(k\omega_0)\cos(l\omega_0)$ nao depende so de
# $k-l$ -- verificamos isso numericamente. Para deixar tambem a media
# dependente de $n$ (caso classico do contraexemplo dos slides), usamos
# $A \sim \mathcal U(0,2)$, com $E\{A\}=1 \neq 0$:

# %%
rng = np.random.default_rng(2)
omega0 = np.pi / 5
N = 40
n = np.arange(N)
Mreal = 20000

A = 2 * rng.random(Mreal)                 # A ~ U(0,2), E{A}=1, shape (Mreal,)
X1 = np.outer(np.cos(n * omega0), A)      # x(n) = A cos(n w0), fase fixa; shape (N, Mreal)

m1_hat = X1.mean(axis=1)
m1_teo = A.mean() * np.cos(n * omega0)    # E{A} cos(n w0): depende de n

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
fig.canvas.manager.set_window_title('Caso 1: NAO estacionario (amplitude aleatoria, fase fixa)')

ax1.plot(n, m1_hat, 'o-', label='estimada')
ax1.plot(n, m1_teo, 'r--', label=r'teoria: E(A)cos(n$\omega_0$)')
ax1.set_xlabel('n')
ax1.set_ylabel('m_x(n)')
ax1.legend()
ax1.set_title(r'Media de conjunto DEPENDE de n $\Rightarrow$ processo NAO estacionario')

l0 = 0
r1_hat = np.mean(X1 * X1[l0, :], axis=1)
ax2.plot(n, r1_hat, 'o-')
ax2.set_xlabel('k')
ax2.set_ylabel('r_x(k,l_0=0)')
ax2.set_title('Autocorrelacao r_x(k,0) -- tambem nao tem a forma r_x(k-l_0)')

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Caso 2 (WSS): fase aleatoria uniforme
#
# $$x(n) = A\sin(n\omega_0+\phi), \qquad \phi \sim \mathcal U(-\pi,\pi)$$
#
# Aqui $m_x(n)=0$ para todo $n$, e
# $r_x(k,l) = \frac{1}{2} A^2\cos[(k-l)\omega_0]$ depende so da diferenca
# $k-l$ -- a assinatura de um processo WSS.

# %%
Aconst = 1.4
phi = rng.uniform(-np.pi, np.pi, size=(1, Mreal))
n_col = n.reshape(-1, 1)
X2 = Aconst * np.sin(n_col * omega0 + phi)   # shape (N, Mreal), broadcasting

m2_hat = X2.mean(axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
fig.canvas.manager.set_window_title('Caso 2: WSS (fase aleatoria)')

ax1.plot(n, m2_hat, 'o-')
ax1.axhline(0, color='r', linestyle='--')
ax1.set_xlabel('n')
ax1.set_ylabel('m_x(n)')
ax1.set_title(r'Media de conjunto CONSTANTE ($\approx$ 0) para todo n $\Rightarrow$ consistente com WSS')

l0 = 0
r2_hat = np.mean(X2 * X2[l0, :], axis=1)
r2_teo = 0.5 * Aconst**2 * np.cos((n - l0) * omega0)
ax2.plot(n, r2_hat, 'o', label='estimada')
ax2.plot(n, r2_teo, '-', label=r'teoria: 0.5A^2cos((k-l_0)$\omega_0$)')
ax2.set_xlabel('k')
ax2.set_ylabel('r_x(k,l_0=0)')
ax2.legend()
ax2.set_title(r'Autocorrelacao depende so de k-l_0 $\Rightarrow$ consistente com WSS')

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Verificando a forma $r_x(k-l)$ para varios deslocamentos $l_0$
#
# Se o processo e de fato WSS, $r_x(k,l_0)$ para diferentes $l_0$ deve
# ser apenas uma versao deslocada da mesma funcao de $k-l_0$. Sobrepomos
# as curvas para $l_0 = 0, 5, 10$:

# %%
fig = plt.figure()
fig.canvas.manager.set_window_title('WSS: r_x(k,l0) so depende de k-l0')
for l0 in [0, 5, 10]:
    r_hat = np.mean(X2 * X2[l0, :], axis=1)
    plt.plot(n - l0, r_hat, 'o-', label=f'l_0={l0}')
plt.legend()
plt.grid(True)
plt.xlabel('k - l_0')
plt.ylabel('r_x(k,l_0)')
plt.title(r'As tres curvas colapsam sobre a mesma forma $\Rightarrow$ WSS')
plt.show()

# %% [markdown]
# ## Resumo
#
# - O processo de Bernoulli (script 01) e a senoide de fase aleatoria
#   (Caso 2 acima) sao WSS.
# - $x(n)=A\cos(n\omega_0)$ com fase fixa (Caso 1) nao e WSS -- a
#   aleatoriedade sozinha nao garante estacionariedade; o que importa e
#   como as estatisticas de conjunto se comportam sob deslocamento
#   temporal.
