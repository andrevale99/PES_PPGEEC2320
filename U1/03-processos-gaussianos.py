# %% [markdown]
# # Processos Aleatorios em Tempo Discreto -- Parte 3: Processos Gaussianos
#
# Um vetor $\mathbf{x}=[x_1,\ldots,x_n]^T$ e Gaussiano se sua densidade
# conjunta e
#
# $$f_{\mathbf x}(\mathbf x) = \frac{1}{(2\pi)^{n/2}|\mathbf C_x|^{1/2}} \exp\left\{-\frac{1}{2}(\mathbf x-\mathbf m_x)^T \mathbf C_x^{-1}(\mathbf x-\mathbf m_x)\right\}$$
#
# Um processo $x(n)$ e dito Gaussiano se toda colecao finita de amostras
# e conjuntamente Gaussiana. Consequencia importante: um processo
# Gaussiano fica completamente especificado pelo vetor de medias e pela
# matriz de covariancia.
#
# Convertido do MATLAB Live Script (`.mlx`) para Python. Use um editor com
# suporte a celulas Jupytext (VS Code, Spyder, PyCharm) para rodar como
# notebook interativo, ou execute como script comum.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

# %% [markdown]
# ## Gerando um vetor Gaussiano com covariancia prescrita
#
# Dada uma matriz de covariancia $\mathbf C_x$ (simetrica, definida
# positiva) desejada, podemos gerar amostras via decomposicao de
# Cholesky: se $\mathbf C_x = \mathbf L \mathbf L^T$ e
# $\mathbf v \sim \mathcal N(0,\mathbf I)$, entao
# $\mathbf x = \mathbf L \mathbf v$ tem covariancia $\mathbf C_x$.
#
# Usamos como exemplo a autocorrelacao de um processo AR(1)-like,
# $r_x(k)=\alpha^{|k|}$ (a mesma familia dos scripts 05 e 08), com $p=4$
# (vetor de 5 amostras). Nota: nao usamos aqui a senoide de fase
# aleatoria do Exemplo 3.3.4 -- por ser soma de apenas 2 exponenciais
# complexas, sua matriz de autocorrelacao tem posto no maximo 2, e para
# $p>1$ fica apenas semidefinida positiva (singular), o que faz a
# fatoracao de Cholesky falhar ("matriz precisa ser definida positiva").
# O AR(1)-like abaixo e genuinamente definido positivo para qualquer
# tamanho.

# %%
alpha = 0.6
p = 4
k = np.arange(p + 1)
r = alpha ** np.abs(k)
Cx = toeplitz(r)                    # matriz de autocorrelacao/covariancia teorica (media zero)

print('Matriz de covariancia teorica C_x:')
print(Cx)

L = np.linalg.cholesky(Cx)          # numpy ja retorna a fatoracao "lower" (Cx = L @ L.T)
Mreal = 50000
rng = np.random.default_rng()
V = rng.standard_normal((p + 1, Mreal))
Xg = L @ V                          # cada coluna: uma amostra do vetor Gaussiano

Cx_hat = np.cov(Xg)                 # covariancia amostral (linhas=variaveis, colunas=observacoes)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
fig.canvas.manager.set_window_title('Covariancia teorica vs estimada')

im1 = ax1.imshow(Cx)
ax1.set_aspect('equal')
ax1.set_title('C_x teorica')
fig.colorbar(im1, ax=ax1)

im2 = ax2.imshow(Cx_hat)
ax2.set_aspect('equal')
ax2.set_title('C_x estimada (amostral)')
fig.colorbar(im2, ax=ax2)

fig.tight_layout()
plt.show()

erro_max = np.max(np.abs(Cx - Cx_hat))
print(f'Erro maximo |C_x - C_x_hat| = {erro_max:.4f}')

# %% [markdown]
# ## Verificando a marginal Gaussiana
#
# Cada componente $x_i$ do vetor deve ter distribuicao
# $\mathcal N(0,\sigma_i^2)$, com $\sigma_i^2 = C_x(i,i)$.

# %%
i = 0    # indice 0-based (equivalente a i=1 no MATLAB 1-based)

fig = plt.figure()
fig.canvas.manager.set_window_title('Marginal Gaussiana')
plt.hist(Xg[i, :], bins=80, density=True)
xx = np.linspace(Xg[i, :].min(), Xg[i, :].max(), 200)
sig = np.sqrt(Cx[i, i])
gausspdf = np.exp(-0.5 * (xx / sig) ** 2) / (sig * np.sqrt(2 * np.pi))  # pdf N(0,sig^2), sem toolbox
plt.plot(xx, gausspdf, 'r-', linewidth=1.5)
plt.xlabel(f'x_{i}')
plt.ylabel('densidade')
plt.legend(['N(0,C_x(i,i)) teorica', 'histograma (amostras)'])
plt.title(f'Marginal de x({i}): Gaussiana com variancia {Cx[i, i]:.3f}')
plt.show()

# %% [markdown]
# ## Teorema Central do Limite: por que processos reais sao "quase" Gaussianos
#
# Muitos ruidos fisicos surgem da soma de muitas contribuicoes pequenas e
# independentes -- pelo TCL, essa soma tende a uma Gaussiana mesmo que
# cada parcela nao seja Gaussiana. Ilustramos somando $K$ variaveis i.i.d.
# uniformes (claramente nao Gaussianas) e observando a forma da
# distribuicao da soma normalizada para $K$ crescente.

# %%
Kvals = [1, 2, 4, 20]
Mreal2 = 100000

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
fig.canvas.manager.set_window_title('Teorema Central do Limite')
for idx, K in enumerate(Kvals):
    U = rng.random((K, Mreal2)) - 0.5             # uniforme em (-0.5,0.5), media 0
    S = U.sum(axis=0) / np.sqrt(K / 12)            # normalizada para variancia 1

    ax = axes.flat[idx]
    ax.hist(S, bins=80, density=True)
    xx = np.linspace(-4, 4, 200)
    gausspdf = np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi)   # pdf N(0,1), sem toolbox
    ax.plot(xx, gausspdf, 'r-', linewidth=1.5)
    ax.set_title(f'Soma de K={K} uniformes (normalizada)')
    ax.set_xlim(-4, 4)

fig.suptitle('TCL: soma de v.a. uniformes converge para forma Gaussiana')
fig.tight_layout()
plt.show()
