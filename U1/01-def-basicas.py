# %% [markdown]
# # Processos Aleatorios em Tempo Discreto -- Parte 1: Definicoes Basicas
#
# Este script acompanha o Capitulo 3 de Hayes, *Statistical Digital Signal
# Processing and Modeling*, e ilustra numericamente os conceitos dos slides
# `processos_aleatorios_cap3.tex`.
#
# Convertido do MATLAB Live Script (`.mlx`) para Python. Use um editor com
# suporte a celulas Jupytext (VS Code, Spyder, PyCharm) para rodar como
# notebook interativo, ou execute como script comum.

# %% [markdown]
# ## O que e um processo aleatorio em tempo discreto?
#
# Uma variavel aleatoria mapeia o espaco amostral $\Omega$ nos numeros
# reais. Um **processo aleatorio em tempo discreto** mapeia $\Omega$ no
# conjunto de sinais em tempo discreto $x(n)$: a cada resultado
# experimental $\omega_i$ associamos um sinal completo $x_i(n)$.
#
# Ha duas visoes equivalentes:
#
# - Para cada $\omega$, obtemos um sinal completo $x(n)$ (visao funcional
#   -- cada linha do ensemble abaixo).
# - Para um instante fixo $n=n_0$, o valor $x(n_0)$ e uma variavel
#   aleatoria definida em $\Omega$ (visao de coluna do ensemble).

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# %% [markdown]
# ## Exemplo: processo de Bernoulli
#
# Lancamos repetidamente uma moeda honesta:
#
# $$x(n) = \begin{cases} +1 & \text{cara no instante } n \\ -1 & \text{coroa no instante } n \end{cases}$$
#
# Se os lancamentos sao independentes, $x(n)$ e chamado processo de
# Bernoulli. Geramos varias realizacoes (sinais completos) do mesmo
# processo.

# %%
rng = np.random.default_rng(0)         # reprodutibilidade (rng(0) no MATLAB)
N = 40                                  # numero de amostras por realizacao
M = 6                                   # numero de realizacoes (ensemble)

X = 2 * (rng.random((M, N)) > 0.5) - 1  # cada linha = uma realizacao x_i(n)
n = np.arange(N)

fig, axes = plt.subplots(M, 1, figsize=(8, 10), sharex=True)
fig.canvas.manager.set_window_title('Processo de Bernoulli - Ensemble')
for i in range(M):
    ax = axes[i]
    ax.stem(n, X[i, :])
    ax.set_ylim(-1.5, 1.5)
    ax.set_ylabel(f'$x_{i+1}(n)$')
    if i == 0:
        ax.set_title('Ensemble de realizacoes do processo de Bernoulli')
axes[-1].set_xlabel('n')
fig.tight_layout()
plt.show()

# %% [markdown]
# Visao funcional: cada subplot acima e uma realizacao completa,
# correspondente a um $\omega_i$ fixo (uma sequencia inteira de
# lancamentos).
#
# Visao de coluna: fixando $n=n_0$, os valores $x_i(n_0)$ ao longo do
# ensemble formam uma variavel aleatoria. Visualizamos isso para
# $n_0=10$, contando quantas realizacoes deram +1 e quantas deram -1
# naquele instante especifico:

# %%
n0 = 10
fig = plt.figure()
fig.canvas.manager.set_window_title('Vista de coluna (n fixo)')
plt.hist(X[:, n0], bins=[-1.5, 0, 1.5])
plt.xlabel(f'$x(n_0={n0})$')
plt.ylabel('contagem no ensemble')
plt.title(f'Distribuicao de x(n_0={n0}) ao longo do ensemble')
plt.show()

# %% [markdown]
# ## Exemplo: processo harmonico com amplitude aleatoria
#
# Rolamos um dado honesto, atribuindo o resultado a $A\in\{1,\ldots,6\}$,
# e definimos
#
# $$x(n) = A\cos(n\omega_0)$$
#
# gerando um ensemble de 6 sinais distintos, igualmente provaveis (um
# para cada face do dado).

# %%
omega0 = np.pi / 8
A_vals = np.arange(1, 7)

fig = plt.figure()
fig.canvas.manager.set_window_title('Processo harmonico - amplitude aleatoria')
colors = plt.cm.tab10(np.linspace(0, 1, 6))  # equivalente a lines(6) do MATLAB
for k in range(6):
    plt.plot(n, A_vals[k] * np.cos(n * omega0),
              color=colors[k], label=f'A={k+1}')
plt.legend()
plt.xlabel('n')
plt.ylabel(r'$x(n) = A \cos(n\omega_0)$')
plt.title('Ensemble de 6 sinais igualmente provaveis (um por face do dado)')
plt.show()

# %% [markdown]
# ## Transformando processos: filtragem
#
# Um processo pode ser transformado em outro por meio de uma operacao
# matematica. Um caso importante e a filtragem linear. Filtramos o
# processo de Bernoulli com o filtro recursivo de primeira ordem
#
# $$y(n) = 0.5\,y(n-1) + x(n)$$
#
# Isso gera um novo processo $y(n)$, correlacionado no tempo, a partir de
# um processo $x(n)$ originalmente nao correlacionado amostra a amostra.

# %%
x = X[0, :]                          # uma realizacao do Bernoulli
y = lfilter([1], [1, -0.5], x)       # y(n) = 0.5 y(n-1) + x(n)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
fig.canvas.manager.set_window_title('Filtragem: Bernoulli -> processo correlacionado')
ax1.stem(n, x)
ax1.set_ylabel('x(n)')
ax1.set_title('x(n): processo de Bernoulli (entrada)')
ax2.stem(n, y)
ax2.set_ylabel('y(n)')
ax2.set_xlabel('n')
ax2.set_title('y(n) = 0.5 y(n-1) + x(n) (saida filtrada)')
fig.tight_layout()
plt.show()

# %% [markdown]
# Note como $y(n)$ exibe uma estrutura muito mais "suave" -- amostras
# vizinhas ficam parecidas -- enquanto $x(n)$ salta de forma independente
# a cada instante. Essa correlacao temporal introduzida pela filtragem
# sera quantificada na Parte 2 (medias de conjunto) e na Parte 8
# (espectro de potencia).
