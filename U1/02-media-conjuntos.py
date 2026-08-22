# %% [markdown]
# # Processos Aleatorios em Tempo Discreto -- Parte 2: Medias de Conjunto
#
# Continuacao do script 01. Aqui ilustramos as medias de conjunto (ensemble
# averages): media, variancia, autocovariancia, autocorrelacao e
# correlacao cruzada -- estimadas por Monte Carlo e comparadas com as
# expressoes teoricas dos slides.
#
# Convertido do MATLAB Live Script (`.mlx`) para Python. Use um editor com
# suporte a celulas Jupytext (VS Code, Spyder, PyCharm) para rodar como
# notebook interativo, ou execute como script comum.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Media e variancia do processo
#
# Como um processo e uma sequencia indexada de variaveis aleatorias,
# definimos, para cada instante $n$:
#
# $$m_x(n) = E\{x(n)\}, \qquad \sigma_x^2(n) = E\{|x(n)-m_x(n)|^2\}$$
#
# Essas sao estatisticas de primeira ordem: em geral dependem de $n$.

# %% [markdown]
# ## Autocovariancia e autocorrelacao
#
# Para capturar a relacao entre $x(k)$ e $x(l)$:
#
# $$c_x(k,l) = E\{[x(k)-m_x(k)][x(l)-m_x(l)]^{*}\}, \qquad r_x(k,l) = E\{x(k)x^{*}(l)\}$$
#
# Por convencao (como no livro-texto), trabalhamos com processos de media
# zero: nesse caso $c_x \equiv r_x$.

# %% [markdown]
# ## Exemplo 3.3.1 -- Processo harmonico com fase aleatoria
#
# Seja a senoide de fase aleatoria
#
# $$x(n) = A\sin(n\omega_0+\phi), \qquad \phi \sim \mathcal{U}(-\pi,\pi)$$
#
# Teoricamente: $m_x(n) = 0$ e
#
# $$r_x(k,l) = \frac{1}{2}A^2\cos[(k-l)\omega_0]$$
#
# Vamos estimar essas quantidades por Monte Carlo (muitas realizacoes,
# cada uma com uma fase $\phi$ diferente) e comparar com a teoria.

# %%
rng = np.random.default_rng(1)
A = 2
omega0 = np.pi / 6
N = 30                                   # instantes n = 0,...,N-1
Mreal = 20000                            # numero de realizacoes (tamanho do ensemble)

n = np.arange(N).reshape(-1, 1)                    # coluna N x 1
phi = rng.uniform(-np.pi, np.pi, size=(1, Mreal))   # fase uniforme em (-pi,pi), uma por realizacao
Xh = A * np.sin(n * omega0 + phi)                   # Xh[n, realizacao], broadcasting N x Mreal

m_hat = Xh.mean(axis=1)                  # media estimada em cada instante n

plt.figure()
plt.get_current_fig_manager().set_window_title('Media do processo harmonico')
plt.plot(n, m_hat, 'o-')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('n')
plt.ylabel('m_x(n) estimada')
plt.title(r'Media de conjunto (deve ser $\approx$ 0 para todo n)')
plt.legend(['estimativa Monte Carlo', 'teoria m_x(n)=0'])
plt.show()

# %% [markdown]
# Agora a autocorrelacao $r_x(k,l)$, fixando $l=0$ e variando $k$:

# %%
l0 = 0
r_hat = np.mean(Xh * Xh[l0, :], axis=1)             # r_x(k, l0) estimada
r_teo = 0.5 * A**2 * np.cos((n[:, 0] - l0) * omega0)  # r_x(k, l0) teorica

plt.figure()
plt.get_current_fig_manager().set_window_title('Autocorrelacao do processo harmonico')
plt.plot(n, r_hat, 'o', n, r_teo, '-')
plt.grid(True)
plt.xlabel('k')
plt.ylabel('r_x(k, l_0=0)')
plt.legend(['estimativa Monte Carlo', r'teoria: 0.5 $A^2$ cos((k-l_0)$\omega_0$)'])
plt.title('Autocorrelacao r_x(k,l_0) -- Exemplo 3.3.1')
plt.show()

erro_rel_331 = np.linalg.norm(r_hat - r_teo) / np.linalg.norm(r_teo)
print(f'Erro relativo (r_x estimado vs teoria, Exemplo 3.3.1) = {erro_rel_331:.4f}')

# %% [markdown]
# ## Exemplo 3.3.2 -- Correlacao cruzada
#
# Seja $y(n) = x(n-1)$. Teoricamente $r_{xy}(k,l) = r_x(k,l-1)$. Usamos a
# mesma realizacao em ensemble de $x(n)$ e formamos $y(n)$ deslocando no
# tempo.

# %%
x_neg1 = A * np.sin(-1 * omega0 + phi)     # amostra real em n=-1 (mesma fase phi de cada realizacao), shape (1, Mreal)
Yh = np.vstack([x_neg1, Xh[:-1, :]])       # y(n) = x(n-1), incluindo y(0)=x(-1) de verdade

k0 = 5
rxy_hat = np.mean(Xh[k0, :] * Yh, axis=1)              # r_xy(k0, l) para todo l no eixo l
rxy_teo = 0.5 * A**2 * np.cos((k0 - (n[:, 0] - 1)) * omega0)  # r_x(k0, l-1)

plt.figure()
plt.get_current_fig_manager().set_window_title('Correlacao cruzada x(n), y(n)=x(n-1)')
plt.plot(n, rxy_hat, 'o', n, rxy_teo, '-')
plt.grid(True)
plt.xlabel('l')
plt.ylabel('r_{xy}(k_0, l)')
plt.legend(['estimativa Monte Carlo', 'teoria: r_x(k_0, l-1)'])
plt.title(f'Correlacao cruzada r_xy(k_0={k0}, l)')
plt.show()

erro_rel_332 = np.linalg.norm(rxy_hat - rxy_teo) / np.linalg.norm(rxy_teo)
print(f'Erro relativo (r_xy estimado vs teoria, Exemplo 3.3.2) = {erro_rel_332:.4f}')

# %% [markdown]
# ## Propriedade: soma de processos nao correlacionados
#
# Modelo de ruido aditivo $y(n) = x(n) + w(n)$, com $w(n)$ ruido de media
# zero, nao correlacionado com $x(n)$. Nesse caso $r_y(k,l) = r_x(k,l) +
# r_w(k,l)$. Verificamos numericamente com $x(n)$ o harmonico acima e
# $w(n)$ ruido branco Gaussiano independente.

# %%
sigma_w2 = 1.5
Wh = np.sqrt(sigma_w2) * rng.standard_normal((N, Mreal))
Yh2 = Xh + Wh

l0 = 0
ry_hat = np.mean(Yh2 * Yh2[l0, :], axis=1)
rx_hat = np.mean(Xh * Xh[l0, :], axis=1)
rw_hat = np.mean(Wh * Wh[l0, :], axis=1)

plt.figure()
plt.get_current_fig_manager().set_window_title('Soma de processos nao correlacionados')
plt.plot(n, ry_hat, 'o-', n, rx_hat + rw_hat, 'r--')
plt.grid(True)
plt.xlabel('k')
plt.ylabel('r_y(k,0)')
plt.legend(['r_y estimada diretamente', 'r_x + r_w (estimadas separadamente)'])
plt.title('r_y(k,l) = r_x(k,l) + r_w(k,l) quando x e w sao nao correlacionados')
plt.show()
