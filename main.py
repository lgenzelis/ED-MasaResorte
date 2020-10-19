import numpy as np

from simu_resorte import SimuResorte

# Fuerza externa a tiempo t
def f_ext_ejemplo(t):
    if t < 2 * np.pi:
       _f = 0
    else:
       _f = 257/32. * np.cos(4 * t)
    return _f

# Constantes del sistema
m = 2.  # masa
b = 2 # constante de amortiguamiento
k = 65/2.  # constante del resorte

# Condiciones iniciales
x0 = .5  # desplazamiento inicial
v0 = -9/4.  # velocidad inicial

# Tiempo simulado [s]
t_tot = 20.

# Parámetros para la animación
step_sim = 10 # a mayor valor, más rápido se va a reproducir la animación
largo_resorte_eq = 2. # distancia desde la base del resorte hasta la posición de equilibrio

if __name__ == '__main__':
    simuResorte = SimuResorte(m, b, k, x0, v0, t_tot, step_sim, largo_resorte_eq, f_ext_ejemplo)
    simuResorte.simulate()
