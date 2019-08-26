import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


class SimuResorte():

    def __init__(self, m, b, k, x0, v0, t_tot, step_sim, largo_resorte_eq, f_ext=None):
        self.m = m
        self.b = b
        self.k = k
        self.y0 = [x0, v0]
        self.t_tot = t_tot
        self.step_sim = step_sim

        if f_ext is None:
            self._f_ext = lambda t: 0
        else:
            self._f_ext = f_ext

        W = 400
        H = 800
        self.res_top = [W//2, H // 50]
        res_l0 = 300
        self.largo_resorte_eq = largo_resorte_eq
        self.x_scale_factor = float(res_l0) / largo_resorte_eq

        self.res_npoints = 30
        self.ancho_caja = 70

        self.canvas = 255 * np.ones((H, W, 3), dtype=np.uint8)
        cv2.rectangle(self.canvas, (W // 10, 0), (int(.9 * W), self.res_top[1]), (100, 100, 100), thickness=cv2.FILLED)

        dotted_line = []
        pos_eq = self.res_top[1] + res_l0 + self.ancho_caja // 2
        line_points = np.int32(np.linspace(0, W, 50))
        k = 0
        while k < len(line_points)-1:
            p1 = line_points[k]
            p2 = line_points[k + 1]
            dotted_line.append(np.array([[p1, pos_eq], [p2, pos_eq]]))
            k += 2

        cv2.polylines(self.canvas, dotted_line, False, (120, 120, 120), thickness=1)

    def _draw_res(self, canvas, xk, fk):
        res_l = (self.largo_resorte_eq + xk) * self.x_scale_factor

        points = []
        points.append(self.res_top)
        h = res_l / (self.res_npoints+3)

        d = self.ancho_caja // 3
        points.append([self.res_top[0], self.res_top[1] + h])
        for k in range(self.res_npoints):
            d = -d
            points.append([self.res_top[0] + d, points[-1][1] + h])
        points.append([self.res_top[0], points[-1][1] + h])
        points.append([self.res_top[0], points[-1][1] + h])

        points = np.array(points, dtype=np.int)
        cv2.polylines(canvas, [points], False, (170, 170, 170), thickness=3)

        cv2.rectangle(canvas, (points[-1][0] - self.ancho_caja // 2, points[-1][1]),
                      (points[-1][0] + self.ancho_caja // 2, points[-1][1] + self.ancho_caja),
                      (200, 100, 100), thickness=cv2.FILLED)

        if fk != 0:
            centro_caja = (points[-1][0], points[-1][1] + self.ancho_caja // 2)
            fin_flecha = (points[-1][0], points[-1][1] + self.ancho_caja // 2 + int(fk))
            cv2.line(canvas, centro_caja, fin_flecha, (0, 0, 255), thickness=3)

            if fk > 0:
                cv2.drawMarker(canvas, fin_flecha, (0, 0, 255), cv2.MARKER_TRIANGLE_DOWN, thickness=3, markerSize=10)
            else:
                cv2.drawMarker(canvas, fin_flecha, (0, 0, 255), cv2.MARKER_TRIANGLE_UP, thickness=3, markerSize=10)

        return canvas

    def _dy_dt(self, t, y):
        # y = [x, v]
        dy = np.array([
            y[1],  # derivada de la posición
            (-self.b * y[1] - self.k * y[0] + self._f_ext(t)) / self.m   # derivada de la velocidad
        ])
        return dy

    def _animate_one_cycle(self, t, x, f, k):
        drawn_canvas = self._draw_res(self.canvas.copy(), x[k], f[k] * 8)

        cv2.imshow("Sistema masa - resorte", drawn_canvas)

        plt.cla()
        plt.plot([0., self.t_tot], [0., 0.], 'gray')
        plt.plot(t, x, 'blue')
        plt.plot(t[k], x[k], 'ro', markersize=5)
        plt.xlim([0., self.t_tot], )
        plt.xlabel('t [s]', fontsize=12)
        plt.ylabel('x [m]', fontsize=12)
        plt.show()

        # time.sleep(.1)
        plt.tight_layout()
        plt.pause(.1)

    def simulate(self):
        n_steps = 1500
        t = np.linspace(0, self.t_tot, n_steps)
        y_sol = integrate.solve_ivp(self._dy_dt, [0., self.t_tot], self.y0, t_eval=t)['y']
        x = y_sol[0]
        f = np.array([self._f_ext(t_i) for t_i in t])

        plt.ion()
        fig = plt.figure(1, figsize=[6, 3.5])
        fig.canvas.set_window_title('Desplazamiento vs tiempo')

        for k in range(0, len(t), self.step_sim):
            self._animate_one_cycle(t, x, f, k)

            # Si esto se cumple el sistema "se rompió" (como en el caso de la resonancia)
            if self.largo_resorte_eq + x[k] <= 0:
                print("Sistema destruido :( [http://www.nooooooooooooooo.com/]")
                break
        else:
            # esta línea se va a ejectutar si el for finaliza normalmente (sin ningún break)
            self._animate_one_cycle(t, x, f, len(t)-1)

        cv2.waitKey(0)
