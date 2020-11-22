import numpy as np
import matplotlib.pyplot as plt


class Optimization2D():

    def __init__(self):
        self._path = []
        self.tolerance = 0.0001

    def objective_function(self, a, b):
        return 100*(b - a**2)**2 + (1 - a)**2

    def partial_derivative(self, a, b):
        return np.array([400*(a**3) - 400*a*b + 2*a - 2, 200*(b-a**2)])

    def set_initial_point(self, point):
        self._initial_point = np.array(point)

    def set_tolerance(self, error):
        self.tolerance = error

    def _add_new_point(self, new_point):
        self._path = np.vstack((self._path, [new_point]))

    def plot_convergence_graph(self, contour=True,
                               save_fig=False, fig_name="img.png"):

        fig, ax = plt.subplots()

        ax.plot(self._path[:, 0], self._path[:, 1], color="red")

        if contour:
            min1 = float(self._path[:, 0].min())
            max1 = float(self._path[:, 0].max())

            start1 = min1 - 0.1
            end1 = max1 + 0.1

            min2 = float(self._path[:, 1].min())
            max2 = float(self._path[:, 1].max())

            start2 = min2 - 0.1
            end2 = max2 + 0.1

            # start1 = 0.5
            # start2 = 0.5
            # end1 = 1.5
            # end2 = 1.5

            x = np.linspace(start1, end1, 1000)
            y = np.linspace(start2, end2, 1000)

            X, Y = np.meshgrid(x, y)
            Z = 100*(Y - X**2)**2 + (1 - X)**2
            ax.contour(X, Y, Z, cmap="Blues", levels=12)

        if save_fig:
            plt.savefig(fig_name)

        plt.show()


class GradientDescent(Optimization2D):
    def __init__(self):
        self.armijo = 0.0001
        self.wolfe = 0.9
        self.rho = 0.8
        self.alpha0 = 1
        self.alpha = 0
        self.tolerance = 0.00000001

    def optimize(self):
        self._path = np.array([self._initial_point])
        while True:
            norm = np.linalg.norm(self.partial_derivative(self._path[-1, 0],
                                                          self._path[-1, 1]))

            if norm <= self.tolerance:
                print(self._path[-1])
                break

            self._calc_direction()
            self._calc_alpha()
            self._update_solution()

    def _calc_direction(self):
        self.d = -self.partial_derivative(self._path[-1, 0],
                                          self._path[-1, 1])

    def _calc_alpha(self):
        self.alpha = self.alpha0
        while True:
            if self.ArmijoRule(self.alpha):
                return
            self.alpha *= self.rho

    def ArmijoRule(self, a):
        return (self.objective_function(self._path[-1, 0] + a*self.d[0],
                                        self._path[-1, 1] + a*self.d[1])
                <= self.objective_function(self._path[-1, 0],
                                           self._path[-1, 1])
                - self.armijo*a*np.inner(self.d, self.d))

    def _update_solution(self):
        self._add_new_point(np.array(self._path[-1] + self.alpha*self.d))

    def steps(self):
        return self._path.shape[0]


class NewtonsMethod(GradientDescent):
    def Hessian(self, a, b):
        return np.array([[1200*(a**2)-400*b+2, -400*a], [-400*a, 200]])

    def _calc_direction(self):
        self.d = np.linalg.solve(
            self.Hessian(self._path[-1, 0],
                         self._path[-1, 1]),
            -self.partial_derivative(self._path[-1, 0],
                                     self._path[-1, 1]))

    def _update_solution(self):
        self._add_new_point(np.array(self._path[-1] + self.d))

    def optimize(self):
        self._path = np.array([self._initial_point])
        while True:
            norm = np.linalg.norm(self.partial_derivative(self._path[-1, 0],
                                                          self._path[-1, 1]))

            if norm <= self.tolerance:
                print(self._path[-1])
                break

            self._calc_direction()
            self._update_solution()


class quasiNewtonsMethod(NewtonsMethod):

    def optimize(self):
        self._B = np.array([[1, 0], [0, 1]])
        self._path = np.array([self._initial_point])
        while True:
            norm = np.linalg.norm(self.partial_derivative(self._path[-1, 0],
                                                          self._path[-1, 1]))

            if norm <= self.tolerance:
                print(self._path[-1])
                break

            self._calc_direction()
            self._calc_alpha()
            self._update_solution()
            self._update_matrix

    def _calc_direction(self):
        self.d = np.linalg.solve(
            self._B, -self.partial_derivative(self._path[-1, 0],
                                              self._path[-1, 1]))

    def _update_solution(self):
        self._add_new_point(np.array(self._path[-1] + self.alpha*self.d))

    def _update_matrix(self):
        # BFGS
        self._B = (self._B - np.dot(self._B * self._s, (self._B * self._s).T)
                   / (np.inner(self._s, self._B * self._s))
                   + (self._y * (self._y).T) / (self._s).T * self._y)

    def _s(self):
        return self._path[-1] - self._path[-2]

    def _y(self):
        return self.partial_derivative(self._path[-1, 0], self._path[-1, 1])\
            - self.partial_derivative(self._path[-2, 0], self._path[-2, 1])
