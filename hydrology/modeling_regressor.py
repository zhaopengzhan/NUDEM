import numpy as np
from scipy.optimize import curve_fit


class AutoRegressor:
    def __init__(self, x, y, fit_type):

        self.x = x
        self.y = y
        self.fit_type = fit_type

        if fit_type == "linear":
            self._f_xy = np.poly1d(np.polyfit(x, y, 1))
            self._f_yx = np.poly1d(np.polyfit(y, x, 1))

        elif fit_type == "poly2":
            self._f_xy = np.poly1d(np.polyfit(x, y, 2))
            self._f_yx = np.poly1d(np.polyfit(y, x, 2))

        elif fit_type == "poly3":
            self._f_xy = np.poly1d(np.polyfit(x, y, 3))
            self._f_yx = np.poly1d(np.polyfit(y, x, 3))

        elif fit_type == "poly4":
            self._f_xy = np.poly1d(np.polyfit(x, y, 4))
            self._f_yx = np.poly1d(np.polyfit(y, x, 4))

        elif fit_type == "poly5":
            self._f_xy = np.poly1d(np.polyfit(x, y, 5))
            self._f_yx = np.poly1d(np.polyfit(y, x, 5))

        elif fit_type == "exp":
            # y = a * exp(bx)
            def exp_func(x, a, b):
                return a * np.exp(b * x)

            popt, _ = curve_fit(exp_func, self.x, self.y, p0=(1.0, 0.1))
            a, b = popt
            self._f_xy = lambda t: a * np.exp(b * t)
            self._f_yx = lambda t: (np.log(t / a)) / b

        elif fit_type == "log":
            # y = a * ln(x) + b
            def log_func(x, a, b):
                return a * np.log(x) + b

            popt, _ = curve_fit(log_func, self.x, self.y, p0=(1.0, 1.0))
            a, b = popt
            self._f_xy = lambda t: a * np.log(t) + b
            self._f_yx = lambda t: np.exp((t - b) / a)

        elif fit_type == "power":
            # y = a * x^b
            def power_func(x, a, b):
                return a * np.power(x, b)

            popt, _ = curve_fit(power_func, self.x, self.y, p0=(1.0, 1.0))
            a, b = popt
            self._f_xy = lambda t: a * (t ** b)
            self._f_yx = lambda t: (t / a) ** (1.0 / b)

        else:
            raise ValueError(f"Unknown fit_type: {fit_type}")

    def get_y_from_x(self, x):
        return self._f_xy(np.asarray(x, dtype=float))

    def get_x_from_y(self, y):
        return self._f_yx(np.asarray(y, dtype=float))

    def show(self):
        from matplotlib import pyplot as plt
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["axes.unicode_minus"] = False

        xmin, xmax = float(self.x.min()), float(self.x.max())
        N_sample = 50
        x_sample = np.linspace(xmin, xmax, num=N_sample)

        y_pred = self.get_y_from_x(x_sample)

        # plot
        fig_width, fig_height = (11, 8)
        font_size = round(fig_height * 2.25)
        plt.rcParams.update({'font.size': font_size})

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        plt.plot(x_sample, y_pred, label=f"{self.fit_type} fit line", linewidth=font_size * 0.2, zorder=1)
        plt.scatter(self.x, self.y, color="red", s=font_size, marker="o", label="Input Data", zorder=2)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.legend()
        plt.grid(True, ls="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

