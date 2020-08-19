from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random


plt.style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def data_set(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        print(i)
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


# as we lno y =mx +b here we are calculating m, b
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - (m * mean(xs))
    return m, b


def squared_error(ys_og, ys_line):
    return sum((ys_og - ys_line) ** 2)


def coefficient_determination(ys_og, ys_line):
    y_mean_line = [mean(ys_og) for y in ys_og]
    sq_error_reression = squared_error(ys_og, ys_line)
    sq_error_y_mean = squared_error(ys_og, y_mean_line)
    return 1 - (sq_error_reression / sq_error_y_mean)


xs, ys = data_set(40, 10, correlation='pos')


m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [((m * x) + b) for x in xs]

r_squared = coefficient_determination(ys, regression_line)
print(r_squared)

# # scattering the xs and ys
plt.scatter(xs, ys)
# # plotting our linear regression line
plt.plot(xs, regression_line)
plt.show()
