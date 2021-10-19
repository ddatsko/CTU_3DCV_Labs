import numpy as np
import matplotlib.pyplot as plt
import sys
from RANSAC import ransac, ransac_plus_line_fit, mlesac, mlesac_plus_line_fit


def plot_line(line, color: str = 'blue'):
    ax = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
    plt.plot(ax, list(map(lambda x: (-line[2] - line[0] * x) / line[1], ax)), color=color)


def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'linefit_1.txt'

    x = np.loadtxt(input_file).T

    plt.scatter(x[0], x[1], color='black', marker='.')

    fitted_line = np.polyfit(x[0], x[1], 1)
    plot_line(np.array([-10, 3, 1200]), color='black')
    plot_line(np.array([-fitted_line[0], 1, -fitted_line[1]]), color='green')
    plot_line(ransac(x, 7))
    plot_line(ransac_plus_line_fit(x, 7), color='red')
    plot_line(mlesac(x, 7), color='cyan')
    plot_line(mlesac_plus_line_fit(x, 7), color='yellow')

    # plt.plot(ransac_line[0], ransac_line[1])
    # plt.lin
    plt.axis([-30, 470, -20, 350]);
    plt.show()


if __name__ == "__main__":
    main()
