# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import pandas as pd
# from scipy import interpolate

def plot_3d_runtime():
    csv_file = '../vis/runtime/3d/3d.csv'
    data = pd.read_csv(csv_file, comment='#')

    all_nodes = data['skewedphynodes'].unique()
    all_nodes.sort()
    # print(all_nodes)

    all_x = []
    all_y = []
    all_z = []

    for nodes in all_nodes:
        cur_data = data[data['skewedphynodes'] == nodes]
        cur_data = cur_data.groupby('skewpct').agg({
            'runtime': ['mean']
        })
        cur_data['skewpct'] = cur_data.index

        x_skewpct = cur_data['skewpct']
        y_numnodes = [nodes] * len(x_skewpct)
        z_runtime = cur_data['runtime']['mean']

        x_skewpct = x_skewpct.map(lambda x: x*10-100)

        # print(x_skewpct)
        # print(y_numnodes)
        # print(z_runtime)
        #
        all_x.append(x_skewpct)
        all_y.append(y_numnodes)
        all_z.append(z_runtime)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    npconv = lambda x: np.array([np.array(i) for i in x])
    # npconv = lambda x: np.array(np.matrix(x))

    X = np.arange(-5, 5, 0.25)
    Y =np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # X = np.array(all_x)
    # Y = np.array(all_y)
    # Z = np.array(all_z)

    X = npconv(all_x)
    Y = npconv(all_y)
    Z = npconv(all_z)

    ax.set_xlabel('Skew %')
    ax.set_ylabel('# skewed nodes')
    ax.set_zlabel('Runtime (s)')
    ax.set_title('Skew vs Runtime')


    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.view_init(40, 130)
    # plt.show()
    plt.savefig('../vis/runtime/3d/3dv3.pdf')

if __name__ == '__main__':
    plot_3d_runtime()
