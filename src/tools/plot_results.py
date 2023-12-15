import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_csv_results(paths, titles=None, labels=None, opts=None, logs=None):
    dfs = []
    alg = []
    linestyles = ['-', '--', '-.', ':']
    for path in paths:
        print('open {}.'.format(path))
        df = pd.read_csv(path)
        dfs.append(df)
        alg.append(re.search('-.+?-', path).group().replace('-', '').upper())

    for column in dfs[0].columns[1:]:
        fig, ax = plt.subplots()

        for i in range(len(paths)):
            df = dfs[i]
            linestyle = linestyles[i % len(linestyles)]
            x = df['iter']
            y = df[column]
            if opts is not None and column in opts.keys():
                y = y - opts[column]
            if logs is not None and column in logs.keys() and logs[column]:
                y = np.abs(y)
                ax.set_yscale('log')
            ax.plot(x, y, label=alg[i], linestyle=linestyle)

            if titles is not None and column in titles.keys():
                ax.set_title(titles[column])
            if labels is not None and column in labels.keys():
                ax.set_xlabel(labels[df.columns[0]])
                ax.set_ylabel(labels[column])
            else:
                ax.set_xlabel(df.columns[0])
                ax.set_ylabel(column)
        ax.grid()
        fig.legend()

        filename = paths[0].replace('.csv', '')
        filename = re.sub('-.+?-', '-', filename, count=1)
        filename = '{}-{}plot.png'.format(filename, column)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.show(bbox_inches='tight')
