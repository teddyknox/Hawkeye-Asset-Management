# a bar plot with errorbars
from numpy import *
from pylab import *
import matplotlib.pyplot as plt


def plot_chart(results, labels, name="figure", title=None, yaxis=None):

    N = len(results)

    ind = np.arange(N)
    width = 0.35

    fig = plt.figure(facecolor="#ffffff")
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind + (width / 4), results, width, color='#6AA8EB')

    # add some
    ax.set_ylabel(yaxis, family="serif")
    ax.set_title(title, family="serif")
    ax.set_xticks(ind + (width * 3 / 4))
    ax.set_xticklabels(labels, family="serif", fontname="Computer Modern")

    savefig(name + '.png')

    plt.show()

def plot_chart_3(results, results2, results3, labels, name="figure", title=None, yaxis=None):

    N = len(results)

    ind = np.arange(N)
    width = 0.30

    fig = plt.figure(facecolor="#ffffff")
    ax = fig.add_subplot(111)
    pos = ax.bar(ind, results, width, color='#6AA8EB')
    neg = ax.bar(ind + width, results2, width, color='r')
    avg = ax.bar(ind + 2 * width, results3, width, color='g')

    # add some
    ax.set_ylabel(yaxis, family="serif")
    ax.set_title(title, family="serif")
    ax.set_xticks(ind + 3 * width / 2)
    ax.set_xticklabels(labels, family="serif", fontname="Computer Modern")

    ax.legend((pos, neg, avg), ('Positive', 'Negative', 'Average'))

    savefig(name + '.png')

    plt.show()
