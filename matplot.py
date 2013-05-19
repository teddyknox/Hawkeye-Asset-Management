# a bar plot with errorbars
from numpy import *
from pylab import *
import matplotlib.pyplot as plt

# rc('text',usetex=True)
# rc('font',**{'family':'serif','serif':['Computer Modern']})


def plot_chart(results, labels, name="figure"):
    N = len(results)

    ind = np.arange(N)
    width = 0.35

    fig = plt.figure(facecolor="#ffffff")
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind + (width / 4), results, width, color='#6AA8EB')

    # add some
    ax.set_ylabel('Accuracy (%)', family="serif")
    ax.set_title('Stock Classification Results by Model', family="serif")
    ax.set_xticks(ind + (width * 3 / 4))
    ax.set_xticklabels(labels, family="serif", fontname="Computer Modern")

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    1.05 * height,
                    '%d' % int(height),
                    ha='center',
                    va='bottom')

    savefig(name + '.png')

    plt.show()

plot_chart([0.345, 0.429, 0.429, 0.697, 0.765],
           ['Gaussian NB', 'Linear SVM', 'Linear SVC', 'Multinomial NB', 'Bernoulli NB'])
