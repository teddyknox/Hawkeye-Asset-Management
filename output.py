from numpy import *
from sklearn.metrics import *
# from pylab import *
import matplotlib.pyplot as plt


def graph_data(labs, preds, model="A Model"):

    print "*" * 25
    print "Printing Output for " + model
    pos_acc = accuracy_score(labs, preds)
    neg_acc = accuracy_score(labs, preds)
    avg_acc = accuracy_score(labs, preds)
    pos_prec = precision_score(labs, preds, labels=[-1, 1], pos_label=1)
    neg_prec = precision_score(labs, preds, labels=[-1, 1], pos_label=-1)
    avg_prec = precision_score(labs, preds, labels=[-1, 1], pos_label=None, average="weighted")
    pos_rec = recall_score(labs, preds, labels=[-1, 1], pos_label=1)
    neg_rec = recall_score(labs, preds, labels=[-1, 1], pos_label=-1)
    avg_rec = recall_score(labs, preds, labels=[-1, 1], pos_label=None, average="weighted")
    pos_f1 = f1_score(labs, preds, labels=[-1, 1], pos_label=1)
    neg_f1 = f1_score(labs, preds, labels=[-1, 1], pos_label=-1)
    avg_f1 = f1_score(labs, preds, labels=[-1, 1], pos_label=None, average="weighted")

    output = '\\begin{tabular}{c | c c c c}\n'
    output += "\\textbf{%s}\t& Accuracy\t& Precision\t& Recall\t& F1 Score\t\\\\\n" % (name)
    output += "\\hline \n"
    output += "Negative\t\t& %.3f\t\t& %.3f\t\t& %.3f\t\t& %.3f\t\t\\\\\n" % (neg_acc, neg_prec, neg_rec, neg_f1)
    output += "Positive\t\t& %.3f\t\t& %.3f\t\t& %.3f\t\t& %.3f\t\t\\\\\n" % (pos_acc, pos_prec, pos_rec, pos_f1)
    output += "Average \t\t& %.3f\t\t& %.3f\t\t& %.3f\t\t& %.3f\t\t\\\\\n" % (pos_acc, avg_prec, avg_rec, avg_f1)
    output += "\\end{tabular}"
    print output

    self.pos_acc.append(pos_acc)
    self.neg_acc.append(neg_acc)
    self.avg_acc.append(avg_acc)
    self.pos_prec.append(pos_prec)
    self.neg_prec.append(neg_prec)
    self.avg_prec.append(avg_prec)
    self.pos_rec.append(pos_rec)
    self.neg_rec.append(neg_rec)
    self.avg_rec.append(avg_rec)
    self.pos_f1.append(pos_f1)
    self.neg_f1.append(neg_f1)
    self.avg_f1.append(avg_f1)



