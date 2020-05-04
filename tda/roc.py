import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def area(y_test, y_scores, pos_label=None, title=None, plot_auc=True):
    fpr, tpr, threshold = roc_curve(y_test, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    if plot_auc:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(title + 'ROC Curve')
        plt.show()

    return roc_auc
