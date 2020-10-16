import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go


def area(y_test, y_scores, pos_label=None, title=None, plot_roc=True):
    fpr, tpr, threshold = roc_curve(y_test, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    if plot_roc:
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


def roc_auc(y_true, y_score, pos_label=None, plot_roc=True):
    fpr, tpr, threshold = roc_curve(y_true=y_true, y_score=y_score, pos_label=pos_label)
    area_under_roc = auc(fpr, tpr)
    if plot_roc:
        trace1 = go.Scatter(x=fpr, y=tpr, name='AUC = %0.3f' % area_under_roc, mode='lines')
        trace2 = go.Scatter(x=[0, 1], y=[0, 1], name='Reference',
                            line={"dash": "dash", "color": "rgba(255, 0, 0, 255)", "width": 4})
        layout = {
            "title": "ROC Curve",
            "xaxis": {"title": "False Positive Rate"},
            "yaxis": {"title": "True Positive Rate"}
        }
        # fig = go.Figure(data=[trace1, trace2], layout=layout)
        fig = go.Figure(layout=layout)
        fig.add_trace(trace=trace1)
        fig.add_trace(trace=trace2)
        # fig.update(layout=dict(title=dict(x=0.5)))  # title 居中
        fig.show()

    return area_under_roc


if __name__ == "__main__":
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_scores = [2, 3, 2, 2, 3, 1.8, 2, 2.2, 2.3, 1.9]
    roc_auc(y_true=y_true, y_score=y_scores)
