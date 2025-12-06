import torch
from typing import Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

class metrics:
    def __init__(self):
        self.preds = []
        self.labels = []

    def update(self, labels: Union[torch.Tensor, list], preds: Union[torch.Tensor, list]):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy().tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy().tolist()
        self.preds.extend(preds)
        self.labels.extend(labels)
    
    def compute(self):
        accuracy = accuracy_score(self.labels, self.preds)
        precision = precision_score(self.labels, self.preds)
        recall = recall_score(self.labels, self.preds)
        f1 = f1_score(self.labels, self.preds)
        return accuracy, precision, recall, f1
    
    def reset(self):
        self.preds = []
        self.labels = []

def plot_roc(labels, pobs):

    fpr, tpr, _ = roc_curve(labels, pobs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.show()

def plot_scores(sig_scores, bkg_scores, bins=50, range=(0,1), logscale=False):
    fig, ax = plt.subplots()
    ax.hist(sig_scores, bins=bins, range=range, density=True, alpha=0.5, label='Signal', histtype='step', linewidth=1.5)
    ax.hist(bkg_scores, bins=bins, range=range, density=True, alpha=0.5, label='Background', histtype='step', linewidth=1.5)
    ax.set_xlabel('Classifier Score')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log' if logscale else 'linear')
    ax.set_title('Classifier Score Distribution')
    ax.legend(loc='upper center')
    ax.grid(True)
    plt.show()