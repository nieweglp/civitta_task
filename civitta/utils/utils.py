from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def evaluate_print_metrics(y_test, preds):
    """"""
    print(classification_report(y_test, preds))
    print("confusion matrix")
    print(confusion_matrix(y_test, preds))
    print(f"roc_auc_score: {roc_auc_score(y_test, preds)}")

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.title("Confusion matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def evaluate_ks(y_real, y_proba):
    df = pd.DataFrame()
    df['real'] = y_real
    df['proba'] = y_proba[:, 1]
    
    not_default = df[df['real'] == 0]
    default = df[df['real'] == 1]
    
    ks = ks_2samp(not_default['proba'], default['proba'])
    
    print(f"KS: {ks.statistic:.4f} (p-value: {ks.pvalue:.3e})")
    return {"ks_statistic": ks.statistic}

def find_cut_off_point(clf, X_test, y_test):
    """"""
    preds = clf.predict_proba(X_test)
    _, _, thresholds = roc_curve(y_test, preds[:,1])
    
    f1_list = []
    for thres in thresholds:
        y_pred = np.where(preds[:,1] > thres, 1, 0)
        f1 = f1_score(y_test, y_pred)
        f1_list.append({"f1_test": f1, "threshold": thres})
        
    df_f1_cut_off = pd.DataFrame(f1_list)
    cut_off_max_f1 = df_f1_cut_off["f1_test"].max()
    return df_f1_cut_off[df_f1_cut_off["f1_test"] == cut_off_max_f1]


def apply_optimize_threshold(preds, threshold):
    """"""
    return np.where(preds[:, 1] >= threshold, 1, 0)


def display_feature_importance(clf):
    """Displays only 10 most important features"""
    df_importance = pd.DataFrame(
        {"feature": clf.feature_names_in_, 
        "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False).iloc[:10, :]
    sns.barplot(data=df_importance, x="importance", y="feature")
    plt.title(f"Feature importance for \n{clf}")
    plt.grid()
    plt.show()

def plot_roc_curve(fpr, tpr, name):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.grid(visible=True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")