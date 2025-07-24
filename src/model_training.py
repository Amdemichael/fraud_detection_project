import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, auc, classification_report


def train_logistic_regression(X_train, y_train, random_state=42, class_weight='balanced', max_iter=1000):
    """
    Train a Logistic Regression model.
    Returns the fitted model.
    """
    model = LogisticRegression(max_iter=max_iter, class_weight=class_weight, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42, scale_pos_weight=1):
    """
    Train an XGBoost classifier.
    Returns the fitted model.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true, y_pred, y_proba, average='binary', pos_label=1, plot_pr_curve=True, title=None):
    """
    Evaluate a model using confusion matrix, F1 score, AUC-PR, and optionally plot the PR curve.
    Returns a dictionary of metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, pos_label=pos_label)
    precision, recall, _ = precision_recall_curve(y_true, y_proba, pos_label=pos_label)
    auc_pr = auc(recall, precision)
    report = classification_report(y_true, y_pred)
    if plot_pr_curve:
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title or 'Precision-Recall Curve')
        plt.show()
    print('Confusion Matrix:')
    print(cm)
    print('F1 Score:', f1)
    print('AUC-PR:', auc_pr)
    print('Classification Report:')
    print(report)
    return {'confusion_matrix': cm, 'f1': f1, 'auc_pr': auc_pr, 'classification_report': report}
