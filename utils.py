"""Utilities librairies."""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def print_metrics(labels, preds):
    print("Precision Score: {}".format(precision_score(labels, preds)))
    print("Recall Score: {}".format(recall_score(labels, preds)))
    print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
    print("F1 Score: {}".format(f1_score(labels, preds)))
    print("ROC_AUC_Score: {}".format(roc_auc_score(labels, preds)))


def print_confusion_matrix(y_true, y_pred, class_names=[0, 1]):
    cnf_matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))

    # create heatmap
    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.xticks(tick_marks + .5, class_names)
    plt.yticks(tick_marks + .5, class_names)


def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    best_k = best_score = 0

    for k in range(min_k, max_k + 1, 2):
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)

        y_preds = knn_classifier.predict(X_test)
        k_f1_score = f1_score(y_test, y_preds)

        if k_f1_score > best_score:
            best_score = k_f1_score
            best_k = k

    best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
    best_knn_classifier.fit(X_train, y_train)

    y_preds = best_knn_classifier.predict(X_test)
    print_metrics(y_test, y_preds)
    print('Best Value for k:', best_k)
    print('F1-Score:', best_score)
    return None
