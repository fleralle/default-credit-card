"""Utilities librairies."""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


def print_metrics(labels, preds):
    print("Precision Score: {}".format(precision_score(labels, preds)))
    print("Recall Score: {}".format(recall_score(labels, preds)))
    print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
    print("F1 Score: {}".format(f1_score(labels, preds)))
    print("ROC_AUC_Score: {}".format(roc_auc_score(labels, preds)))


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
