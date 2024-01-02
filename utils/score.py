import sys

sys.path.append("../")

from constant.score import CROSS_VALIDATION_SCORING
from constant.columns import FEATURES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import confusion_matrix


def fold_cross_validate(classifier, X, y, n_splits=10):
    # Define the cross-validation procedure
    cv = KFold(n_splits=n_splits, shuffle=False)

    # Evaluate the model using cross-validation
    scores = cross_validate(
        classifier, X, y, scoring=CROSS_VALIDATION_SCORING, cv=cv, n_jobs=-1
    )

    return scores


def visualize_classification_result(y_test, y_pred):
    clf_matrix = confusion_matrix(y_test, y_pred)

    # Assuming clf_matrix is the confusion matrix you've computed
    clf_matrix_df = pd.DataFrame(
        clf_matrix, index=["s:ok", "s:failed"], columns=["s:ok", "s:failed"]
    )

    # Create heatmap
    class_names = ["s:ok", "s:failed"]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(clf_matrix_df, annot=True, cmap="YlGnBu", fmt="g")
    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.title("Confusion matrix decision tree", y=1.1)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")


def visualize_feature_importance(importance_scores, feature_cols=FEATURES):
    # # Normalize the importances
    # normalized_importance = importance_scores / np.sum(importance_scores)

    # Plot the normalized importances
    plt.figure(figsize=(10, 5))
    plt.bar(feature_cols, importance_scores)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()
