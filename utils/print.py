from numpy import mean, std
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_cross_validation_scores(
    scores, is_micro_required=True, is_macro_required=True
):
    if is_macro_required:
        print("MACRO:")
        print(
            "Precision: %.4f (%.4f)"
            % (
                mean(scores["test_precision_macro"]),
                std(scores["test_precision_macro"]),
            )
        )
        print(
            "Recall: %.4f (%.4f)"
            % (mean(scores["test_recall_macro"]), std(scores["test_recall_macro"]))
        )
        print(
            "F1 score: %.4f (%.4f)"
            % (mean(scores["test_f1_macro"]), std(scores["test_f1_macro"]))
        )
        print("---------------------------------")

    if is_micro_required:
        print("MICRO:")
        print(
            "Precision: %.4f (%.4f)"
            % (
                mean(scores["test_precision_micro"]),
                std(scores["test_precision_micro"]),
            )
        )
        print(
            "Recall: %.4f (%.4f)"
            % (mean(scores["test_recall_micro"]), std(scores["test_recall_micro"]))
        )
        print(
            "F1 score: %.4f (%.4f)"
            % (mean(scores["test_f1_micro"]), std(scores["test_f1_micro"]))
        )
        print("---------------------------------")


def print_scores(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
