import sys
sys.path.append('../')

from constant.score import CROSS_VALIDATION_SCORING

from sklearn.model_selection import cross_validate, KFold

def fold_cross_validate(classifier, X, y, n_splits=10):
    # Define the cross-validation procedure
    cv = KFold(n_splits=n_splits, shuffle=False)

    # Evaluate the model using cross-validation
    scores = cross_validate(
        classifier, X, y, scoring=CROSS_VALIDATION_SCORING, cv=cv, n_jobs=-1)

    return scores