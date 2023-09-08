from numpy import mean, std

def print_cross_validation_scores(scores):
    print('MACRO:')
    print('Precision: %.4f (%.4f)' % (mean(scores['test_precision_macro']), std(scores['test_precision_macro'])))
    print('Recall: %.4f (%.4f)' % (mean(scores['test_recall_macro']), std(scores['test_recall_macro'])))
    print('F1 score: %.4f (%.4f)' % (mean(scores['test_f1_macro']), std(scores['test_f1_macro'])))
    print('---------------------------------')
    print('MICRO:')
    print('Precision: %.4f (%.4f)' % (mean(scores['test_precision_micro']), std(scores['test_precision_micro'])))
    print('Recall: %.4f (%.4f)' % (mean(scores['test_recall_micro']), std(scores['test_recall_micro'])))
    print('F1 score: %.4f (%.4f)' % (mean(scores['test_f1_micro']), std(scores['test_f1_micro'])))