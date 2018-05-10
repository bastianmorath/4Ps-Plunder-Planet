# main-file:
'''
# Does feature selection with SVM l1 loss
cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = dict(enumerate(cw))
feature_selection = False
if feature_selection:
    print('Feature selection with LinearSVC l1-loss: \n')
    clf = svm.LinearSVC(class_weight=class_weight_dict, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    X = model.transform(X)
    features = [f_factory.feature_names[i] for i in range(0, len(f_factory.feature_names)) if not model.get_support()[i]]
    print('Features not selected:: ' + str(features))

'''