# Demo - Classification
import pandas as pd
from dynapipe.autoFS import dynaFS_clf

tr_features = pd.read_csv('./data/classification/train_features.csv')
tr_labels = pd.read_csv('./data/classification/train_labels.csv')

clf_fs_demo = dynaFS_clf( fs_num =5,random_state=13,cv = 5)

clf_fs_demo.fit_fs_clf(tr_features,tr_labels)


# # Demo - Regression
# import pandas as pd
# from dynapipe.autoFS import dynaFS_reg

# tr_features = pd.read_csv('./data/regression/train_features.csv')
# tr_labels = pd.read_csv('./data/regression/train_labels.csv')

# reg_fs_demo = dynaFS_reg( fs_num = 5,random_state = 13,cv = 5,input_from_file = True)

# reg_fs_demo.fit_fs_reg(tr_features,tr_labels)


# Selectors Demo - classification

# import pandas as pd

# tr_features = pd.read_csv('../data/classification/train_features.csv')
# tr_labels = pd.read_csv('../data/classification/train_labels.csv')
# val_features = pd.read_csv('../data/classification/val_features.csv')
# val_labels = pd.read_csv('../data/classification/val_labels.csv')
# te_features = pd.read_csv('../data/classification/test_features.csv')
# te_labels = pd.read_csv('../data/classification/test_labels.csv')

# tr_labels = tr_labels.values.ravel()
# clf_demo = clf_fs(fs_num = 3)
# clf_demo = clf_demo.rfecv_rf()
# result = clf_demo.fit(tr_features,tr_labels)
# print(result.get_support())


'''
# Selectors Demo - regression

# import pandas as pd

# tr_features = pd.read_csv('../data/regression/train_features.csv')
# tr_labels = pd.read_csv('../data/regression/train_labels.csv')
# # val_features = pd.read_csv('../data/val_features.csv')
# # val_labels = pd.read_csv('../data/val_labels.csv')
# # te_features = pd.read_csv('../data/test_features.csv')
# # te_labels = pd.read_csv('../data/test_labels.csv')

# tr_labels = tr_labels.values.ravel()
# reg_demo = reg_fs(fs_num = 3)
# reg_demo = reg_demo.rfecv_rf()
# result = reg_demo.fit(tr_features,tr_labels)
# print(result.get_support())


'''