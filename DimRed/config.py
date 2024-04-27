from DimRed import *


xgb_config = {
    'objective': 'multi:softmax',
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    # 'subsample': 0.8,
    'seed': 42,
    'tree_method': 'gpu_hist',
    "num_class": 10
}
lgb_config = {
    'objective': 'multiclass',
    # 'num_leaves': 31,
    'learning_rate': 0.05,
    # 'max_depth': -1,
    # 'subsample': 0.8,
    'metric': 'multi_logloss',
    'seed': 42,
    'device': 'gpu',
    "num_class": 10,
    "verbose": -1
}
sklearn_config = {
    LogisticRegression: {"max_iter": [100]},
    SVC: {"kernel": ["rbf"], "probability": [True]},
    DecisionTreeClassifier: {"max_depth": [3]},
    RandomForestClassifier: {"n_estimators": [100, 1000]},
    KNeighborsClassifier: {'n_neighbors': [3]}
}
