from DimRed import *


xgb_config = {
    "objective": "multi:softmax",
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "seed": 42,
    "tree_method": "gpu_hist",
    "num_class": 10,
}
lgb_config = {
    "objective": "multiclass",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": -1,
    "subsample": 0.8,
    "metric": "multi_logloss",
    "seed": 42,
    "device": "gpu",
    "num_class": 10,
    "verbose": -1,
}

sklearn_config = {
    LogisticRegression: {},
    SVC: {
        "kernel": ["rbf"],
        "probability": [True],
    },
    DecisionTreeClassifier: {},
    RandomForestClassifier: {},
    KNeighborsClassifier: {},
    GaussianProcessClassifier: {},
    MLPClassifier: {"alpha": [1, 2, 3, 4, 5], "max_iter": [100, 200, 400, 800]},
    AdaBoostClassifier: {"algorithm": ["SAMME"]},
    GaussianNB: {},
    QuadraticDiscriminantAnalysis: {},
}
