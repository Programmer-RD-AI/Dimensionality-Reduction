from DimRed import *

# XGBoost configuration
xgb_config = {
    "objective": "multi:softmax",  # Objective function for XGBoost
    "n_estimators": 100,  # Number of trees in the forest
    "max_depth": 3,  # Maximum depth of each tree
    "learning_rate": 0.1,  # Learning rate for boosting
    "subsample": 0.8,  # Subsample ratio of the training instances
    "seed": 0,  # Random seed
    "tree_method": "gpu_hist",  # Tree construction method
    "num_class": 10,  # Number of classes
}

# LightGBM configuration
lgb_config = {
    "objective": "multiclass",  # Objective function for LightGBM
    "num_leaves": 31,  # Maximum number of leaves in one tree
    "learning_rate": 0.05,  # Learning rate for boosting
    "max_depth": -1,  # Maximum depth of each tree
    "subsample": 0.8,  # Subsample ratio of the training instances
    "metric": "multi_logloss",  # Metric to be used for evaluation
    "seed": 0,  # Random seed
    "device": "gpu",  # Device to use for training
    "num_class": 10,  # Number of classes
    "verbose": -1,  # Verbosity mode
}

# Scikit-learn configuration
sklearn_config = {
    LogisticRegression: {},  # Configuration for Logistic Regression
    SVC: {
        "kernel": ["rbf"],  # Kernel type for Support Vector Classifier
        "probability": [True],  # Whether to enable probability estimates
    },
    DecisionTreeClassifier: {},  # Configuration for Decision Tree Classifier
    RandomForestClassifier: {},  # Configuration for Random Forest Classifier
    KNeighborsClassifier: {},  # Configuration for K-Nearest Neighbors Classifier
    GaussianProcessClassifier: {},  # Configuration for Gaussian Process Classifier
    MLPClassifier: {
        "alpha": [1, 2, 3, 4, 5],  # Regularization parameter for MLP Classifier
        "max_iter": [100, 200, 400, 800],  # Maximum number of iterations
    },
    AdaBoostClassifier: {
        "algorithm": ["SAMME"]
    },  # Configuration for AdaBoost Classifier
    GaussianNB: {},  # Configuration for Gaussian Naive Bayes Classifier
    QuadraticDiscriminantAnalysis: {},  # Configuration for Quadratic Discriminant Analysis Classifier
}
