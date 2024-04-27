from DimRed import *


def config():
    param_grids = [{"n_components": [2],
                    "random_state": [42]}, {"n_components": [2]}]
    standard_pipeline = Pipeline([("StandardScalar", StandardScaler())])
    reduction_methods = [PCA, IncrementalPCA]
    return param_grids, standard_pipeline, reduction_methods


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    param_grids, standard_pipeline, reduction_methods = config()
    all_possible_variations = Variations(param_grids=param_grids,
                                         reduction_methods=reduction_methods, standard_pipeline=standard_pipeline, analysis_instance=Analysis(X_train, y_train)).produce_variations()
    all_pipeline_performance, best_performances = Evaluation(_data={"X_train": X_train, "X_test": X_test, "y_train": y_train,
                                                                    "y_test": y_test}, all_possible_variations=all_possible_variations, labels=np.unique(y_train)).evaluate()
    pprint(best_performances)
