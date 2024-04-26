from sklearn.decomposition import PCA, IncrementalPCA
from DimRed import *
if __name__ == "__main__":
    param_grids = [{"n_components": [1, 2, 3],
                    "random_state": [1, 42, 69, 100]}, {"n_components": [1, 2, 3]}]
    standard_pipeline = Pipeline([("StandardScalar", StandardScaler())])
    reduction_methods = [PCA, IncrementalPCA]
    all_possible_variations = Variations(param_grids=param_grids,
                                         reduction_methods=reduction_methods, standard_pipeline=standard_pipeline).produce_variations()
    print(all_possible_variations)
