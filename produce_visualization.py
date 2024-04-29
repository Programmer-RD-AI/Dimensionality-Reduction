from DimRed import *
from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    KernelPCA,
    TruncatedSVD,
    FastICA,
    MiniBatchDictionaryLearning,
    SparsePCA,
)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis


def config():
    param_grids = [
        {
            "n_components": [2],
            "algorithm": ["parallel"],
            "whiten": [True],
            "max_iter": [100],
        },
        {
            "n_components": [2],
            "n_neighbors": [10],
            "method": ["modified"],
            "n_jobs": [4],
        },
    ]
    reduction_methods = [FastICA, LocallyLinearEmbedding]
    standard_pipeline = Pipeline([("StandardScalar", StandardScaler())])
    return param_grids, standard_pipeline, reduction_methods


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    param_grids, standard_pipeline, reduction_methods = config()
    all_possible_variations = Variations(
        param_grids=param_grids,
        reduction_methods=reduction_methods,
        standard_pipeline=standard_pipeline,
        analysis_instance=Analysis(X_train, y_train),
    ).produce_variations()
