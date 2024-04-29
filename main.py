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
        {"n_components": [1, 2, 3]},
        {"n_components": [1, 2, 3]},
        {
            "kernel": ["linear", "poly", "rbf", "cosine"],
            "n_components": [1, 2, 3],
            "gamma": [None],
            "fit_inverse_transform": [True, False],
            "n_jobs": [-1],
        },
        {
            "n_components": [1, 2, 3],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "n_jobs": [-1],
        },
        {
            "n_components": [1, 2, 3],
            "algorithm": ["randomized"],
            "n_iter": [1, 2, 3, 4, 5],
        },
        {"n_components": [1, 2, 3], "eps": [0.125, 0.25, 0.5, 0.625, 0.75, 1]},
        {"n_components": [1, 2, 3]},
        {"n_components": [1, 2, 3]},
        {
            "n_components": [1, 2, 3],
            "density": ["auto"],
            "eps": [0.125, 0.25, 0.5, 0.625, 0.75, 1],
            "dense_output": [True, False],
        },
        {"n_components": [1, 2, 3], "n_jobs": [-1], "n_neighbors": [1, 3, 5, 7, 9]},
        {
            "n_components": [1, 2, 3],
            "batch_size": [50, 100, 200, 400],
            "alpha": [1, 0.0001, 0.001, 0.01, 0.1],
            "n_iter": [1, 2, 3, 4, 5],
        },
        {
            "n_components": [1, 2, 3],
            "algorithm": ["parallel", "deflation"],
            "whiten": [True, False],
            "max_iter": [25, 50, 75, 100],
        },
        {
            "n_components": [1, 2, 3],
            "n_neighbors": [10],
            "method": ["modified"],
            "n_jobs": [4],
        },
    ]
    reduction_methods = [
        PCA,
        IncrementalPCA,
        KernelPCA,
        SparsePCA,
        TruncatedSVD,
        GaussianRandomProjection,
        LinearDiscriminantAnalysis,
        NeighborhoodComponentsAnalysis,
        SparseRandomProjection,
        Isomap,
        MiniBatchDictionaryLearning,
        FastICA,
        LocallyLinearEmbedding,
    ]
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
    all_pipeline_performance, best_performances = Evaluation(
        _data={
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        all_possible_variations=all_possible_variations,
        labels=np.unique(y_train),
    ).evaluate()
    pprint(best_performances)
