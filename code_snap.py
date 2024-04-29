inc_pca = make_pipeline(StandardScaler(), IncrementalPCA(n_components=2))
