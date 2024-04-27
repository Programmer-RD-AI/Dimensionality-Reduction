from DimRed import *


class Analysis:
    def __init__(self, X: np.array, y: np.array, standard_pipeline=None, pipeline=None):
        self.standard_pipeline = standard_pipeline
        self.pipeline = pipeline
        # self.standard_pipeline.fit(X)
        # self.pipeline.fit(X)
        self.X = X[0]
        self.size_per_side = int(math.sqrt(len(self.X)))
        self.X = self.X
        self.y = y[0].reshape(1, -1)

    def produce_combinations(self, name: str, standard_pipeline: Pipeline, pipeline: Pipeline) -> None:
        pass
