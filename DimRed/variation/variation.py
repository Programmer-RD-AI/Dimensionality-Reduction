from DimRed import *


class _Variation:
    def __init__(self,  standard_pipeline: Pipeline) -> None:
        self.standard_pipeline = standard_pipeline

    def create_variation(self, param_grid: Dict, reduction_method: Any, reduction_method_name: str) -> List:
        grid = ParameterGrid(param_grid)
        reduction_method = reduction_method
        variations = []
        for params in grid:
            var = [(reduction_method_name, reduction_method(**params))]
            pipeline = Pipeline(
                [step for step in self.standard_pipeline.steps] + var)
            variations.append(pipeline)
        return variations
