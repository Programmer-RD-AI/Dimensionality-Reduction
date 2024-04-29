from DimRed import *


class _Variation:
    def __init__(
        self, standard_pipeline: Pipeline, analysis_instance: Analysis
    ) -> None:
        self.standard_pipeline = standard_pipeline
        self.analysis_instance = analysis_instance

    def create_variation(
        self, param_grid: Dict, reduction_method: Any, reduction_method_name: str
    ) -> List:
        grid = ParameterGrid(param_grid)
        reduction_method = reduction_method
        variations = []
        variation_iterator = tqdm(grid, leave=False)
        for params in variation_iterator:
            var = [(reduction_method_name, reduction_method(**params))]
            pipeline = Pipeline([step for step in self.standard_pipeline.steps] + var)
            self.analysis_instance.produce_combinations(
                reduction_method_name, self.standard_pipeline, pipeline
            )
            variations.append(pipeline)
        return variations
