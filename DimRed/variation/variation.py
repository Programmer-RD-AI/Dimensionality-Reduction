from DimRed import *


class _Variation:
    """
    Produce variations for each reduction method.

    Returns:
        Dict[str, List]: Dictionary containing the variations for each reduction method.
    """

    def __init__(
        self, standard_pipeline: Pipeline, analysis_instance: Analysis
    ) -> None:
        self.standard_pipeline = standard_pipeline
        self.analysis_instance = analysis_instance

    def create_variation(
        self, param_grid: Dict, reduction_method: Any, reduction_method_name: str
    ) -> List:
        """
        Creates variations of the standard pipeline by varying its hyperparameters.

        Args:
            param_grid (Dict): A dictionary that maps hyperparameter names to lists of values.
            reduction_method (Any): The reduction method that will be used to create the variations.
            reduction_method_name (str): The name of the reduction method that will be used to create the variations.

        Returns:
            List: A list of pipelines that are variations of the standard pipeline.
        """
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
