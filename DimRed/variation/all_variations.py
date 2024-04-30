from DimRed import *
from DimRed.variation.variation import _Variation


class Variations(object):
    def __init__(
        self,
        param_grids: List[Dict],
        reduction_methods: List,
        standard_pipeline: Pipeline,
        analysis_instance: Analysis,
    ) -> None:
        """
        Initialize the Variations object.

        Args:
            param_grids (List[Dict]): List of parameter grids for each reduction method.
            reduction_methods (List): List of dimensionality reduction methods.
            standard_pipeline (Pipeline): Standard pipeline for analysis.
            analysis_instance (Analysis): Analysis instance.

        Returns:
            None
        """
        self.param_grids = param_grids
        self.dimensionality_reduction_methods = reduction_methods
        self.standard_pipeline = standard_pipeline
        self.internal_var = _Variation(
            self.standard_pipeline, analysis_instance=analysis_instance
        )

    def __iter__(self) -> Tuple[str, str]:
        """
        Iterate over the Variations object.

        Yields:
            Tuple[str, str]: Tuple containing reduction method and parameter grid.
        """
        for param_grid, reduction_method in zip(
            self.param_grids, self.dimensionality_reduction_methods
        ):
            yield reduction_method, param_grid

    def __len__(self) -> Tuple[int, int]:
        """
        Get the length of the Variations object.

        Returns:
            Tuple[int, int]: Tuple containing the length of param_grids and dimensionality_reduction_methods.
        """
        return len(self.param_grids), len(self.dimensionality_reduction_methods)

    def produce_variations(self) -> Dict[str, List]:
        """
        Produce variations for each reduction method.

        Returns:
            Dict[str, List]: Dictionary containing the variations for each reduction method.
        """
        variations = {}
        iterator = tqdm(zip(self.param_grids, self.dimensionality_reduction_methods))
        for param_grid, reduction_method in iterator:
            reduction_method_name = reduction_method().__class__.__name__
            iterator.set_description(reduction_method_name)
            variations[reduction_method_name] = self.internal_var.create_variation(
                param_grid, reduction_method, reduction_method_name
            )
        return variations
