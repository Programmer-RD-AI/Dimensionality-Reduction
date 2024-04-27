from DimRed import *
from DimRed.variation.variation import _Variation


class Variations(object):
    def __init__(self, param_grids: List[Dict], reduction_methods: List, standard_pipeline: Pipeline, analysis_instance: Analysis) -> None:
        self.param_grids = param_grids
        self.dimensionality_reduction_methods = reduction_methods
        self.standard_pipeline = standard_pipeline
        self.internal_var = _Variation(self.standard_pipeline, analysis_instance=analysis_instance)

    def __iter__(self) -> Tuple[str, str]:
        for param_grid, reduction_method in zip(self.param_grids, self.dimensionality_reduction_methods):
            yield reduction_method, param_grid

    def __len__(self) -> Tuple[int, int]:
        return len(self.param_grids), len(self.dimensionality_reduction_methods)

    def produce_variations(self) -> Dict[str, List]:
        variations = {}
        for param_grid, reduction_method in zip(self.param_grids, self.dimensionality_reduction_methods):
            reduction_method_name = reduction_method().__class__.__name__
            variations[reduction_method_name] = self.internal_var.create_variation(
                param_grid, reduction_method, reduction_method_name)
        return variations
