from cfast_slic import SlicModel
from .base_slic import BaseSlic


class Slic(BaseSlic):
    def __init__(self, num_components=None, slic_model=None, compactness=20, min_size_factor=0.05, quantize_level=6):
        super().__init__(
            num_components=num_components,
            slic_model=slic_model,
            compactness=compactness,
            min_size_factor=min_size_factor,
            quantize_level=quantize_level,
        )

    def make_slic_model(self, num_components):
        return SlicModel(num_components)

