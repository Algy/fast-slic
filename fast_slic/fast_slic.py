from cfast_slic import SlicModel
from .base_slic import BaseSlic


class Slic(BaseSlic):
    def __init__(self, num_components=None, slic_model=None, compactness_shift=10, quantize_level=6):
        super().__init__(
            num_components=num_components,
            slic_model=slic_model,
            compactness_shift=compactness_shift,
            quantize_level=quantize_level,
        )

    def make_slic_model(self, num_components):
        return SlicModel(num_components)

