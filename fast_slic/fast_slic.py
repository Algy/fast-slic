from cfast_slic import SlicModel


class Slic(object):
    def __init__(self, num_components=None, slic_model=None, compactness_shift=6, quantize_level=6):
        self.compactness_shift = compactness_shift
        self.quantize_level = quantize_level
        self._slic_model = slic_model or SlicModel(num_components or 100)

    @property
    def slic_model(self):
        return self._slic_model.copy()

    def iterate(self, image, max_iter=10):
        if not self._slic_model.initialized:
            self._slic_model.initialize(image)
        return self._slic_model.iterate(image, max_iter, self.compactness_shift, self.quantize_level)

