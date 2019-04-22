class BaseSlic(object):
    def __init__(self, num_components, slic_model, compactness, quantize_level):
        self.compactness = compactness
        self.quantize_level = quantize_level
        self._slic_model = slic_model and slic_model.copy() or self.make_slic_model(num_components or 100)
        self._last_assignment = None

    @property
    def slic_model(self):
        return self._slic_model

    @property
    def last_assignment(self):
        return self._last_assignment

    def iterate(self, image, max_iter=10):
        if not self._slic_model.initialized:
            self._slic_model.initialize(image)
        assignment = self._slic_model.iterate(image, max_iter, self.compactness, self.quantize_level)
        self._last_assignment = assignment
        return assignment

    @property
    def num_components(self):
        return self._slic_model.num_components

    def make_slic_model(self, num_components):
        raise NotImplementedError
