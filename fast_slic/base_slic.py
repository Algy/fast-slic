from cfast_slic import SlicModel

class BaseSlic(object):
    arch_name = "__TODO__"

    def __init__(self,
                 num_components=400,
                 slic_model=None,
                 compactness=20,
                 min_size_factor=0.05,
                 subsample_stride=3):
        self.compactness = compactness
        self.subsample_stride = subsample_stride
        self.min_size_factor = min_size_factor
        self._slic_model = slic_model and slic_model.copy() or self.make_slic_model(num_components)
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
        assignment = self._slic_model.iterate(image, max_iter, self.compactness, self.min_size_factor, self.subsample_stride)
        self._last_assignment = assignment
        return assignment

    @property
    def num_components(self):
        return self._slic_model.num_components

    def make_slic_model(self, num_components):
        return SlicModel(num_components, self.arch_name)

class Slic(BaseSlic):
    arch_name = 'standard'

class SlicRealDist(BaseSlic):
    arch_name = 'standard'

    def make_slic_model(self, num_components):
        model = SlicModel(num_components, self.arch_name)
        model.real_dist = True
        return model
