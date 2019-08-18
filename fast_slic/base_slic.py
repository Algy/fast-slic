from cfast_slic import SlicModel

class BaseSlic(object):
    arch_name = "__TODO__"

    def __init__(self,
                 num_components=400,
                 slic_model=None,
                 compactness=20,
                 min_size_factor=0.05,
                 subsample_stride=3,
                 convert_to_lab=False,
                 strict_cca=True,
                 preemptive=False,
                 preemptive_thres=0.05):
        self.compactness = compactness
        self.subsample_stride = subsample_stride
        self.min_size_factor = min_size_factor
        self._slic_model = slic_model and slic_model.copy() or self.make_slic_model(num_components)
        self._last_assignment = None

        self.convert_to_lab = convert_to_lab
        self._slic_model.strict_cca = strict_cca
        self._slic_model.preemptive = preemptive
        self._slic_model.preemptive_thres = preemptive_thres

    @property
    def convert_to_lab(self):
        return self._slic_model.convert_to_lab

    @convert_to_lab.setter
    def convert_to_lab(self, v):
        self._slic_model.convert_to_lab = v

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
    real_dist_type = 'standard'

    def make_slic_model(self, num_components):
        model = SlicModel(num_components, self.arch_name)
        model.real_dist = True
        model.real_dist_type = self.real_dist_type
        return model

class SlicRealDistL2(SlicRealDist):
    arch_name = 'standard'
    real_dist_type = 'l2'

class LSC(SlicRealDist):
    arch_name = 'standard'
    real_dist_type = 'lsc'
