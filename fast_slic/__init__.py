from .base_slic import Slic, SlicRealDist, SlicRealDistL2
from cfast_slic import get_supported_archs

supported_archs = tuple(get_supported_archs())
