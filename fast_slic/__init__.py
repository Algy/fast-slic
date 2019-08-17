from .base_slic import Slic, SlicRealDist, SlicRealDistL2, LSC
from cfast_slic import get_supported_archs, enforce_connectivity

supported_archs = tuple(get_supported_archs())
