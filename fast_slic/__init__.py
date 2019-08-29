from .base_slic import *
from cfast_slic import get_supported_archs, enforce_connectivity

supported_archs = tuple(get_supported_archs())
