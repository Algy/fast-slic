from cfast_slic import is_supported_arch
from .base_slic import BaseSlic, LSC

if not is_supported_arch("arm/neon"):
    raise ImportError(
        "fast_slic is not configured with neon support. "
        "Compile it again with flag USE_NEON."
    )

class SlicNeon(BaseSlic):
    arch_name = "arm/neon"

class LSCNeon(LSC):
    arch_name = "arm/neon"
