from cfast_slic import SlicModel, is_supported_arch
from .base_slic import BaseSlic, LSC

if not is_supported_arch("x64/avx2"):
    raise ImportError(
        "fast_slic is not configured with avx2 support. "
        "Compile it again with flag USE_AVX2."
    )

class SlicAvx2(BaseSlic):
    arch_name = "x64/avx2"

class LSCAvx2(LSC):
    arch_name = "x64/avx2"
