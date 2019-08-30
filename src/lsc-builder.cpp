#include <vector>
#include <string>
#include <map>
#include "lsc.h"

#ifdef USE_AVX2
#include "arch/x64/avx2.h"
#endif

#ifdef USE_NEON
#include "arch/arm/neon.h"
#endif


const char* archtbl_lsc[] = {
    "standard",
#ifdef USE_AVX2
    "x64/avx2",
#endif
#ifdef USE_NEON
    "arm/neon",
#endif
    nullptr
};

namespace fslic {
    class ContextLSCBuilderImpl {
    public:
        std::string arch;
    };

    ContextLSCBuilder::ContextLSCBuilder(const char* arch) {
        impl = std::unique_ptr<ContextLSCBuilderImpl> { new ContextLSCBuilderImpl() };
        impl->arch = arch;
    }

    ContextLSCBuilder::~ContextLSCBuilder() {}
    const char** ContextLSCBuilder::supported_archs() { return archtbl_lsc; };
    const char* ContextLSCBuilder::get_arch() { return impl->arch.c_str(); };
    void ContextLSCBuilder::set_arch(const char* arch) { impl->arch = arch; };

    bool ContextLSCBuilder::is_supported_arch() {
        for (const char **p = archtbl_lsc; *p; p++) {
            if (impl->arch == *p) return true;
        }
        return false;
    }

    ContextLSC* ContextLSCBuilder::build(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
        const std::string &arch = impl->arch;
        if (arch == "standard") {
            return new ContextLSC(H, W, K, image, clusters);
#ifdef USE_AVX2
        } else if (arch == "x64/avx2") {
            return new ContextLSC_X64_AVX2(H, W, K, image, clusters);
#endif
#ifdef USE_NEON
        } else if (arch == "arm/neon") {
            return new ContextLSC_ARM_NEON(H, W, K, image, clusters);
#endif
        } else {
            return nullptr;
        }
    };
};
