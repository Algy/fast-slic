#include <vector>
#include <string>
#include <map>
#include "context.h"

#ifdef USE_AVX2
#include "arch/x64/avx2.h"
#endif

#ifdef USE_NEON
#include "arch/arm/neon.h"
#endif


const char* archtbl[] = {
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
    class ContextBuilderImpl {
    public:
        std::string arch;
    };

    ContextBuilder::ContextBuilder(const char* arch) {
        impl = std::unique_ptr<ContextBuilderImpl> { new ContextBuilderImpl() };
        impl->arch = arch;
    }

    ContextBuilder::~ContextBuilder() {}
    const char** ContextBuilder::supported_archs() { return archtbl; };
    const char* ContextBuilder::get_arch() { return impl->arch.c_str(); };
    void ContextBuilder::set_arch(const char* arch) { impl->arch = arch; };

    bool ContextBuilder::is_supported_arch() {
        for (const char **p = archtbl; *p; p++) {
            if (impl->arch == *p) return true;
        }
        return false;
    }

    Context* ContextBuilder::build(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
        const std::string &arch = impl->arch;
        if (arch == "standard") {
            return new Context(H, W, K, image, clusters);
#ifdef USE_AVX2
        } else if (arch == "x64/avx2") {
            return new Context_X64_AVX2(H, W, K, image, clusters);
#endif
#ifdef USE_NEON
        } else if (arch == "arm/neon") {
            return new Context_ARM_NEON(H, W, K, image, clusters);
#endif
        } else {
            return nullptr;
        }
    };
};
