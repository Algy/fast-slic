#include <cstdlib>
#include <string>
#include <stdexcept>
#ifdef _OPENMP
#define PARALLELISM_SUPPORTED true
#include <omp.h>
#else
#define PARALLELISM_SUPPORTED false
#define omp_get_max_threads() 1
#endif
#include "parallel.h"

namespace fsparallel {
    thread_local int num_threads = -1;

    bool parallelism_supported() {
        return PARALLELISM_SUPPORTED;
    }

    static int parse_env(const char *name) {
        if (const char * env_p = std::getenv(name)) {
            try {
                return std::stoi(std::string(env_p));
            } catch (const std::invalid_argument &ia) {
                return -1;
            }
        }
        return -1;
    }

    int nth() {
        if (num_threads < 0) {
            int n;
            n = parse_env("FSLIC_NUM_THREADS");
            if (n > 0) {
                return n;
            } else if (n == 0) {
                return omp_get_max_threads();
            }

            n = parse_env("OMP_NUM_THREADS");
            if (n > 0) {
                return n;
            } else if (n == 0) {
                return omp_get_max_threads();
            }
            return omp_get_max_threads();
        } else if (num_threads == 0) {
            return omp_get_max_threads();
        } else {
            return num_threads;
        }
    }

    Scope::Scope(int n) {
        old_val = num_threads;
        num_threads = n;
    }

    Scope::~Scope() {
        num_threads = old_val;
    }
};
