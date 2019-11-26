#ifndef FSLIC_PARALLEL_H
#define FSLIC_PARALLEL_H

namespace fsparallel {
    bool parallelism_supported();
    int nth();
    int thindex();
    class Scope {
    private:
        int old_val;
    public:
        Scope(int n);
        ~Scope();
    };
};

#endif
