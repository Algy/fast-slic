#ifndef _SIMD_HELPER_HPP
#define _SIMD_HELPER_HPP
#include <cstdlib>


#ifdef _MSC_VER
#    if (_MSC_VER >= 1800)
#        define __alignas_is_defined 1
#    endif
#    if (_MSC_VER >= 1900)
#        define __alignof_is_defined 1
#    endif
#else
// #    include <cstdalign>   // __alignas/of_is_defined directly from the implementation
#endif

#ifdef __alignas_is_defined
#    define ALIGN(X) alignas(X)
#else
#    pragma message("C++11 alignas unsupported :( Falling back to compiler attributes")
#    ifdef __GNUG__
#        define ALIGN(X) __attribute__ ((aligned(X)))
#    elif defined(_MSC_VER)
#        define ALIGN(X) __declspec(align(X))
#    else
#        error Unknown compiler, unknown alignment attribute!
#    endif
#endif

#ifdef __alignof_is_defined
#    define ALIGNOF(X) alignof(x)
#else
#    pragma message("C++11 alignof unsupported :( Falling back to compiler attributes")
#    ifdef __GNUG__
#        define ALIGNOF(X) __alignof__ (X)
#    elif defined(_MSC_VER)
#        define ALIGNOF(X) __alignof(X)
#    else
#        error Unknown compiler, unknown alignment attribute!
#    endif
#endif



// AVX2 needs 32 byte alignment
#define Alignment 32
#define ALIGN_SIMD ALIGN(Alignment)


#ifndef _MSC_VER
#define HINT_ALIGNED(variable) __builtin_assume_aligned(variable, Alignment)
#define HINT_ALIGNED_AS(variable, alignment) __builtin_assume_aligned(variable, alignment)
#else
#define HINT_ALIGNED(variable) variable
#define HINT_ALIGNED_AS(variable, alignment) variable
#endif


#include <memory>

namespace simd_helper { 
    template <typename T>
    static T* alloc_aligned_array(std::size_t count) {
        std::size_t size = count * sizeof(T);
        std::size_t alignment = Alignment;
        uintptr_t r = (uintptr_t)calloc(size + --alignment + sizeof(uintptr_t), 1);
        uintptr_t t = r + sizeof(uintptr_t);
        uintptr_t o =(t + alignment) & ~(uintptr_t)alignment;
        if (!r) return NULL;
        ((uintptr_t*)o)[-1] = r;
        return (T *)o;
    }

    template <typename T>
    static T* copy_and_align_array(const T* x, std::size_t original_count) {
        T* new_obj = alloc_aligned_array<T>(original_count);
        std::memcpy(new_obj, x, sizeof(T) * original_count);
        return new_obj;
    }

    static void free_aligned_array(void* array) {
        free((void*)(((uintptr_t*)array)[-1]));
    }

    template <typename T>
    static T align_to_next(T x) {
        return (x + (Alignment - 1)) & (~(Alignment - 1));
    }

}


#endif
