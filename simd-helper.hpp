#ifndef _SIMD_HELPER_HPP
#define _SIMD_HELPER_HPP
#include <cstdlib>
#include <algorithm>


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


    template <typename T>
    class AlignedArray {
    private:
        T* base_arr;
        T* arr;
        int height;
        int width;
        int padding_t;
        int padding_b;
        int padding_l;
        int padding_r;
        int outer_height;
        int outer_width;
        int memory_width;
    public:
        AlignedArray() : AlignedArray(0, 0) {};
        AlignedArray(int height, int width,
                    int padding_t = 0, int padding_b = 0, int padding_l = 0, int padding_r = 0)
                : height(height), width(width),
                padding_t(padding_t),
                padding_b(padding_b),
                padding_l(padding_l),
                padding_r(padding_r),
                outer_height(height + padding_t + padding_b),
                outer_width(width + padding_l + padding_r) {
            memory_width = simd_helper::align_to_next(outer_width);
            base_arr = alloc_aligned_array<T>(outer_height * memory_width);
            arr = base_arr + padding_t * memory_width + padding_l;
        };

        AlignedArray(const AlignedArray<T> &rhs) {
            width = rhs.width;
            height = rhs.height;
            padding_t = rhs.padding_t;
            padding_b = rhs.padding_b;
            padding_l = rhs.padding_l;
            padding_r = rhs.padding_r;
            outer_height = rhs.outer_height;
            outer_width = rhs.outer_width;
            memory_width = rhs.memory_width;
            base_arr = alloc_aligned_array<T>(outer_height * memory_width);
            std::copy(rhs.base_arr, rhs.base_arr + outer_height * memory_width, base_arr);
            arr = base_arr + padding_t * memory_width + padding_l;
        }

        AlignedArray& operator=(const AlignedArray<T> &rhs) {
            free_aligned_array(base_arr);
            width = rhs.width;
            height = rhs.height;
            padding_t = rhs.padding_t;
            padding_b = rhs.padding_b;
            padding_l = rhs.padding_l;
            padding_r = rhs.padding_r;
            outer_height = rhs.outer_height;
            outer_width = rhs.outer_width;
            memory_width = rhs.memory_width;
            base_arr = alloc_aligned_array<T>(outer_height * memory_width);
            std::copy(rhs.base_arr, rhs.base_arr + outer_height * memory_width, base_arr);
            arr = base_arr + padding_t * memory_width + padding_l;
            return *this;
        };

        ~AlignedArray() {
            free_aligned_array(base_arr);
        };

        inline T* get_row(int i, int j = 0) const {
            return &arr[i * memory_width + j];
        }

        inline T& get(int i, int j) const {
            return arr[i * memory_width + j];
        }

        int contiguous_memory_size() const {
            return memory_width * height;
        }

        int get_width() const {
            return width;
        }

        int get_height() const {
            return height;
        }

        int get_memory_width() const {
            return memory_width;
        }
    };
}


#endif
