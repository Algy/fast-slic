#include <gtest/gtest.h>
#include <cielab.h>
#include <vector>

TEST(CIELABTest, rgb_to_cielab) {
    std::vector<uint8_t> quad_array(256 * 256 * 256 * 4, 0);
    std::vector<uint8_t> lab_array(256 * 256 * 256 * 4, 0);

    for (unsigned int i = 0; i < 256 * 256 * 256; i++) {
        quad_array[4 * i] = (i >> 16) & 0xFF;
        quad_array[4 * i + 1] = (i >> 8) & 0xFF;
        quad_array[4 * i + 2] = i & 0xFF;
    }

    rgb_to_cielab(&quad_array[0], &lab_array[0], quad_array.size(), true);
    for (unsigned int i = 0; i < 256 * 256 * 256; i++) {
        quad_array[4 * i] = (i >> 16) & 0xFF;
        quad_array[4 * i + 1] = (i >> 8) & 0xFF;
        quad_array[4 * i + 2] = i & 0xFF;
    }

#define EXP(R, G, B, l, a, b) { \
    int i; \
    i = R * (256 * 256) + G * (256) + B; \
    ASSERT_EQ(lab_array[4 * i + 0], l); \
    ASSERT_EQ(lab_array[4 * i + 1], a); \
    ASSERT_EQ(lab_array[4 * i + 2], b); \
}

    EXP(139, 91, 30, 42, 142, 169)
    EXP(111, 197, 143, 73, 89, 147)
    EXP(255, 255, 255, 100, 128, 128)
    EXP(255, 255, 0, 97, 106, 222)
    EXP(255, 0, 255, 60, 226, 67)
    EXP(0, 255, 255, 91, 79, 113)
    EXP(30, 57, 184, 30, 166, 58)
}
