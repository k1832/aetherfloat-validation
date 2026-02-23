#include <iostream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <cstdlib>

// Pack standard float to AetherFloat-16
int16_t float_to_af16(float x) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &x, sizeof(float));
    uint32_t sign = (f_bits >> 31) & 1;

    // Handle Special Cases (NaNs go to absolute extremities)
    if (std::isnan(x)) {
        uint16_t res = (127 << 8) | 255;
        if (sign) res = (~res) & 0x7FFF;
        return (int16_t)((sign << 15) | res);
    }
    if (std::isinf(x)) {
        uint16_t res = (127 << 8) | 0;
        if (sign) res = (~res) & 0x7FFF;
        return (int16_t)((sign << 15) | res);
    }
    if (x == 0.0f) {
        uint16_t res = 0;
        if (sign) res = (~res) & 0x7FFF;
        return (int16_t)((sign << 15) | res);
    }

    float abs_x = std::abs(x);
    int frexp_e;
    std::frexp(abs_x, &frexp_e);
    int p = frexp_e - 1;

    // Quad-Radix (Base-4) Exponent Scaling
    int E_true_b4 = std::floor(p / 2.0f);
    int E = E_true_b4 + 63;
    int M = 0;

    if (E <= 0) {
        E = 0; // Branchless Subnormal handling
        // Cast to double to prevent 32-bit float boundary overflow on 2^130
        M = std::round(static_cast<double>(abs_x) * std::pow(2.0, 130.0));
        if (M >= 64) { E = 1; }
    } else if (E >= 127) {
        E = 127; M = 0; // Overflow
    } else {
        M = std::round((abs_x / std::pow(4.0f, E_true_b4)) * 64.0f);
        if (M >= 256) { // Mantissa rounding boundary push
            E += 1; M = M / 4;
            if (E >= 127) { E = 127; M = 0; }
        }
    }

    uint16_t U = (E << 8) | M;
    if (sign) U = (~U) & 0x7FFF; // 1-Gate Delay One's Complement

    return (int16_t)((sign << 15) | U);
}

// Unpack AetherFloat-16 to standard float
float af16_to_float(int16_t x) {
    uint16_t bits = (uint16_t)x;
    uint32_t sign = (bits >> 15) & 1;
    uint16_t U = bits & 0x7FFF;

    if (sign) U = (~U) & 0x7FFF; // Reverse One's Complement

    int E = (U >> 8) & 0x7F;
    int M = U & 0xFF;

    double val;
    if (E == 127) {
        if (M == 0) val = INFINITY;
        else val = NAN;
    } else if (E == 0) {
        // Math executed in double to prevent overflow, then cast
        val = static_cast<double>(M) * std::pow(2.0, -130.0);
    } else {
        val = (static_cast<double>(M) / 64.0) * std::pow(4.0, E - 63);
    }

    return sign ? static_cast<float>(-val) : static_cast<float>(val);
}

int main() {
    std::vector<float> test_vals = {
        0.0f, -0.0f, 1.0f, -1.0f, 3.14159f, -3.14159f,
        INFINITY, -INFINITY, NAN, -NAN, 1e-10f, -1e-10f
    };

    // Add 1,000,000 random floating point numbers
    srand(42);
    for (int i = 0; i < 1000000; i++) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        // Expand range to 10^-40 to 10^40 to aggressively test subnormals
        float scale = std::pow(10.0f, (rand() % 80) - 40);
        test_vals.push_back((rand() % 2 == 0 ? r : -r) * scale);
    }

    std::vector<int16_t> packed;
    for (float f : test_vals) {
        packed.push_back(float_to_af16(f));
    }

    // Lexicographic sort via native integer comparison
    std::sort(packed.begin(), packed.end());

    int errors = 0;
    float prev_val = -INFINITY;

    for (int16_t p : packed) {
        float unpacked = af16_to_float(p);
        if (!std::isnan(unpacked)) {
            if (unpacked < prev_val) {
                errors++;
            }
            prev_val = unpacked;
        }
    }

    std::cout << "--- O(1) Lexicographical Sorting Validation ---\n";
    std::cout << "Total numbers natively sorted: " << packed.size() << "\n";
    std::cout << "Monotonicity Errors: " << errors << "\n";
    if (errors == 0) {
        std::cout << "SUCCESS: Perfect mathematical ordering validated deep into subnormal space!\n";
    }

    return 0;
}
