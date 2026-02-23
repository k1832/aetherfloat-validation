import numpy as np
import matplotlib.pyplot as plt
import math
import struct

def float_to_af16(x):
    if x == 0.0 or math.isnan(x) or math.isinf(x): return x
    abs_x = abs(x)
    sign = -1.0 if x < 0.0 else 1.0

    frac, frexp_e = math.frexp(abs_x)
    E_true_b4 = (frexp_e - 1) // 2
    E = E_true_b4 + 63

    if E <= 0:
        E = 0
        M = round(abs_x * (2.0 ** 130))
        if M >= 64: E = 1
    elif E >= 127:
        return sign * float('inf')
    else:
        M = round((abs_x / (4.0 ** E_true_b4)) * 64.0)
        if M >= 256:
            E += 1; M = M // 4
            if E >= 127: return sign * float('inf')

    val = (M * (2.0 ** -130)) if E == 0 else ((M / 64.0) * (4.0 ** (E - 63)))
    return sign * val

def float_to_bf16(x):
    if x == 0.0 or math.isnan(x) or math.isinf(x): return x
    b = struct.pack('>f', x)
    i = struct.unpack('>I', b)[0]
    rounding_bias = 0x7FFF + ((i >> 16) & 1)
    i_rounded = min(i + rounding_bias, 0xFFFFFFFF)
    return struct.unpack('>f', struct.pack('>I', i_rounded & 0xFFFF0000))[0]

np.random.seed(42)
print("Simulating 100,000 floats...")
num_points = 100000
exponents = np.random.uniform(-20, 20, num_points)
test_vals = 2.0 ** exponents

af16_sqnr_list, bf16_sqnr_list = [], []

for v in test_vals:
    af16_v = float_to_af16(v)
    bf16_v = float_to_bf16(v)

    if v == 0:
        af16_sqnr_list.append(0)
        bf16_sqnr_list.append(0)
        continue

    diff_af = abs(v - af16_v)
    diff_bf = abs(v - bf16_v)

    af16_err = diff_af / v if v != 0 else 0
    bf16_err = diff_bf / v if v != 0 else 0

    af16_sqnr_list.append(20 * math.log10(1 / (af16_err + 1e-12)))
    bf16_sqnr_list.append(20 * math.log10(1 / (bf16_err + 1e-12)))

af16_sqnr = np.array(af16_sqnr_list)
bf16_sqnr = np.array(bf16_sqnr_list)

print(f"bfloat16 Mean SQNR: {np.mean(bf16_sqnr):.2f} dB")
print(f"AF16 Mean SQNR:     {np.mean(af16_sqnr):.2f} dB")
print(f"AF16 Wobble Cost:   {np.mean(bf16_sqnr) - np.mean(af16_sqnr):.2f} dB")

# --- Subsampling ---
sample_size = 10000
indices = np.random.choice(num_points, sample_size, replace=False)

exponents_sampled = exponents[indices]
af16_sampled = af16_sqnr[indices]
bf16_sampled = bf16_sqnr[indices]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

# Upper
ax1.scatter(exponents_sampled, bf16_sampled, color='orange', alpha=0.5, s=2, label='bfloat16 (Base-2)')
ax1.set_title('Baseline: bfloat16 (Base-2)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Lower
ax2.scatter(exponents_sampled, af16_sampled, color='tab:blue', alpha=0.5, s=2, label='AetherFloat-16 (Base-4)')
ax2.set_title('Proposed: AetherFloat-16 (Base-4)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

fig.text(0.5, 0.04, 'Magnitude (Log2)', ha='center')
fig.text(0.04, 0.5, 'Signal-to-Quantization-Noise Ratio (dB)', va='center', rotation='vertical')

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig('wobble_plot.pdf', format='pdf', bbox_inches='tight')
print("Saved wobble_plot.pdf!")
