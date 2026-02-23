"""
AetherFloat Hardware-Software Co-Design Simulation Library
Contains PyTorch native simulations for Base-4 Quad-Radix Scaling,
Vector-Shared Stochastic Rounding, and FP8 AMAX Block-Scaling baselines.
"""

import torch
import torch.nn as nn

# =============================================================================
# 1. CORE AETHERFLOAT TENSOR QUANTIZATION
# =============================================================================

def float_to_af_tensor(x, bits, stochastic=False, vector_chunk_size=16):
    """
    Hardware-accurate quantization simulation with autograd graph preservation.
    """
    orig_dtype = x.dtype
    val = x.float().clone()

    # Hardware Sign Extraction
    sign = torch.sign(val)
    sign[sign == 0] = 1.0

    # Absolute value in-place
    val.abs_()

    # Base-4 Quad-Radix Exponent
    # torch.frexp extracts the binary exponent via bit manipulation — exact by IEEE 754.
    # This avoids fast-math compiler approximations that can affect torch.log2.
    # frexp returns mantissa in [0.5, 1.0), so true floor(log2(x)) = exponent - 1.
    _, e_b2_raw = torch.frexp(val)          # bit-exact; returns int exponent
    e_b2 = e_b2_raw.float() - 1.0          # frexp mantissa in [0.5,1), so true floor(log2) = exp-1
    e_b2.masked_fill_(val == 0.0, -256.0)  # push zeros far into subnormal space
    e_true_b4 = torch.floor(e_b2 / 2.0)
    del e_b2, e_b2_raw

    if bits == 16:
        bias, mant_mul, max_exp, sub_bound = 63.0, 64.0, 127.0, -130.0
    elif bits == 8:
        bias, mant_mul, max_exp, sub_bound = 7.0, 2.0, 15.0, -13.0
    else:
        raise ValueError("Specified bit width not instantiated in this specific emulation path.")

    E = e_true_b4 + bias
    is_subnormal = E <= 0.0

    # Explicit Mantissa Extraction
    # Prevent float32 overflow (max ~3.4e38) by upcasting to float64
    # for the 2^130 multiplication, then casting back to float32.
    if bits == 16:
        M_sub = (val.double() * (2.0 ** 130.0)).float()
    else:
        M_sub = val * (2.0 ** -sub_bound)

    M_norm = (val / (4.0 ** e_true_b4)) * mant_mul
    del val, e_true_b4

    M_float = torch.where(is_subnormal, M_sub, M_norm)
    del M_sub, M_norm

    E.masked_fill_(is_subnormal, 0.0)
    E.clamp_(max=max_exp)
    del is_subnormal

    # Hardware Rounding
    if stochastic:
        flat_M = M_float.view(-1)
        numel = flat_M.numel()
        num_chunks = (numel + vector_chunk_size - 1) // vector_chunk_size

        noise = torch.rand(num_chunks, 1, device=x.device)
        noise = noise.expand(num_chunks, vector_chunk_size).reshape(-1)[:numel].view_as(M_float)

        M_int = torch.floor(M_float + noise)
        del flat_M, noise
    else:
        M_int = torch.round(M_float)

    del M_float

    # Mantissa Overflow Boundary
    is_overflow = (M_int >= mant_mul * 4.0) & (E > 0.0)
    M_int = torch.where(is_overflow, torch.floor(M_int / 4.0), M_int)

    E = E + is_overflow.to(E.dtype)
    del is_overflow

    sub_to_norm = (E == 0.0) & (M_int >= mant_mul)
    E.masked_fill_(sub_to_norm, 1.0)
    del sub_to_norm

    # Hardware Reconstruction
    # Upcast to float64 to prevent float32 underflow flush-to-zero
    if bits == 16:
        val_sub = (M_int.double() * (2.0 ** -130.0)).float()
    else:
        val_sub = M_int * (2.0 ** sub_bound)

    val_norm = (M_int / mant_mul) * (4.0 ** (E - bias))
    del M_int

    val = torch.where(E == 0.0, val_sub, val_norm)
    del val_sub, val_norm

    val.masked_fill_(E >= max_exp, float('inf'))
    del E

    val.mul_(sign)
    del sign

    val.masked_fill_(x == 0.0, 0.0)
    return val.to(orig_dtype)

# =============================================================================
# 2. POST-TRAINING QUANTIZATION (PTQ) INFERENCE CLASSES
# =============================================================================

class AetherLinearPTQ(nn.Module):
    def __init__(self, original_linear: nn.Linear, bits: int):
        super().__init__()
        self.bits = bits
        with torch.no_grad():
            self.weight = nn.Parameter(
                float_to_af_tensor(original_linear.weight, bits=bits, stochastic=False),
                requires_grad=False
            )
            if original_linear.bias is not None:
                self.bias = nn.Parameter(
                    float_to_af_tensor(original_linear.bias, bits=bits, stochastic=False),
                    requires_grad=False
                )
            else:
                self.register_parameter('bias', None)

    def forward(self, x):
        x_q = float_to_af_tensor(x, self.bits, stochastic=False)
        w_curr = self.weight.to(device=x.device, dtype=x.dtype)
        b_curr = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return nn.functional.linear(x_q, w_curr, b_curr)

class FP8LinearPTQ(nn.Module):
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        with torch.no_grad():
            orig_dtype = original_linear.weight.dtype
            w_f32 = original_linear.weight.float()
            self.w_amax = w_f32.abs().max().clamp(min=1e-5)
            self.w_scale = 448.0 / self.w_amax
            w_q = (w_f32 * self.w_scale).to(torch.float8_e4m3fn).float()
            self.weight = nn.Parameter((w_q / self.w_scale).to(orig_dtype), requires_grad=False)
            if original_linear.bias is not None:
                self.bias = nn.Parameter(original_linear.bias.clone().to(orig_dtype), requires_grad=False)
            else:
                self.register_parameter('bias', None)

    def forward(self, x):
        x_orig_dtype = x.dtype
        x_f32 = x.float()
        x_amax = x_f32.abs().max().clamp(min=1e-5)
        x_scale = 448.0 / x_amax
        x_q = (x_f32 * x_scale).to(torch.float8_e4m3fn).float()
        x_dequant = (x_q / x_scale).to(x_orig_dtype)
        w_curr = self.weight.to(device=x.device, dtype=x_orig_dtype)
        b_curr = self.bias.to(device=x.device, dtype=x_orig_dtype) if self.bias is not None else None
        return nn.functional.linear(x_dequant, w_curr, b_curr)

def patch_model_ptq(model, fmt, bits=None):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name != "lm_head":
            if fmt == "af":
                setattr(model, name, AetherLinearPTQ(module, bits))
            elif fmt == "fp8":
                setattr(model, name, FP8LinearPTQ(module))
        else:
            patch_model_ptq(module, fmt, bits)
    return model

# =============================================================================
# 3. QUANTIZATION-AWARE TRAINING (QAT) AUTOGRAD CLASSES
# =============================================================================

class AetherSimQAT(torch.autograd.Function):
    """Applies Deterministic Inference Simulation in Forward, STE in Backward."""
    @staticmethod
    def forward(ctx, x, bits, chunk_size):
        # Forward MUST be deterministic (stochastic=False) to mirror physical hardware execution.
        return float_to_af_tensor(x, bits, stochastic=False, vector_chunk_size=chunk_size)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE) prevents converged LLM gradient underflow
        return grad_output, None, None

class FP8SimQAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_f32 = x.float()
        x_amax = x_f32.abs().max().clamp(min=1e-5)
        scale = 448.0 / x_amax
        x_q = (x_f32 * scale).to(torch.float8_e4m3fn).float()
        return (x_q / scale).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class QATLinear(nn.Linear):
    def __init__(self, original_linear, fmt, bits, chunk_size=16):
        super().__init__(original_linear.in_features, original_linear.out_features, bias=original_linear.bias is not None, device=original_linear.weight.device, dtype=original_linear.weight.dtype)
        self.fmt = fmt
        self.bits = bits
        self.chunk_size = chunk_size
        self.weight.data = original_linear.weight.data.clone()
        if original_linear.bias is not None:
            self.bias.data = original_linear.bias.data.clone()

    def forward(self, x):
        if self.fmt == 'af':
            w_q = AetherSimQAT.apply(self.weight, self.bits, self.chunk_size)
            x_q = AetherSimQAT.apply(x, self.bits, self.chunk_size)
        elif self.fmt == 'fp8':
            w_q = FP8SimQAT.apply(self.weight)
            x_q = FP8SimQAT.apply(x)
        else:
            w_q, x_q = self.weight, x
        return nn.functional.linear(x_q, w_q, self.bias)

def patch_model_qat(model, fmt, bits=16, chunk_size=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name != "lm_head":
            setattr(model, name, QATLinear(module, fmt, bits, chunk_size))
        else:
            patch_model_qat(module, fmt, bits, chunk_size)
    return model
