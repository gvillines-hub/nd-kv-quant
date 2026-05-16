"""
Sanity checks for the TurboQuant implementation.

Run from the repo root:
    python tests\\test_turboquant.py
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from nd_kv_quant.turboquant import (
    _beta_pdf, get_random_rotation, get_wht_matrix,
    get_turboquant_codebook, turboquant_data,
)
from scipy.integrate import quad


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{(': ' + detail) if detail else ''}")
    return ok


print("=" * 72)
print("TurboQuant implementation sanity checks")
print("=" * 72)

all_pass = True
D = 128

LLOYD_MAX_GAUSSIAN_MSE = {1: 0.3634, 2: 0.1175, 3: 0.0345, 4: 0.00946}

integral, _ = quad(lambda x: _beta_pdf(x, D), -1, 1, limit=200)
all_pass &= check("Beta PDF integrates to 1", abs(integral - 1.0) < 1e-3, f"got {integral:.6f}")

R = get_random_rotation(D)
all_pass &= check("Random rotation orthogonal", np.max(np.abs(R @ R.T - np.eye(D))) < 1e-10)
H = get_wht_matrix(D)
all_pass &= check("WHT orthogonal", np.max(np.abs(H @ H.T - np.eye(D))) < 1e-10)

cb1 = get_turboquant_codebook(1, D)
expected_b1 = np.sqrt(2 / (np.pi * D))
all_pass &= check(
    "b=1 centroid magnitude matches paper page 10",
    abs(np.abs(cb1[0]) - expected_b1) < 5e-3,
    f"expected +/-{expected_b1:.5f}, got +/-{np.abs(cb1[0]):.5f}",
)
cb2 = get_turboquant_codebook(2, D)
expected_b2 = np.array([-1.51, -0.453, 0.453, 1.51]) / np.sqrt(D)
all_pass &= check(
    "b=2 centroids match paper page 10",
    np.max(np.abs(np.sort(cb2) - expected_b2)) < 5e-3,
    f"max err {np.max(np.abs(np.sort(cb2) - expected_b2)):.5f}",
)

print("\n  MSE on random unit vectors (vs. Lloyd-Max Gaussian reference):")
rng = np.random.default_rng(0)
X = rng.standard_normal((4000, D))
X = X / np.linalg.norm(X, axis=1, keepdims=True)

for bits in [1, 2, 3, 4]:
    X_hat, _ = turboquant_data(X, bits, rotation="random")
    mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
    ref = LLOYD_MAX_GAUSSIAN_MSE[bits]
    rel_err = abs(mse - ref) / ref
    all_pass &= check(
        f"b={bits} MSE (random rotation)",
        rel_err < 0.05,
        f"got {mse:.5f}, ref {ref:.5f}, rel err {rel_err*100:.1f}%",
    )

print("\n  WHT variant - should match random-rotation MSE on isotropic input:")
for bits in [2, 3, 4]:
    X_hat_r, _ = turboquant_data(X, bits, rotation="random")
    X_hat_w, _ = turboquant_data(X, bits, rotation="wht")
    mse_r = float(np.mean(np.sum((X - X_hat_r) ** 2, axis=1)))
    mse_w = float(np.mean(np.sum((X - X_hat_w) ** 2, axis=1)))
    all_pass &= check(
        f"b={bits} random vs WHT agree on isotropic input",
        abs(mse_r - mse_w) / mse_r < 0.02,
        f"random {mse_r:.5f}, wht {mse_w:.5f}",
    )

print("\n  Compression ratios (head_dim=128, fp16 norms):")
seq = 1000
test_K = rng.standard_normal((seq, D))
for bits in [2, 3, 4]:
    _, nbytes = turboquant_data(test_K, bits)
    baseline_fp16_bytes = seq * D * 2
    ratio = baseline_fp16_bytes / nbytes
    print(f"      b={bits}: {ratio:.2f}x compression ({nbytes:.0f} bytes vs {baseline_fp16_bytes} fp16)")

print()
print("=" * 72)
print("RESULT:", "ALL CHECKS PASS" if all_pass else "SOME CHECKS FAILED")
print("=" * 72)
