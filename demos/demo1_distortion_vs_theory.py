"""
Demo 1 — Empirical distortion matches Shannon-theoretic bounds.

Story for the audience:
  "We compress unit vectors to b bits per coordinate. The black dashed line
   is the information-theoretic *lower* bound from Shannon source coding;
   the orange line is what *any* possible algorithm could ever achieve
   in the limit. TurboQuant (blue) sits within a small constant of optimum
   for every bit-width, with zero tuning."

Run:
    python -m demos.demo1_distortion_vs_theory
Outputs:
    demos/out/distortion.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from tiny_turboquant.numpy_reference import TurboQuantMSE, TurboQuantProd

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    rng = np.random.default_rng(0)
    d, n = 512, 4000
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    bits = [1, 2, 3, 4, 5, 6]
    mse_emp, mse_lb, mse_ub = [], [], []
    for b in bits:
        q = TurboQuantMSE(d, b, seed=0)
        Xh = q.dequant(q.quant(X))
        mse_emp.append(float(np.mean(np.sum((X - Xh) ** 2, axis=1))))
        mse_lb.append(4.0 ** (-b))                    # Shannon lower bound
        mse_ub.append(3 * np.pi / 2 * 4.0 ** (-b))    # paper Theorem 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bits, mse_ub,  "--",  color="orange", label="Paper upper bound (3π/2·4⁻ᵇ)")
    ax.plot(bits, mse_lb,  "--",  color="black",  label="Shannon lower bound (4⁻ᵇ)")
    ax.plot(bits, mse_emp, "o-",  color="tab:blue", lw=2, label="TurboQuant empirical")
    ax.set_yscale("log")
    ax.set_xlabel("bits per coordinate")
    ax.set_ylabel("MSE  E‖x − x̂‖²")
    ax.set_title(f"TurboQuant-MSE vs information-theoretic bounds  (d={d}, n={n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT, "distortion.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")

    # Also show inner-product unbiasedness — the *killer* feature for ANN/RAG.
    Y = rng.standard_normal((n, d)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    print("\n  bits |   bias   |  ip-MSE   |  ratio vs full-precision")
    print("  -----+----------+-----------+--------------------------")
    for b in (2, 3, 4):
        qp = TurboQuantProd(d, b, seed=0)
        idx, s, g = qp.quant(X)
        Xh = qp.dequant(idx, s, g)
        true = np.sum(X * Y, 1)
        est  = np.sum(Xh * Y, 1)
        bias = float(np.mean(est - true))
        var  = float(np.mean((est - true) ** 2))
        print(f"   {b}   | {bias:+.4f} | {var:.5f}   | "
              f"{32/b:.0f}× smaller than fp32")


if __name__ == "__main__":
    main()
