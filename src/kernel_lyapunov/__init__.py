"""
kernel_lyapunov
===============
Generalised d-dimensional kernel/Koopman-eigenfunction Lyapunov function
construction, as described in:

  Giesl, Hafstein, Hamzi, Lee, Owhadi, Santin, Vaidya (2026).
  "Kernel Methods for the Construction of Certified Lyapunov Functions
   via Approximate Koopman Eigenfunctions."

Public API
----------
build_gram_matrix   -- assemble the (n+d+1) × (n+d+1) symmetric kernel-
                       collocation Gram matrix for one eigenfunction solve
solve_eigenfunction  -- solve for the representer coefficients alpha
eval_phi             -- evaluate the learned eigenfunction phi_lambda at
                        arbitrary test points
eval_grad_phi        -- evaluate the analytic gradient ∇phi_lambda (exact,
                        no secondary finite-difference regression needed)
"""

from .kernel import (
    make_kernel_funcs,
    pairwise_diffs,
    build_gram_matrix,
    solve_eigenfunction,
    eval_phi,
    eval_grad_phi,
)

__all__ = [
    "make_kernel_funcs",
    "pairwise_diffs",
    "build_gram_matrix",
    "solve_eigenfunction",
    "eval_phi",
    "eval_grad_phi",
]
