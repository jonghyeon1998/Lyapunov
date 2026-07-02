"""
Generalised d-dimensional kernel PDE collocation for Koopman eigenfunctions.

Method
------
Given a stable equilibrium at the origin with Jacobian E = Df(0), let
lambda_i be an eigenvalue of E with left eigenvector w_i (i.e. w_i^T E =
lambda_i w_i^T). The corresponding Koopman eigenfunction decomposes as

    phi_lambda(x) = w^T x + h_lambda(x),

where h_lambda solves the linear PDE

    grad(h)(x) . f(x) - lambda * h(x) = -w^T G(x),   h(0) = 0,  grad(h)(0) = 0,

with G(x) = f(x) - E x  the purely nonlinear remainder.

Symmetric kernel collocation (RKHS / representer theorem, Section 4.2 of the
paper) turns this into a linear system of size (n + d + 1) x (n + d + 1):
  rows/cols 0        : point evaluation at the origin  [h(0) = 0]
  rows/cols 1 .. d   : gradient at the origin          [grad(h)(0) = 0]
  rows/cols d+1 .. n : PDE collocation at X_1, ..., X_n

The kernel used is the anisotropic Gaussian

    K(x, y) = exp( -sum_k (x_k - y_k)^2 / (2 sigma_k^2) ).

References
----------
Giesl, Hafstein, Hamzi, Lee, Owhadi, Santin, Vaidya (2026).
"Kernel Methods for the Construction of Certified Lyapunov Functions
 via Approximate Koopman Eigenfunctions."  Section 4.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Kernel and its derivatives
# ---------------------------------------------------------------------------

def pairwise_diffs(X, Y):
    """Return a tuple of d arrays diffs[k] = X[:,k,None] - Y[None,:,k]  (shape n x m)."""
    d = X.shape[1]
    return tuple(X[:, None, k] - Y[None, :, k] for k in range(d))


def make_kernel_funcs(sigmas):
    """
    Return closed-form kernel and partial-derivative functions for the
    anisotropic Gaussian kernel with length-scales *sigmas* (array of length d).

    Returns
    -------
    K    : diffs -> K(x,y)                  [shape (n,m)]
    dx   : (diffs, K, k) -> d/dx_k K(x,y)  [shape (n,m)]
    dy   : (diffs, K, k) -> d/dy_k K(x,y)  [shape (n,m)]
    dxdy : (diffs, K, k, l) -> d^2/(dx_k dy_l) K(x,y)  [shape (n,m)]
    d    : int, dimension

    Here *diffs* is the tuple returned by :func:`pairwise_diffs`.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    d = len(sigmas)

    def K(diffs):
        return np.exp(-sum(diffs[k] ** 2 / (2 * sigmas[k] ** 2) for k in range(d)))

    def dx(diffs, Kval, k):
        return -diffs[k] / sigmas[k] ** 2 * Kval

    def dy(diffs, Kval, k):
        return diffs[k] / sigmas[k] ** 2 * Kval

    def dxdy(diffs, Kval, k, l):
        if k == l:
            return (sigmas[k] ** 2 - diffs[k] ** 2) / sigmas[k] ** 4 * Kval
        else:
            return -diffs[k] * diffs[l] / (sigmas[k] ** 2 * sigmas[l] ** 2) * Kval

    return K, dx, dy, dxdy, d


# ---------------------------------------------------------------------------
# Gram matrix
# ---------------------------------------------------------------------------

def build_gram_matrix(ev, X_train, F_train, sigmas):
    """
    Assemble the symmetric (n+d+1) x (n+d+1) kernel-collocation Gram matrix
    for eigenvalue *ev* and collocation data *(X_train, F_train)*.

    Parameters
    ----------
    ev      : float  -- Koopman eigenvalue lambda_i
    X_train : (n, d) array -- collocation points
    F_train : (n, d) array -- f(X_train)  [full vector field, not just G]
    sigmas  : length-d array -- Gaussian kernel length-scales

    Returns
    -------
    M : (n+d+1, n+d+1) symmetric positive-semi-definite array
    """
    K, dx, dy, dxdy, d = make_kernel_funcs(sigmas)
    n = X_train.shape[0]
    size = n + d + 1

    diffsXX = pairwise_diffs(X_train, X_train)
    KXX = K(diffsXX)

    zero = np.zeros((1, d))
    diffs0X = pairwise_diffs(zero, X_train)      # shape (1, n, d) effectively
    K0X = K(tuple(diff[0] for diff in diffs0X))  # shape (n,)

    M = np.zeros((size, size))
    # 0,0 block: K(0,0) = 1
    M[0, 0] = 1.0
    # gradient-at-origin diagonal: d^2/dx_k dy_k K(0,0) = 1/sigma_k^2
    for k in range(d):
        M[1 + k, 1 + k] = sigmas[k] ** -2

    # cross terms: boundary functionals vs PDE functionals
    dy0X = np.stack([dy(diffs0X, K0X[None, :], k)[0] for k in range(d)], axis=-1)   # (n, d)
    dx0X = np.stack([dx(diffs0X, K0X[None, :], k)[0] for k in range(d)], axis=-1)   # (n, d)

    M[0, d + 1:] = np.sum(dy0X * F_train, axis=-1) - ev * K0X
    for i in range(d):
        d2 = np.stack([dxdy(diffs0X, K0X[None, :], i, k)[0] for k in range(d)], axis=-1)
        M[1 + i, d + 1:] = np.sum(d2 * F_train, axis=-1) - ev * dx0X[:, i]

    # symmetry
    M[d + 1:, 0] = M[0, d + 1:]
    for i in range(d):
        M[d + 1:, 1 + i] = M[1 + i, d + 1:]

    # PDE-PDE block
    Fi = F_train[:, None, :]   # (n, 1, d)
    Fj = F_train[None, :, :]   # (1, n, d)
    term_FF = sum(
        Fi[..., k] * Fj[..., l] * dxdy(diffsXX, KXX, k, l)
        for k in range(d) for l in range(d)
    )
    term_Fi = sum(Fi[..., k] * dx(diffsXX, KXX, k) for k in range(d))
    term_Fj = sum(Fj[..., l] * dy(diffsXX, KXX, l) for l in range(d))
    M[d + 1:, d + 1:] = term_FF - ev * term_Fj - ev * term_Fi + ev ** 2 * KXX

    return M


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve_eigenfunction(ev, X_train, F_train, G_train, w, sigmas, nugget=1e-10):
    """
    Solve for the representer coefficients alpha of h_lambda.

    Parameters
    ----------
    ev      : float  -- eigenvalue lambda_i
    X_train : (n, d) -- collocation points
    F_train : (n, d) -- f(X_train)
    G_train : (n, d) -- G(X_train) = f(X_train) - E @ X_train.T  (nonlinear remainder)
    w       : (d,)   -- left eigenvector of E for eigenvalue ev
    sigmas  : (d,)   -- kernel length-scales
    nugget  : float  -- Tikhonov regularisation added to the diagonal of M

    Returns
    -------
    alpha : (n+d+1,) array of representer coefficients
    """
    M = build_gram_matrix(ev, X_train, F_train, sigmas)
    n, d = X_train.shape
    rhs = np.zeros(n + d + 1)
    rhs[d + 1:] = -(G_train @ w)
    return np.linalg.solve(M + nugget * np.eye(M.shape[0]), rhs)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _representer_pieces(X_test, X_train, ev, F_train, sigmas):
    """Return the three representer pieces (v0, vgrad, vcol) for h_lambda."""
    K, dx, dy, dxdy, d = make_kernel_funcs(sigmas)
    zero = np.zeros((1, d))

    diffsZ0 = pairwise_diffs(X_test, zero)
    Kz0 = K(tuple(diff[:, 0] for diff in diffsZ0))     # (m,)
    v0 = Kz0
    vgrad = [dy(tuple(diff[:, 0] for diff in diffsZ0), Kz0, i) for i in range(d)]

    diffsZX = pairwise_diffs(X_test, X_train)
    KzX = K(diffsZX)
    vcol = (
        sum(F_train[None, :, l] * dy(diffsZX, KzX, l) for l in range(d))
        - ev * KzX
    )
    return v0, vgrad, vcol, d


def eval_phi(X_test, X_train, ev, w, alpha, F_train, sigmas):
    """
    Evaluate the learned Koopman eigenfunction

        phi_lambda(x) = w^T x + h_lambda(x)

    at the rows of *X_test*.

    Parameters
    ----------
    X_test  : (m, d) -- evaluation points
    X_train : (n, d) -- training/collocation points used when solving
    ev      : float  -- eigenvalue lambda_i
    w       : (d,)   -- left eigenvector
    alpha   : (n+d+1,) -- representer coefficients from solve_eigenfunction
    F_train : (n, d) -- f(X_train) used when solving
    sigmas  : (d,)   -- kernel length-scales

    Returns
    -------
    phi : (m,) array
    """
    v0, vgrad, vcol, d = _representer_pieces(X_test, X_train, ev, F_train, sigmas)
    h = alpha[0] * v0 + sum(alpha[1 + i] * vgrad[i] for i in range(d)) + vcol @ alpha[d + 1:]
    return h + X_test @ w


def eval_grad_phi(X_test, X_train, ev, w, alpha, F_train, sigmas):
    """
    Evaluate the analytic gradient ∇_x phi_lambda(x) at the rows of *X_test*.

    This differentiates the representer formula directly -- no secondary kernel
    regression or finite-difference approximation is needed.

    Returns
    -------
    grad_phi : (m, d) array  -- each row is ∇phi at the corresponding test point
    """
    K, dx, dy, dxdy, d = make_kernel_funcs(sigmas)
    m = X_test.shape[0]
    n = X_train.shape[0]
    zero = np.zeros((1, d))

    diffsZ0 = pairwise_diffs(X_test, zero)
    Kz0 = K(tuple(diff[:, 0] for diff in diffsZ0))
    # gradient of v0 = K(z,0) wrt z
    dv0 = np.stack([dx(tuple(diff[:, 0] for diff in diffsZ0), Kz0, k) for k in range(d)], axis=-1)
    # gradient of vgrad[i] = d/dy_i K(z,0) wrt z
    dvgrad = [
        np.stack([dxdy(tuple(diff[:, 0] for diff in diffsZ0), Kz0, k, i) for k in range(d)], axis=-1)
        for i in range(d)
    ]

    diffsZX = pairwise_diffs(X_test, X_train)
    KzX = K(diffsZX)
    # gradient of vcol = sum_l F_l dy_l K(z, X_j) - ev K(z, X_j)  wrt z
    dvcol = np.zeros((m, n, d))
    for k in range(d):
        dvcol[:, :, k] = (
            sum(F_train[None, :, l] * dxdy(diffsZX, KzX, k, l) for l in range(d))
            - ev * dx(diffsZX, KzX, k)
        )

    dh = (
        alpha[0] * dv0
        + sum(alpha[1 + i] * dvgrad[i] for i in range(d))
        + np.tensordot(dvcol, alpha[d + 1:], axes=([1], [0]))
    )
    return dh + w[None, :]
