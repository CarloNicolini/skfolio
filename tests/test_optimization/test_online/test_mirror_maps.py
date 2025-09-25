import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import dirichlet

from skfolio.optimization.online._mirror_maps import (
    AdaptiveLogBarrierMap,
    AdaptiveMahalanobisMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
    LogBarrierMap,
    TsallisMirrorMap,
)


# Helper functions
def is_positive_definite(H, tol=1e-10):
    """Check if matrix H is positive definite (all eigenvalues > 0)."""
    eigvals = np.linalg.eigvalsh(H)
    return np.all(eigvals > tol)


def finite_diff_jacobian(func, x, eps=1e-8):
    """Finite difference approximation of Jacobian of func at x."""
    n = x.shape[0]
    J = np.zeros((n, n))
    x_eps = x.copy()
    for i in range(n):
        x_eps[i] += eps
        J[:, i] = (func(x_eps) - func(x)) / eps
        x_eps[i] -= eps
    return J


def tangent_vectors(n, num_samples=10):
    """Generate random tangent vectors for simplex (sum(v) = 0)."""
    vectors = []
    for _ in range(num_samples):
        v = np.random.randn(n)
        v -= np.mean(v)  # project to tangent space
        vectors.append(v)
    return np.array(vectors)


def sample_interior_simplex(n, num_samples=1, alpha=1.0):
    """Sample strictly interior points on simplex using Dirichlet."""
    samples = dirichlet.rvs([alpha] * n, size=num_samples)
    return samples


def sample_interior_orthant(n, num_samples=1):
    """Sample strictly positive points in orthant."""
    return np.random.exponential(size=(num_samples, n)) + 1e-6


def near_boundary_simplex(n, coord_idx=0, closeness=1e-6):
    """Sample point near boundary (small value in coord_idx)."""
    w = sample_interior_simplex(n, 1)[0]
    w[coord_idx] = closeness
    w /= np.sum(w)
    return w


def get_map_domain(map_obj):
    """Determine domain of mirror map."""
    if isinstance(map_obj, EntropyMirrorMap | TsallisMirrorMap):
        return "simplex"
    elif isinstance(map_obj, LogBarrierMap | AdaptiveLogBarrierMap):
        return "orthant"
    else:
        return "unconstrained"


def simulate_adaptive_updates(map_obj, n=3, num_updates=10):
    """Simulate gradient updates for adaptive maps."""
    if hasattr(map_obj, "update_state"):
        grads = [np.random.randn(n) for _ in range(num_updates)]
        for g in grads:
            map_obj.update_state(g)


# Test strict convexity via Hessian positive definiteness
@pytest.mark.parametrize(
    "mirror_map_cls",
    [
        EuclideanMirrorMap,
        EntropyMirrorMap,
        LogBarrierMap,
        # Skip Tsallis q<1 as it's not convex, q=1 is entropy limit
        AdaptiveMahalanobisMap,
        # AdaptiveLogBarrierMap needs updates - test separately
    ],
)
def test_hessian_positive_definite(mirror_map_cls):
    n = 5
    map_obj = mirror_map_cls()

    # Simulate updates for adaptive maps
    simulate_adaptive_updates(map_obj, n)

    domain = get_map_domain(map_obj)
    if domain == "simplex":
        x = sample_interior_simplex(n, 1)[0]
    elif domain == "orthant":
        x = sample_interior_orthant(n, 1)[0]
    else:
        x = np.random.rand(n) + 0.1

    def psi(x):
        return map_obj.grad_psi(x)

    # Finite difference Hessian
    J = finite_diff_jacobian(psi, x)
    assert is_positive_definite(J), (
        f"Hessian must be positive definite for strict convexity. Got eigenvalues: {np.linalg.eigvals(J)}"
    )


def test_adaptive_log_barrier_hessian():
    """Test AdaptiveLogBarrierMap Hessian after updates."""
    n = 3
    map_obj = AdaptiveLogBarrierMap(n, eps=1e-12)
    x = sample_interior_orthant(n, 1)[0]

    # Simulate updates to make _D > 0
    simulate_adaptive_updates(map_obj, n)

    def psi(x):
        return map_obj.grad_psi(x)

    J = finite_diff_jacobian(psi, x)
    assert is_positive_definite(J), "AdaptiveLogBarrier should be convex after updates"


# Test essential smoothness: ||grad_psi(x)|| → ∞ as x → boundary
@pytest.mark.parametrize(
    "mirror_map_cls, domain",
    [
        (EntropyMirrorMap, "simplex"),
        (LogBarrierMap, "orthant"),
        # Skip Tsallis as q<1 is not strictly convex, q=1 is entropy
        # AdaptiveLogBarrierMap tested separately after updates
    ],
)
def test_essential_smoothness(mirror_map_cls, domain):
    map_obj = mirror_map_cls()
    n = 3

    if domain == "simplex":
        x_interior = sample_interior_simplex(n, 1)[0]
        grad_interior = map_obj.grad_psi(x_interior)
        norm_interior = np.linalg.norm(grad_interior)

        x_boundary = near_boundary_simplex(n, coord_idx=0, closeness=1e-10)
        grad_boundary = map_obj.grad_psi(x_boundary)
        norm_boundary = np.linalg.norm(grad_boundary)

        assert norm_boundary > norm_interior + 1e-5, (
            "Gradient norm should increase near boundary"
        )
    else:  # orthant
        x_interior = sample_interior_orthant(n, 1)[0]
        grad_interior = map_obj.grad_psi(x_interior)
        norm_interior = np.linalg.norm(grad_interior)

        x_near_zero = x_interior.copy()
        x_near_zero[0] = 1e-10
        grad_near_zero = map_obj.grad_psi(x_near_zero)
        norm_near_zero = np.linalg.norm(grad_near_zero)

        assert norm_near_zero > norm_interior + 1e5, (
            "Gradient should blow up as w_i → 0+ for barrier"
        )


def test_adaptive_log_barrier_smoothness():
    """Test AdaptiveLogBarrierMap smoothness after updates."""
    n = 3
    map_obj = AdaptiveLogBarrierMap(n, eps=1e-12)

    # Simulate updates
    simulate_adaptive_updates(map_obj, n)

    x_interior = sample_interior_orthant(n, 1)[0]
    grad_interior = map_obj.grad_psi(x_interior)
    norm_interior = np.linalg.norm(grad_interior)

    x_near_zero = x_interior.copy()
    x_near_zero[0] = 1e-10
    grad_near_zero = map_obj.grad_psi(x_near_zero)
    norm_near_zero = np.linalg.norm(grad_near_zero)

    assert norm_near_zero > norm_interior + 1e5, (
        "AdaptiveLogBarrier gradient should blow up near boundary after updates"
    )


# Test invertibility: compositions ≈ identity
@pytest.mark.parametrize(
    "mirror_map_cls",
    [
        EuclideanMirrorMap,
        EntropyMirrorMap,
        LogBarrierMap,
        # Skip Tsallis q>1 not supported, q<1 not convex, q=1 is entropy
    ],
)
def test_compositions_invertibility(mirror_map_cls):
    map_obj = mirror_map_cls()
    n = 5
    num_samples = 10
    tol = 1e-10

    domain = get_map_domain(map_obj)
    if domain == "simplex":
        samples = sample_interior_simplex(n, num_samples)
        for x in samples:
            psi_x = map_obj.grad_psi(x)
            x_reconstructed = map_obj.grad_psi_star(psi_x)
            assert_allclose(x_reconstructed, x, atol=tol, rtol=0)

            # Dual: handle simplex invariance (add constant doesn't change softmax)
            theta = psi_x + np.random.uniform(-1, 1, n)  # perturbed dual
            psi_star_theta = map_obj.grad_psi_star(theta)
            psi_reconstructed = map_obj.grad_psi(psi_star_theta)
            # Subtract mean to handle additive invariance
            assert_allclose(
                psi_reconstructed - np.mean(psi_reconstructed),
                theta - np.mean(theta),
                atol=tol,
                rtol=0,
            )
    else:  # Euclidean or orthant
        samples = (
            sample_interior_orthant(n, num_samples)
            if domain == "orthant"
            else [np.random.rand(n) + 0.1 for _ in range(num_samples)]
        )
        for x in samples:
            psi_x = map_obj.grad_psi(x)
            x_reconstructed = map_obj.grad_psi_star(psi_x)
            assert_allclose(x_reconstructed, x, atol=tol, rtol=0)

            theta = psi_x + np.random.randn(n) * 0.1
            psi_star_theta = map_obj.grad_psi_star(theta)
            psi_reconstructed = map_obj.grad_psi(psi_star_theta)
            # LogBarrier can have numerical issues with very small values, use relaxed tolerance
            test_tol = 1e-8 if domain == "orthant" else tol
            assert_allclose(psi_reconstructed, theta, atol=test_tol, rtol=0)


# Test Jacobian product ≈ I on tangent space (for constrained domains)
@pytest.mark.parametrize(
    "mirror_map_cls",
    [
        EntropyMirrorMap,
        # Skip Tsallis - q>1 not supported, q<1 not convex
    ],
)
def test_jacobian_product_tangent(mirror_map_cls):
    map_obj = mirror_map_cls()
    n = 5
    x = sample_interior_simplex(n, 1)[0]
    tol = 1e-6  # Relaxed tolerance for finite differences

    def psi(x):
        return map_obj.grad_psi(x)

    def psi_star(theta):
        return map_obj.grad_psi_star(theta)

    J_psi = finite_diff_jacobian(psi, x)
    theta = psi(x)
    J_psi_star = finite_diff_jacobian(psi_star, theta)

    # Product
    J_product = J_psi_star @ J_psi

    # Test on tangent vectors (sum=0)
    tangents = tangent_vectors(n, num_samples=5)
    for v in tangents:
        Jv = J_product @ v
        assert_allclose(Jv, v, atol=tol, rtol=0)


# Test Adaptive maps after updates
def test_adaptive_mahalanobis():
    n = 3
    map_obj = AdaptiveMahalanobisMap(n, eps=1e-8)
    x = np.random.rand(n) + 0.1

    # Simulate updates
    simulate_adaptive_updates(map_obj, n)

    # Test composition after adaptation
    psi_x = map_obj.grad_psi(x)
    x_rec = map_obj.grad_psi_star(psi_x)
    assert_allclose(x_rec, x, atol=1e-8, rtol=0)

    # Hessian should be diagonal positive definite
    def psi(t):
        return map_obj.grad_psi(t)

    J = finite_diff_jacobian(psi, x)
    assert is_positive_definite(J)


def test_adaptive_log_barrier():
    n = 3
    map_obj = AdaptiveLogBarrierMap(n, eps=1e-12)
    x = sample_interior_orthant(n, 1)[0]

    # Simulate updates
    simulate_adaptive_updates(map_obj, n)

    # Test composition
    psi_x = map_obj.grad_psi(x)
    x_rec = map_obj.grad_psi_star(psi_x)
    assert_allclose(x_rec, x, atol=1e-8, rtol=0)


# Test Tsallis for q=1 (entropy limit)
def test_tsallis_entropy_limit():
    """Test that Tsallis with q=1 behaves like entropy."""
    map_obj = TsallisMirrorMap(q=1.0)
    entropy_map = EntropyMirrorMap()
    n = 4
    x = sample_interior_simplex(n, 1)[0]
    tol = 1e-10

    # Should delegate to entropy map
    psi_x = map_obj.grad_psi(x)
    psi_x_entropy = entropy_map.grad_psi(x)
    assert_allclose(psi_x, psi_x_entropy, atol=tol, rtol=0)

    x_rec = map_obj.grad_psi_star(psi_x)
    assert_allclose(x_rec, x, atol=tol, rtol=0)


# Test project_geom preserves properties
def test_project_geom_entropy():
    """Test EntropyMirrorMap project_geom on simplex."""
    map_obj = EntropyMirrorMap()
    n = 3
    x = sample_interior_simplex(n, 1)[0]  # Already normalized

    projected = map_obj.project_geom(x)
    assert_allclose(np.sum(projected), 1.0, atol=1e-12)
    assert np.all(projected > 0)
    # Should be close to input since already valid
    assert_allclose(projected, x, atol=1e-10, rtol=0)


def test_project_geom_log_barrier():
    """Test LogBarrierMap project_geom behavior."""
    map_obj = LogBarrierMap()
    n = 3
    x = sample_interior_orthant(n, 1)[0]  # Positive but not normalized

    projected = map_obj.project_geom(x)
    # LogBarrier normalizes to simplex
    assert_allclose(np.sum(projected), 1.0, atol=1e-12)
    assert np.all(projected > 0)
    # Should NOT be equal to x since it normalizes


def test_project_geom_tsallis():
    """Test TsallisMirrorMap project_geom on simplex."""
    map_obj = TsallisMirrorMap(q=1.0)  # Use q=1 (entropy limit)
    n = 3
    x = sample_interior_simplex(n, 1)[0]  # Already normalized

    projected = map_obj.project_geom(x)
    assert_allclose(np.sum(projected), 1.0, atol=1e-12)
    assert np.all(projected > 0)
    # Should be close to input since already valid
    assert_allclose(projected, x, atol=1e-10, rtol=0)


# Test Tsallis edge case q<1 (should fail or warn)
def test_tsallis_q_less_than_one_warning():
    """Test that Tsallis with q<1 behaves as expected (non-convex)."""
    # This documents the current behavior - q<1 is allowed but not convex
    map_obj = TsallisMirrorMap(q=0.7)
    n = 3
    x = sample_interior_simplex(n, 1)[0]

    # Should still work for composition (implementation handles it)
    psi_x = map_obj.grad_psi(x)
    x_rec = map_obj.grad_psi_star(psi_x)
    # Composition may work even if not convex
    assert_allclose(x_rec, x, atol=1e-8, rtol=0)

    # But Hessian should be negative (non-convex)
    def psi(x):
        return map_obj.grad_psi(x)

    J = finite_diff_jacobian(psi, x)
    assert not is_positive_definite(J), "q<1 should not be convex"
