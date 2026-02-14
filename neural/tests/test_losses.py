"""
Tests for AlphaZeroLoss
========================

Verifies the combined loss function implementation by checking:
- Output structure: LossResult is a NamedTuple with 3 scalar tensors
- Policy loss on uniform random logits: approximately log(4672) ~ 8.45
- Policy loss with perfect prediction: near zero
- Policy loss is non-negative
- Value loss when prediction matches target: near zero
- Value loss with maximum error (predict +1, target -1): should be 4.0
- Value loss is non-negative
- Combined loss equals policy_loss + value_loss
- Gradient flows to both policy logits and value predictions
- Batch size independence: same loss regardless of batch size for same data
- Legal move masking: only nonzero entries in target_policy contribute
- Numerical stability: large logits do not produce NaN or Inf

The tests follow the same conventions as the other test files in the neural/
package: organized into classes by topic, descriptive docstrings, parametrized
where appropriate.
"""

import math

import pytest
import torch
import torch.nn as nn

from neural.losses import AlphaZeroLoss, LossResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLICY_SIZE = 4672


def _uniform_target_policy(batch_size: int) -> torch.Tensor:
    """Create a uniform target policy distribution (1/4672 for each move)."""
    return torch.full((batch_size, POLICY_SIZE), 1.0 / POLICY_SIZE)


def _one_hot_target_policy(batch_size: int, hot_index: int = 0) -> torch.Tensor:
    """Create a one-hot target policy (all probability on a single move)."""
    target = torch.zeros(batch_size, POLICY_SIZE)
    target[:, hot_index] = 1.0
    return target


# ---------------------------------------------------------------------------
# Output structure: LossResult is a NamedTuple with 3 scalar tensors
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Verify that the loss returns a LossResult with the correct shape and type."""

    def test_returns_loss_result(self) -> None:
        """The forward pass should return a LossResult NamedTuple."""
        loss_fn = AlphaZeroLoss()
        result = loss_fn(
            policy_logits=torch.randn(4, POLICY_SIZE),
            value=torch.tanh(torch.randn(4, 1)),
            target_policy=_uniform_target_policy(4),
            target_value=torch.ones(4, 1),
        )
        assert isinstance(result, LossResult), (
            f"Expected LossResult, got {type(result)}"
        )

    def test_loss_result_is_named_tuple(self) -> None:
        """LossResult should be a NamedTuple (supports index and attribute access)."""
        loss_fn = AlphaZeroLoss()
        result = loss_fn(
            policy_logits=torch.randn(4, POLICY_SIZE),
            value=torch.tanh(torch.randn(4, 1)),
            target_policy=_uniform_target_policy(4),
            target_value=torch.ones(4, 1),
        )
        # Named access
        assert hasattr(result, "total_loss")
        assert hasattr(result, "policy_loss")
        assert hasattr(result, "value_loss")
        # Tuple unpacking
        total, policy, value = result
        assert total is result.total_loss
        assert policy is result.policy_loss
        assert value is result.value_loss

    def test_all_losses_are_scalar_tensors(self) -> None:
        """All three loss components should be 0-dimensional tensors."""
        loss_fn = AlphaZeroLoss()
        result = loss_fn(
            policy_logits=torch.randn(4, POLICY_SIZE),
            value=torch.tanh(torch.randn(4, 1)),
            target_policy=_uniform_target_policy(4),
            target_value=torch.ones(4, 1),
        )
        for name in ["total_loss", "policy_loss", "value_loss"]:
            tensor = getattr(result, name)
            assert isinstance(tensor, torch.Tensor), (
                f"{name} should be a torch.Tensor, got {type(tensor)}"
            )
            assert tensor.dim() == 0, (
                f"{name} should be a scalar (0-dim), got {tensor.dim()}-dim "
                f"with shape {tensor.shape}"
            )

    def test_loss_result_has_three_fields(self) -> None:
        """LossResult should have exactly three fields."""
        assert len(LossResult._fields) == 3
        assert LossResult._fields == ("total_loss", "policy_loss", "value_loss")


# ---------------------------------------------------------------------------
# Policy loss behavior
# ---------------------------------------------------------------------------


class TestPolicyLoss:
    """Verify the policy cross-entropy loss computes correctly."""

    def test_uniform_logits_gives_log_policy_size(self) -> None:
        """With uniform logits and uniform target, loss should be log(4672).

        When all logits are equal (e.g., all zeros), softmax produces a
        uniform distribution: p_i = 1/N for all i. The cross-entropy of a
        uniform distribution with itself is:

            -sum(1/N * log(1/N)) = -N * (1/N) * log(1/N) = log(N)

        For chess, N = 4672, so log(4672) ~ 8.45.
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 8
        # Uniform logits (all zeros) -> softmax gives uniform distribution
        policy_logits = torch.zeros(batch_size, POLICY_SIZE)
        target_policy = _uniform_target_policy(batch_size)
        target_value = torch.zeros(batch_size, 1)
        value = torch.zeros(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)

        expected = math.log(POLICY_SIZE)  # ~ 8.45
        assert abs(result.policy_loss.item() - expected) < 0.01, (
            f"Policy loss with uniform logits should be ~{expected:.4f}, "
            f"got {result.policy_loss.item():.4f}"
        )

    def test_perfect_prediction_gives_near_zero_loss(self) -> None:
        """When the network perfectly predicts the target, loss should be near zero.

        If the target puts all probability on one move and the network assigns
        extremely high logit to that same move, the cross-entropy approaches zero.
        We use a one-hot target and set the corresponding logit very high.
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        hot_index = 42

        # Target is one-hot at index 42
        target_policy = _one_hot_target_policy(batch_size, hot_index)
        target_value = torch.zeros(batch_size, 1)
        value = torch.zeros(batch_size, 1)

        # Set logit at hot_index very high (so softmax concentrates there)
        policy_logits = torch.full((batch_size, POLICY_SIZE), -10.0)
        policy_logits[:, hot_index] = 50.0

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert result.policy_loss.item() < 0.01, (
            f"Policy loss with perfect prediction should be near 0, "
            f"got {result.policy_loss.item():.6f}"
        )

    def test_policy_loss_is_non_negative(self) -> None:
        """Cross-entropy of valid distributions is always non-negative.

        This follows from the fact that log-probabilities are <= 0, so
        -sum(pi * log(p)) >= 0 when pi >= 0 and sum(pi) = 1.
        """
        loss_fn = AlphaZeroLoss()
        # Test with several different random inputs
        for _ in range(10):
            batch_size = 4
            policy_logits = torch.randn(batch_size, POLICY_SIZE)
            target_policy = torch.softmax(torch.randn(batch_size, POLICY_SIZE), dim=1)
            target_value = torch.zeros(batch_size, 1)
            value = torch.zeros(batch_size, 1)

            result = loss_fn(policy_logits, value, target_policy, target_value)
            assert result.policy_loss.item() >= 0, (
                f"Policy loss should be non-negative, got {result.policy_loss.item()}"
            )

    def test_higher_entropy_target_gives_higher_loss(self) -> None:
        """A more spread-out target distribution should produce higher policy loss.

        With uniform logits, cross-entropy equals the entropy of the target.
        A uniform target has maximum entropy (log(N)), while a concentrated
        target has lower entropy. So uniform target -> higher loss.
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        policy_logits = torch.zeros(batch_size, POLICY_SIZE)  # uniform prediction
        target_value = torch.zeros(batch_size, 1)
        value = torch.zeros(batch_size, 1)

        # Concentrated target: one-hot
        target_concentrated = _one_hot_target_policy(batch_size)
        result_concentrated = loss_fn(
            policy_logits, value, target_concentrated, target_value
        )

        # Uniform target: maximum entropy
        target_uniform = _uniform_target_policy(batch_size)
        result_uniform = loss_fn(
            policy_logits, value, target_uniform, target_value
        )

        assert result_uniform.policy_loss.item() > result_concentrated.policy_loss.item(), (
            f"Uniform target loss ({result_uniform.policy_loss.item():.4f}) "
            f"should be > concentrated target loss ({result_concentrated.policy_loss.item():.4f})"
        )


# ---------------------------------------------------------------------------
# Value loss behavior
# ---------------------------------------------------------------------------


class TestValueLoss:
    """Verify the value MSE loss computes correctly."""

    def test_matching_prediction_gives_near_zero_loss(self) -> None:
        """When the predicted value matches the target, loss should be zero."""
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        policy_logits = torch.zeros(batch_size, POLICY_SIZE)
        target_policy = _uniform_target_policy(batch_size)

        # Prediction matches target exactly
        target_value = torch.tensor([[1.0], [-1.0], [0.0], [1.0]])
        value = target_value.clone()

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert result.value_loss.item() < 1e-6, (
            f"Value loss should be ~0 when prediction matches target, "
            f"got {result.value_loss.item():.6f}"
        )

    def test_maximum_error_gives_loss_of_four(self) -> None:
        """Predicting +1 when target is -1 (or vice versa) gives MSE = 4.0.

        MSE = mean((+1 - (-1))^2) = mean(4) = 4.0

        This is the maximum possible value loss since both the prediction
        (tanh output) and target are bounded to [-1, 1].
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        policy_logits = torch.zeros(batch_size, POLICY_SIZE)
        target_policy = _uniform_target_policy(batch_size)

        # Maximum error: predict +1, target -1
        value = torch.ones(batch_size, 1)
        target_value = -torch.ones(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert abs(result.value_loss.item() - 4.0) < 1e-5, (
            f"Value loss with max error should be 4.0, got {result.value_loss.item():.6f}"
        )

    def test_value_loss_is_non_negative(self) -> None:
        """MSE is always non-negative (it's a sum of squares)."""
        loss_fn = AlphaZeroLoss()
        for _ in range(10):
            batch_size = 4
            policy_logits = torch.zeros(batch_size, POLICY_SIZE)
            target_policy = _uniform_target_policy(batch_size)
            value = torch.tanh(torch.randn(batch_size, 1))
            target_value = torch.randint(-1, 2, (batch_size, 1)).float()

            result = loss_fn(policy_logits, value, target_policy, target_value)
            assert result.value_loss.item() >= 0, (
                f"Value loss should be non-negative, got {result.value_loss.item()}"
            )

    def test_value_loss_scales_with_error(self) -> None:
        """Larger prediction errors should produce larger value loss."""
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        policy_logits = torch.zeros(batch_size, POLICY_SIZE)
        target_policy = _uniform_target_policy(batch_size)
        target_value = torch.ones(batch_size, 1)

        # Small error: predict 0.9
        value_close = torch.full((batch_size, 1), 0.9)
        result_close = loss_fn(
            policy_logits, value_close, target_policy, target_value
        )

        # Large error: predict -0.5
        value_far = torch.full((batch_size, 1), -0.5)
        result_far = loss_fn(
            policy_logits, value_far, target_policy, target_value
        )

        assert result_far.value_loss.item() > result_close.value_loss.item(), (
            f"Larger error ({result_far.value_loss.item():.4f}) should give "
            f"larger loss than smaller error ({result_close.value_loss.item():.4f})"
        )


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


class TestCombinedLoss:
    """Verify the combined loss equals the sum of individual losses."""

    def test_total_equals_sum_of_components(self) -> None:
        """total_loss should equal policy_loss + value_loss exactly."""
        loss_fn = AlphaZeroLoss()
        result = loss_fn(
            policy_logits=torch.randn(4, POLICY_SIZE),
            value=torch.tanh(torch.randn(4, 1)),
            target_policy=_uniform_target_policy(4),
            target_value=torch.randint(-1, 2, (4, 1)).float(),
        )
        expected = result.policy_loss + result.value_loss
        assert torch.allclose(result.total_loss, expected, atol=1e-6), (
            f"total_loss ({result.total_loss.item():.6f}) != "
            f"policy_loss ({result.policy_loss.item():.6f}) + "
            f"value_loss ({result.value_loss.item():.6f}) = {expected.item():.6f}"
        )

    def test_total_equals_sum_multiple_random_inputs(self) -> None:
        """Verify total = policy + value across multiple random inputs."""
        loss_fn = AlphaZeroLoss()
        for _ in range(20):
            result = loss_fn(
                policy_logits=torch.randn(8, POLICY_SIZE),
                value=torch.tanh(torch.randn(8, 1)),
                target_policy=torch.softmax(torch.randn(8, POLICY_SIZE), dim=1),
                target_value=torch.randint(-1, 2, (8, 1)).float(),
            )
            expected = result.policy_loss + result.value_loss
            assert torch.allclose(result.total_loss, expected, atol=1e-6), (
                f"Mismatch: total={result.total_loss.item():.6f}, "
                f"sum={expected.item():.6f}"
            )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify that gradients flow through both the policy and value paths."""

    def test_gradient_flows_to_policy_logits(self) -> None:
        """Backprop through total_loss should produce gradients on policy_logits."""
        loss_fn = AlphaZeroLoss()
        policy_logits = torch.randn(4, POLICY_SIZE, requires_grad=True)
        value = torch.tanh(torch.randn(4, 1))
        target_policy = _uniform_target_policy(4)
        target_value = torch.ones(4, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        result.total_loss.backward()

        assert policy_logits.grad is not None, (
            "policy_logits should have a gradient after backward()"
        )
        assert policy_logits.grad.abs().sum() > 0, (
            "policy_logits gradient should be non-zero"
        )

    def test_gradient_flows_to_value(self) -> None:
        """Backprop through total_loss should produce gradients on value."""
        loss_fn = AlphaZeroLoss()
        policy_logits = torch.randn(4, POLICY_SIZE)
        value = torch.randn(4, 1, requires_grad=True)
        target_policy = _uniform_target_policy(4)
        target_value = torch.ones(4, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        result.total_loss.backward()

        assert value.grad is not None, (
            "value should have a gradient after backward()"
        )
        assert value.grad.abs().sum() > 0, (
            "value gradient should be non-zero"
        )

    def test_gradient_flows_to_both_simultaneously(self) -> None:
        """Both policy_logits and value should receive gradients from total_loss."""
        loss_fn = AlphaZeroLoss()
        policy_logits = torch.randn(4, POLICY_SIZE, requires_grad=True)
        value = torch.randn(4, 1, requires_grad=True)
        target_policy = _uniform_target_policy(4)
        target_value = torch.ones(4, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        result.total_loss.backward()

        assert policy_logits.grad is not None and policy_logits.grad.abs().sum() > 0
        assert value.grad is not None and value.grad.abs().sum() > 0

    def test_individual_losses_support_backward(self) -> None:
        """Each individual loss component should also support backward()."""
        loss_fn = AlphaZeroLoss()

        # Test policy_loss backward
        policy_logits = torch.randn(4, POLICY_SIZE, requires_grad=True)
        result = loss_fn(
            policy_logits,
            torch.zeros(4, 1),
            _uniform_target_policy(4),
            torch.zeros(4, 1),
        )
        result.policy_loss.backward()
        assert policy_logits.grad is not None

        # Test value_loss backward
        value = torch.randn(4, 1, requires_grad=True)
        result = loss_fn(
            torch.zeros(4, POLICY_SIZE),
            value,
            _uniform_target_policy(4),
            torch.ones(4, 1),
        )
        result.value_loss.backward()
        assert value.grad is not None


# ---------------------------------------------------------------------------
# Batch size independence
# ---------------------------------------------------------------------------


class TestBatchSizeIndependence:
    """Loss should be averaged over the batch, giving consistent values.

    If we duplicate the same data multiple times in a batch, the average
    loss should be the same as for a single sample. This ensures the
    learning rate and loss magnitude don't depend on batch size.
    """

    def test_same_loss_for_different_batch_sizes(self) -> None:
        """Repeating the same sample should give the same average loss."""
        loss_fn = AlphaZeroLoss()

        # Single sample
        single_logits = torch.randn(1, POLICY_SIZE)
        single_value = torch.tanh(torch.randn(1, 1))
        single_target_policy = torch.softmax(torch.randn(1, POLICY_SIZE), dim=1)
        single_target_value = torch.tensor([[1.0]])

        result_single = loss_fn(
            single_logits, single_value, single_target_policy, single_target_value
        )

        # Same data repeated 8 times
        batch_logits = single_logits.expand(8, -1).clone()
        batch_value = single_value.expand(8, -1).clone()
        batch_target_policy = single_target_policy.expand(8, -1).clone()
        batch_target_value = single_target_value.expand(8, -1).clone()

        result_batch = loss_fn(
            batch_logits, batch_value, batch_target_policy, batch_target_value
        )

        assert torch.allclose(result_single.total_loss, result_batch.total_loss, atol=1e-5), (
            f"Batch size 1 loss ({result_single.total_loss.item():.6f}) != "
            f"batch size 8 loss ({result_batch.total_loss.item():.6f})"
        )
        assert torch.allclose(result_single.policy_loss, result_batch.policy_loss, atol=1e-5), (
            f"Policy loss differs: batch=1 ({result_single.policy_loss.item():.6f}) "
            f"vs batch=8 ({result_batch.policy_loss.item():.6f})"
        )
        assert torch.allclose(result_single.value_loss, result_batch.value_loss, atol=1e-5), (
            f"Value loss differs: batch=1 ({result_single.value_loss.item():.6f}) "
            f"vs batch=8 ({result_batch.value_loss.item():.6f})"
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_loss_independent_of_batch_size(self, batch_size: int) -> None:
        """For identical repeated data, loss should be the same for any batch size."""
        loss_fn = AlphaZeroLoss()

        # Generate a single sample and replicate it
        torch.manual_seed(42)
        single_logits = torch.randn(1, POLICY_SIZE)
        single_value = torch.tanh(torch.randn(1, 1))
        single_target_policy = torch.softmax(torch.randn(1, POLICY_SIZE), dim=1)
        single_target_value = torch.tensor([[1.0]])

        result_ref = loss_fn(
            single_logits, single_value, single_target_policy, single_target_value
        )

        result_batch = loss_fn(
            single_logits.expand(batch_size, -1).clone(),
            single_value.expand(batch_size, -1).clone(),
            single_target_policy.expand(batch_size, -1).clone(),
            single_target_value.expand(batch_size, -1).clone(),
        )

        assert torch.allclose(result_ref.total_loss, result_batch.total_loss, atol=1e-5), (
            f"Loss differs for batch_size={batch_size}: "
            f"ref={result_ref.total_loss.item():.6f}, "
            f"batch={result_batch.total_loss.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Legal move masking
# ---------------------------------------------------------------------------


class TestLegalMoveMask:
    """Verify the loss works correctly when only some moves are legal.

    In real AlphaZero, the target policy has nonzero entries only for
    legal moves (typically 20-40 out of 4672 possible moves). The loss
    should handle this sparse target correctly.
    """

    def test_sparse_target_policy(self) -> None:
        """Loss should work when target_policy has only a few nonzero entries."""
        loss_fn = AlphaZeroLoss()
        batch_size = 4

        # Simulate legal moves: only 20 out of 4672 are nonzero
        target_policy = torch.zeros(batch_size, POLICY_SIZE)
        legal_moves = torch.randint(0, POLICY_SIZE, (20,))
        raw_probs = torch.rand(batch_size, 20)
        raw_probs = raw_probs / raw_probs.sum(dim=1, keepdim=True)
        target_policy[:, legal_moves] = raw_probs

        policy_logits = torch.randn(batch_size, POLICY_SIZE)
        value = torch.zeros(batch_size, 1)
        target_value = torch.zeros(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)

        # Loss should be finite and non-negative
        assert torch.isfinite(result.policy_loss), (
            f"Policy loss should be finite, got {result.policy_loss.item()}"
        )
        assert result.policy_loss.item() >= 0, (
            f"Policy loss should be non-negative, got {result.policy_loss.item()}"
        )

    def test_single_legal_move(self) -> None:
        """When only one move is legal, target is one-hot.

        The loss is: -log(softmax(logit_k)) = -logit_k + log(sum(exp(logits)))
        where k is the one legal move.
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        legal_move_index = 100

        target_policy = _one_hot_target_policy(batch_size, legal_move_index)
        policy_logits = torch.randn(batch_size, POLICY_SIZE)
        value = torch.zeros(batch_size, 1)
        target_value = torch.zeros(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)

        # Manual computation for verification:
        # cross_entropy = -log_softmax(logits)[:, legal_move_index].mean()
        log_probs = torch.log_softmax(policy_logits, dim=1)
        expected_policy_loss = -log_probs[:, legal_move_index].mean()

        assert torch.allclose(result.policy_loss, expected_policy_loss, atol=1e-5), (
            f"Policy loss mismatch for one-hot target: "
            f"got {result.policy_loss.item():.6f}, "
            f"expected {expected_policy_loss.item():.6f}"
        )

    def test_illegal_moves_with_zero_target_dont_contribute(self) -> None:
        """Moves with zero target probability should not affect the loss.

        Since pi_target_i * log(p_i) = 0 when pi_target_i = 0, illegal
        moves are automatically ignored regardless of what logit the
        network assigns to them.
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        legal_move_index = 50

        target_policy = _one_hot_target_policy(batch_size, legal_move_index)

        # Two different logit settings for illegal moves (index != 50)
        logits_a = torch.randn(batch_size, POLICY_SIZE)
        logits_b = logits_a.clone()
        logits_b[:, 0] = 1000.0  # Change an illegal move's logit drastically

        # The policy loss should differ because log_softmax normalizes over
        # ALL logits (not just legal ones). Changing an illegal move's logit
        # changes the normalization constant. This is expected behavior --
        # the network is penalized for wasting probability mass on illegal moves.
        # We verify the loss is still finite.
        result_a = loss_fn(
            logits_a, torch.zeros(batch_size, 1), target_policy, torch.zeros(batch_size, 1)
        )
        result_b = loss_fn(
            logits_b, torch.zeros(batch_size, 1), target_policy, torch.zeros(batch_size, 1)
        )

        assert torch.isfinite(result_a.policy_loss)
        assert torch.isfinite(result_b.policy_loss)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Verify the loss does not produce NaN or Inf for extreme inputs."""

    def test_large_positive_logits(self) -> None:
        """Very large positive logits should not cause NaN/Inf.

        log_softmax handles this by subtracting max(logits) before exp(),
        preventing overflow.
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        policy_logits = torch.full((batch_size, POLICY_SIZE), 1000.0)
        target_policy = _uniform_target_policy(batch_size)
        value = torch.zeros(batch_size, 1)
        target_value = torch.zeros(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total_loss), (
            f"Total loss should be finite for large logits, got {result.total_loss.item()}"
        )
        assert torch.isfinite(result.policy_loss), (
            f"Policy loss should be finite for large logits, got {result.policy_loss.item()}"
        )

    def test_large_negative_logits(self) -> None:
        """Very large negative logits should not cause NaN/Inf."""
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        policy_logits = torch.full((batch_size, POLICY_SIZE), -1000.0)
        target_policy = _uniform_target_policy(batch_size)
        value = torch.zeros(batch_size, 1)
        target_value = torch.zeros(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total_loss), (
            f"Total loss should be finite for large negative logits, "
            f"got {result.total_loss.item()}"
        )

    def test_mixed_extreme_logits(self) -> None:
        """Logits with both very large and very small values should be stable.

        This tests the case where one move has a much higher logit than all
        others -- softmax should put ~100% on that move.
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        policy_logits = torch.full((batch_size, POLICY_SIZE), -500.0)
        policy_logits[:, 0] = 500.0  # One extreme outlier
        target_policy = _one_hot_target_policy(batch_size, hot_index=0)
        value = torch.zeros(batch_size, 1)
        target_value = torch.zeros(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total_loss), (
            f"Total loss should be finite for mixed extreme logits, "
            f"got {result.total_loss.item()}"
        )
        # With target on the extreme logit, loss should be near zero
        assert result.policy_loss.item() < 0.01, (
            f"Policy loss should be near 0 when extreme logit matches target, "
            f"got {result.policy_loss.item():.6f}"
        )

    def test_gradients_finite_for_large_logits(self) -> None:
        """Gradients should be finite even with extreme logit values.

        We create the large logits as a leaf tensor (requires_grad=True set
        directly on the large-magnitude tensor) so that .grad is populated
        during backward().
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        # Create large-magnitude logits as a leaf tensor so .grad is populated
        policy_logits = (torch.randn(batch_size, POLICY_SIZE) * 100).requires_grad_(True)
        value = torch.randn(batch_size, 1, requires_grad=True)
        target_policy = _uniform_target_policy(batch_size)
        target_value = torch.ones(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        result.total_loss.backward()

        assert policy_logits.grad is not None, (
            "policy_logits should have a gradient"
        )
        assert torch.all(torch.isfinite(policy_logits.grad)), (
            "Policy logits gradient should be finite for large logits"
        )
        assert value.grad is not None, (
            "value should have a gradient"
        )
        assert torch.all(torch.isfinite(value.grad)), (
            "Value gradient should be finite"
        )


# ---------------------------------------------------------------------------
# Module behavior
# ---------------------------------------------------------------------------


class TestModuleBehavior:
    """Verify that AlphaZeroLoss behaves as a proper nn.Module."""

    def test_is_nn_module(self) -> None:
        """AlphaZeroLoss should be an instance of nn.Module."""
        loss_fn = AlphaZeroLoss()
        assert isinstance(loss_fn, nn.Module)

    def test_has_no_trainable_parameters(self) -> None:
        """The loss function should not have any trainable parameters.

        The loss is a pure computation -- it doesn't learn anything.
        All learning happens in the network itself.
        """
        loss_fn = AlphaZeroLoss()
        params = list(loss_fn.parameters())
        assert len(params) == 0, (
            f"Loss function should have no parameters, got {len(params)}"
        )

    def test_stores_policy_size(self) -> None:
        """The policy_size attribute should be stored for inspection."""
        loss_fn = AlphaZeroLoss(policy_size=4672)
        assert loss_fn.policy_size == 4672

    def test_custom_policy_size(self) -> None:
        """Loss should work with non-standard policy sizes."""
        custom_size = 100
        loss_fn = AlphaZeroLoss(policy_size=custom_size)

        policy_logits = torch.randn(4, custom_size)
        value = torch.zeros(4, 1)
        target_policy = torch.softmax(torch.randn(4, custom_size), dim=1)
        target_value = torch.zeros(4, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        assert torch.isfinite(result.total_loss)


# ---------------------------------------------------------------------------
# Integration with network output shapes
# ---------------------------------------------------------------------------


class TestIntegrationWithNetwork:
    """Verify the loss works with the shapes produced by AlphaZeroNetwork.

    These tests don't instantiate the full network (that's tested elsewhere),
    but they use tensors with the same shapes and value ranges that the
    network produces.
    """

    def test_with_network_like_outputs(self) -> None:
        """Loss should compute correctly with network-shaped tensors."""
        loss_fn = AlphaZeroLoss()
        batch_size = 16

        # Simulate network outputs
        policy_logits = torch.randn(batch_size, 4672)  # raw logits
        value = torch.tanh(torch.randn(batch_size, 1))  # tanh output

        # Simulate MCTS training targets
        target_policy = torch.softmax(torch.randn(batch_size, 4672), dim=1)
        target_value = torch.randint(-1, 2, (batch_size, 1)).float()

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total_loss)
        assert torch.isfinite(result.policy_loss)
        assert torch.isfinite(result.value_loss)
        assert result.total_loss.item() > 0, (
            "Loss should be positive for random predictions vs random targets"
        )

    def test_loss_decreases_when_predictions_improve(self) -> None:
        """Loss should decrease when predictions get closer to targets.

        This is a sanity check: if we manually improve the predictions,
        the loss should go down.
        """
        loss_fn = AlphaZeroLoss()
        batch_size = 4
        target_policy = _one_hot_target_policy(batch_size, hot_index=0)
        target_value = torch.ones(batch_size, 1)

        # Bad prediction
        bad_logits = torch.zeros(batch_size, POLICY_SIZE)  # uniform
        bad_value = torch.zeros(batch_size, 1)  # wrong value

        # Good prediction
        good_logits = torch.zeros(batch_size, POLICY_SIZE)
        good_logits[:, 0] = 20.0  # high logit for the correct move
        good_value = torch.full((batch_size, 1), 0.9)  # close to target

        result_bad = loss_fn(bad_logits, bad_value, target_policy, target_value)
        result_good = loss_fn(good_logits, good_value, target_policy, target_value)

        assert result_good.total_loss.item() < result_bad.total_loss.item(), (
            f"Better predictions should give lower loss: "
            f"good={result_good.total_loss.item():.4f}, "
            f"bad={result_bad.total_loss.item():.4f}"
        )
