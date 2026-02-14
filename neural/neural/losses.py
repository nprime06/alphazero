"""
AlphaZero Loss Function
========================

Implements the combined loss function used to train the AlphaZero neural network.
The loss has two components -- a policy loss and a value loss -- that together
teach the network to predict both good moves and accurate position evaluations.

Why two losses?
    AlphaZero's network has two heads that make different predictions:

    1. **Policy head**: predicts a probability distribution over moves. During
       self-play, MCTS produces a visit-count distribution that represents "what
       a strong search thinks is good." The policy loss trains the network to
       match these search results, distilling MCTS knowledge into fast single-
       pass inference.

    2. **Value head**: predicts who is winning (scalar in [-1, 1]). The target
       is the actual game outcome: +1 if the current player won, -1 if they
       lost, 0 for a draw. The value loss trains the network to accurately
       evaluate positions.

    By training both heads jointly with a shared trunk, the network learns
    positional features useful for both move selection and evaluation.

Policy loss: cross-entropy with MCTS visit distribution
    ``L_policy = -sum(pi_target * log_softmax(policy_logits))``

    where ``pi_target`` is the MCTS search probability distribution (sums to 1)
    and ``policy_logits`` are the raw outputs from the policy head.

    Why log_softmax instead of log(softmax)?
        Both compute the same mathematical quantity, but ``log_softmax`` is
        numerically stable. Plain ``softmax`` can overflow for large logits
        (exp(1000) = inf) or underflow for very negative logits (exp(-1000) = 0).
        ``log_softmax`` avoids this by using the identity:

            log(softmax(x_i)) = x_i - log(sum(exp(x_j)))

        and subtracting max(x) before computing the exponentials:

            log(softmax(x_i)) = (x_i - max(x)) - log(sum(exp(x_j - max(x))))

        This keeps all intermediate values in a safe numerical range.

Value loss: mean squared error
    ``L_value = mean((z - v)^2)``

    where ``z`` is the game outcome ({-1, 0, +1}) and ``v`` is the network's
    tanh output (already bounded to [-1, 1]). MSE is used because the value
    prediction is a regression problem -- we want the network's output to be
    as close as possible to the true game outcome.

Combined loss:
    ``L = L_value + L_policy``

    The two losses are simply added together. The original AlphaZero paper also
    includes an L2 regularization term (weight decay), but this is handled by
    the optimizer (e.g., ``torch.optim.SGD(weight_decay=1e-4)``) rather than
    being computed explicitly in the loss function.

    Why handle weight decay in the optimizer?
        - **Decoupled weight decay** (as in AdamW) applies regularization
          directly to the weights, independent of the gradient magnitude.
          This is mathematically different from adding an L2 penalty to the
          loss when using adaptive optimizers like Adam.
        - It keeps the loss function clean: the loss measures prediction
          quality only, without mixing in regularization concerns.
        - It avoids double-counting if you later switch to an optimizer
          that has its own weight decay implementation.
        - The optimizer's weight_decay parameter is more transparent: you
          can see the regularization strength in the optimizer config
          rather than hunting for it inside the loss function.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossResult(NamedTuple):
    """Container for the three loss components returned by AlphaZeroLoss.

    Using a NamedTuple gives us:
    - Named access: ``result.policy_loss`` is clearer than ``result[1]``
    - Tuple unpacking: ``total, policy, value = loss_fn(...)``
    - Immutability: loss values should not be modified after computation
    - Zero overhead: NamedTuple is as efficient as a plain tuple

    All three fields are scalar tensors (0-dimensional) that support
    ``.backward()`` and standard arithmetic operations.

    Attributes:
        total_loss: The combined loss ``L_value + L_policy``. This is the
            value that should be passed to ``loss.backward()`` for training.
        policy_loss: The cross-entropy loss between the MCTS visit distribution
            and the network's policy output. Measures how well the network
            predicts the moves that MCTS would choose.
        value_loss: The MSE loss between the game outcome and the network's
            value prediction. Measures how well the network evaluates positions.
    """

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor


class AlphaZeroLoss(nn.Module):
    """Combined loss function for AlphaZero training.

    Computes the policy cross-entropy loss and value MSE loss, returning both
    individual components and their sum for logging and backpropagation.

    The loss is averaged over the batch dimension, so the magnitude is
    independent of batch size. This is important for consistent training
    dynamics when changing batch size (e.g., for GPU memory reasons).

    Architecture::

        policy_logits (B, 4672)  ──> log_softmax ──> cross-entropy with target ──> L_policy
        value (B, 1)             ──> MSE with target ──────────────────────────> L_value
                                                                                    |
                                                                    L = L_value + L_policy

    Args:
        policy_size: Size of the policy output vector. Default 4672 for chess.
            This is stored for documentation purposes and input validation
            but does not affect the loss computation.

    Example::

        >>> loss_fn = AlphaZeroLoss()
        >>> policy_logits = torch.randn(8, 4672)  # raw network output
        >>> value = torch.tanh(torch.randn(8, 1))  # network value output
        >>> target_policy = torch.softmax(torch.randn(8, 4672), dim=1)  # MCTS distribution
        >>> target_value = torch.tensor([[1.0]] * 4 + [[-1.0]] * 4)  # game outcomes
        >>> result = loss_fn(policy_logits, value, target_policy, target_value)
        >>> result.total_loss.backward()  # train the network
    """

    def __init__(self, policy_size: int = 4672) -> None:
        super().__init__()
        self.policy_size = policy_size

    def forward(
        self,
        policy_logits: torch.Tensor,
        value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> LossResult:
        """Compute the combined AlphaZero loss.

        Args:
            policy_logits: Raw logits from the policy head, shape ``(B, policy_size)``.
                These are NOT softmax-normalized -- the loss function applies
                log_softmax internally for numerical stability.
            value: Predicted value from the value head, shape ``(B, 1)``.
                This should be the tanh output, bounded to [-1, 1].
            target_policy: MCTS visit count distribution, shape ``(B, policy_size)``.
                Each row should sum to 1.0 (a valid probability distribution).
                Entries corresponding to illegal moves should be 0.0.
            target_value: Game outcome, shape ``(B, 1)``.
                Values should be in {-1, 0, +1}: -1 for loss, 0 for draw,
                +1 for win (from the perspective of the player to move).

        Returns:
            LossResult: A named tuple with ``total_loss``, ``policy_loss``, and
            ``value_loss``. All are scalar tensors (0-dimensional) averaged
            over the batch.

        Note:
            The policy loss uses ``F.log_softmax`` on the logits rather than
            computing ``log(F.softmax(logits))``. See the module docstring
            for why this matters for numerical stability.
        """
        # --- Policy loss: cross-entropy with MCTS distribution ---
        #
        # Step 1: Apply log_softmax to the raw logits. This converts them to
        # log-probabilities in a numerically stable way.
        #
        # Step 2: Multiply element-wise by the target distribution and sum.
        # This computes: -sum(pi_target * log(softmax(logits)))
        # which is the cross-entropy between the target and predicted distributions.
        #
        # Step 3: Average over the batch. The negative sign makes the loss
        # positive (since log-probabilities are negative).
        log_probs = F.log_softmax(policy_logits, dim=1)  # (B, policy_size)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()

        # --- Value loss: mean squared error ---
        #
        # Simple MSE: (target - prediction)^2, averaged over the batch.
        # Both target and prediction are shape (B, 1), so the squeeze/mean
        # handles everything cleanly.
        #
        # We use F.mse_loss for clarity and consistency with PyTorch conventions.
        value_loss = F.mse_loss(value, target_value)

        # --- Combined loss ---
        #
        # Simply add the two losses. The original paper weights them equally
        # (both have coefficient 1.0). Some implementations allow configurable
        # weights, but the AlphaZero paper does not tune these -- the losses
        # naturally balance because:
        # - Policy loss is O(log(num_moves)) at random, ~8.4 for chess
        # - Value loss is O(1) at random (MSE of uniform [-1,1] vs targets)
        # So they are already roughly the same order of magnitude.
        total_loss = policy_loss + value_loss

        return LossResult(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
        )
