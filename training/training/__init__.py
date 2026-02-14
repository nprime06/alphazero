"""AlphaZero training pipeline.

This package provides utilities for training the AlphaZero neural network,
including:

- :mod:`training.buffer` -- Replay buffer for loading self-play training data
- :mod:`training.dataloader` -- PyTorch Dataset/DataLoader for the replay buffer
- :mod:`training.train` -- Single-GPU training loop
- :mod:`training.checkpoint` -- Checkpoint save/resume and TorchScript export
- :mod:`training.distributed` -- DDP utilities for multi-GPU training
- :mod:`training.metrics` -- TensorBoard and W&B monitoring
"""
