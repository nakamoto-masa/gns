"""Training utilities for GNS simulator.

This module provides functions for training, validation, checkpointing,
and data loading for the Graph Network-based Simulator.
"""

import collections
import glob
import os
import re
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from gns import config
from gns import data_loader
from gns import distribute
from gns import learned_simulator
from gns import noise_utils


# ============================================================================
# Utility Functions
# ============================================================================

def optimizer_to(optim: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state to specified device.

    Args:
        optim: PyTorch optimizer.
        device: Target device for the optimizer state.
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def acceleration_loss(
        pred_acc: torch.Tensor,
        target_acc: torch.Tensor,
        non_kinematic_mask: torch.Tensor) -> torch.Tensor:
    """Compute the loss between predicted and target accelerations.

    Args:
        pred_acc: Predicted accelerations.
        target_acc: Target accelerations.
        non_kinematic_mask: Mask for kinematic particles (1=non-kinematic, 0=kinematic).

    Returns:
        Masked mean squared error loss.
    """
    loss = (pred_acc - target_acc) ** 2
    loss = loss.sum(dim=-1)
    num_non_kinematic = non_kinematic_mask.sum()
    loss = torch.where(non_kinematic_mask.bool(),
                       loss, torch.zeros_like(loss))
    loss = loss.sum() / num_non_kinematic
    return loss


# ============================================================================
# Batch Preparation
# ============================================================================

def prepare_training_batch(
        example: Tuple[Tuple[torch.Tensor, ...], torch.Tensor],
        n_features: int,
        device_id: torch.device) -> dict[str, torch.Tensor]:
    """Prepare a training batch by extracting features and labels.

    This function replaces the 15-line boilerplate that appears in 3 places
    in the original training code. It handles the conditional logic for
    material properties based on the number of features.

    Args:
        example: Batch example containing ((position, particle_type, [material_property], n_particles), labels).
        n_features: Number of features (2 or 3).
        device_id: Target device for tensors.

    Returns:
        Dictionary containing:
            - position: Particle positions (batch, sequence, particles, dims)
            - particle_type: Particle type IDs (batch, particles)
            - material_property: Material properties (batch, particles) or None
            - n_particles_per_example: Number of particles per example
            - labels: Target next positions (batch, particles, dims)

    Raises:
        NotImplementedError: If n_features is not 2 or 3.
    """
    position = example[0][0].to(device_id)
    particle_type = example[0][1].to(device_id)

    if n_features == 3:
        material_property = example[0][2].to(device_id)
        n_particles_per_example = example[0][3].to(device_id)
    elif n_features == 2:
        material_property = None
        n_particles_per_example = example[0][2].to(device_id)
    else:
        raise NotImplementedError(f"n_features={n_features} not supported. Expected 2 or 3.")

    labels = example[1].to(device_id)

    return {
        'position': position,
        'particle_type': particle_type,
        'material_property': material_property,
        'n_particles_per_example': n_particles_per_example,
        'labels': labels
    }


# ============================================================================
# Learning Rate Management
# ============================================================================

def update_learning_rate(
        optimizer: torch.optim.Optimizer,
        step: int,
        lr_init: float,
        lr_decay: float,
        lr_decay_steps: int,
        world_size: int) -> float:
    """Update learning rate with exponential decay.

    This function replaces the hardcoded learning rate scheduling logic
    that was embedded in the training loop.

    Args:
        optimizer: PyTorch optimizer.
        step: Current training step.
        lr_init: Initial learning rate.
        lr_decay: Decay factor for learning rate.
        lr_decay_steps: Number of steps for one decay period.
        world_size: Number of distributed processes (for scaling).

    Returns:
        New learning rate value.
    """
    lr_new = lr_init * (lr_decay ** (step / lr_decay_steps)) * world_size
    for param in optimizer.param_groups:
        param['lr'] = lr_new
    return lr_new


# ============================================================================
# Noise Addition
# ============================================================================

def add_training_noise(
        position: torch.Tensor,
        particle_type: torch.Tensor,
        noise_std: float,
        kinematic_particle_id: int,
        device_id: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add random walk noise to position sequence for training.

    Args:
        position: Position sequence (batch, sequence, particles, dims).
        particle_type: Particle type IDs (batch, particles).
        noise_std: Standard deviation of noise for the last step.
        kinematic_particle_id: Particle type ID for kinematic particles.
        device_id: Target device for tensors.

    Returns:
        Tuple of (sampled_noise, non_kinematic_mask):
            - sampled_noise: Noise tensor with same shape as position
            - non_kinematic_mask: Binary mask (1=non-kinematic, 0=kinematic)
    """
    sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
        position, noise_std_last_step=noise_std).to(device_id)
    non_kinematic_mask = (particle_type != kinematic_particle_id).clone().detach().to(device_id)
    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)
    return sampled_noise, non_kinematic_mask


# ============================================================================
# Single Step Operations
# ============================================================================

def train_step(
        predict_fn,
        batch: dict[str, torch.Tensor],
        sampled_noise: torch.Tensor,
        non_kinematic_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device_or_rank: Any) -> float:
    """Execute a single training step.

    Args:
        predict_fn: Prediction function (simulator.predict_accelerations or simulator.module.predict_accelerations).
        batch: Batch dictionary from prepare_training_batch().
        sampled_noise: Noise tensor for position sequence.
        non_kinematic_mask: Binary mask for non-kinematic particles.
        optimizer: PyTorch optimizer.
        device_or_rank: Device (CPU) or rank (GPU) for tensors.

    Returns:
        Training loss value for this step.
    """
    # Get predictions and target accelerations
    pred_acc, target_acc = predict_fn(
        next_positions=batch['labels'].to(device_or_rank),
        position_sequence_noise=sampled_noise.to(device_or_rank),
        position_sequence=batch['position'].to(device_or_rank),
        nparticles_per_example=batch['n_particles_per_example'].to(device_or_rank),
        particle_types=batch['particle_type'].to(device_or_rank),
        material_property=batch['material_property'].to(device_or_rank) if batch['material_property'] is not None else None
    )

    # Calculate loss
    loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def validation_step(
        simulator: learned_simulator.LearnedSimulator,
        example: Tuple[Tuple[torch.Tensor, ...], torch.Tensor],
        n_features: int,
        noise_std: float,
        simulator_config: config.SimulatorConfig,
        device_id: torch.device,
        is_distributed: bool) -> torch.Tensor:
    """Execute a single validation step.

    Args:
        simulator: Learned simulator model.
        example: Validation batch example.
        n_features: Number of features (2 or 3).
        noise_std: Standard deviation of noise.
        simulator_config: Simulator configuration.
        device_id: Target device for tensors.
        is_distributed: Whether using distributed training.

    Returns:
        Validation loss tensor.
    """
    # Prepare batch
    batch = prepare_training_batch(example, n_features, device_id)

    # Add noise
    sampled_noise, non_kinematic_mask = add_training_noise(
        batch['position'],
        batch['particle_type'],
        noise_std,
        simulator_config.kinematic_particle_id,
        device_id
    )

    # Get device or rank for prediction
    device_or_rank = device_id if not is_distributed else device_id

    # Select the appropriate prediction function
    predict_accelerations = (
        simulator.module.predict_accelerations if is_distributed
        else simulator.predict_accelerations
    )

    # Get predictions and target accelerations
    with torch.no_grad():
        pred_acc, target_acc = predict_accelerations(
            next_positions=batch['labels'].to(device_or_rank),
            position_sequence_noise=sampled_noise.to(device_or_rank),
            position_sequence=batch['position'].to(device_or_rank),
            nparticles_per_example=batch['n_particles_per_example'].to(device_or_rank),
            particle_types=batch['particle_type'].to(device_or_rank),
            material_property=batch['material_property'].to(device_or_rank) if batch['material_property'] is not None else None
        )

    # Compute loss
    loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

    return loss


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(
        simulator: learned_simulator.LearnedSimulator,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        train_loss: float,
        valid_loss: Optional[float],
        train_loss_hist: list,
        valid_loss_hist: list,
        model_path: str,
        rank: int,
        is_distributed: bool) -> None:
    """Save model checkpoint and training state.

    Args:
        simulator: Learned simulator model.
        optimizer: PyTorch optimizer.
        step: Current training step.
        epoch: Current epoch.
        train_loss: Current training loss.
        valid_loss: Current validation loss (can be None).
        train_loss_hist: Training loss history [(epoch, loss), ...].
        valid_loss_hist: Validation loss history [(epoch, loss), ...].
        model_path: Directory path for saving checkpoints.
        rank: Process rank (0 for main process).
        is_distributed: Whether using distributed training.
    """
    if rank != 0:
        return

    # Save model
    if is_distributed:
        simulator.module.save(f'{model_path}model-{step}.pt')
    else:
        simulator.save(f'{model_path}model-{step}.pt')

    # Save training state
    train_state = dict(
        optimizer_state=optimizer.state_dict(),
        global_train_state={
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        },
        loss_history={"train": train_loss_hist, "valid": valid_loss_hist}
    )
    torch.save(train_state, f'{model_path}train_state-{step}.pt')


def find_latest_checkpoint(model_path: str) -> Tuple[str, str]:
    """Find the latest checkpoint files in the model directory.

    Args:
        model_path: Directory containing checkpoint files.

    Returns:
        Tuple of (model_file, train_state_file) filenames.

    Raises:
        FileNotFoundError: If no checkpoint files are found.
    """
    fnames = glob.glob(f'{model_path}*model*pt')
    if not fnames:
        raise FileNotFoundError(f"No checkpoint files found in {model_path}")

    max_model_number = 0
    expr = re.compile(r".*model-(\d+).pt")
    for fname in fnames:
        match = expr.search(fname)
        if match:
            model_num = int(match.groups()[0])
            if model_num > max_model_number:
                max_model_number = model_num

    return f"model-{max_model_number}.pt", f"train_state-{max_model_number}.pt"


def load_checkpoint(
        simulator: learned_simulator.LearnedSimulator,
        optimizer: torch.optim.Optimizer,
        model_path: str,
        model_file: str,
        train_state_file: str,
        device_id: torch.device,
        is_distributed: bool) -> dict[str, Any]:
    """Load model checkpoint and training state.

    Args:
        simulator: Learned simulator model.
        optimizer: PyTorch optimizer.
        model_path: Directory path for checkpoints.
        model_file: Model filename (e.g., 'model-1000.pt' or 'latest').
        train_state_file: Training state filename (e.g., 'train_state-1000.pt' or 'latest').
        device_id: Target device for loading.
        is_distributed: Whether using distributed training.

    Returns:
        Dictionary containing:
            - step: Resumed training step
            - epoch: Resumed epoch
            - train_loss_hist: Training loss history
            - valid_loss_hist: Validation loss history

    Raises:
        FileNotFoundError: If checkpoint files don't exist.
    """
    # Find latest checkpoint if requested
    if model_file == "latest" and train_state_file == "latest":
        model_file, train_state_file = find_latest_checkpoint(model_path)

    # Check if files exist
    model_filepath = f'{model_path}{model_file}'
    train_state_filepath = f'{model_path}{train_state_file}'

    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file not found: {model_filepath}")
    if not os.path.exists(train_state_filepath):
        raise FileNotFoundError(f"Train state file not found: {train_state_filepath}")

    # Load model
    if is_distributed:
        simulator.module.load(model_filepath)
    else:
        simulator.load(model_filepath)

    # Load train state
    train_state = torch.load(train_state_filepath)

    # Load optimizer state
    optimizer.load_state_dict(train_state["optimizer_state"])
    optimizer_to(optimizer, device_id)

    # Extract global train state
    global_state = train_state["global_train_state"]
    loss_history = train_state["loss_history"]

    return {
        'step': global_state["step"],
        'epoch': global_state["epoch"],
        'train_loss_hist': loss_history["train"],
        'valid_loss_hist': loss_history["valid"]
    }


# ============================================================================
# Simulator Creation
# ============================================================================

def create_simulator(
        simulator_config: config.SimulatorConfig,
        normalization_stats: dict[str, dict[str, torch.Tensor]],
        feature_dims: dict[str, int],
        connectivity_radius: float,
        boundaries: np.ndarray,
        boundary_clamp_limit: float,
        device: torch.device) -> learned_simulator.LearnedSimulator:
    """Create a simulator instance from configuration and computed parameters.

    This function handles the actual instantiation of the LearnedSimulator,
    separating simulator creation from configuration interpretation.

    Args:
        simulator_config: Simulator configuration.
        normalization_stats: Normalization statistics for acceleration and velocity.
        feature_dims: Feature dimensions including nnode_in, nedge_in, particle_dimensions.
        connectivity_radius: Radius for connecting particles in the graph.
        boundaries: Spatial boundaries of the simulation domain.
        boundary_clamp_limit: Boundary clamp limit for training.
        device: PyTorch device 'cpu' or 'cuda'.

    Returns:
        Initialized LearnedSimulator instance.

    Example:
        >>> normalization_stats = config.build_normalization_stats(metadata, 0.0003, 0.0003, device)
        >>> feature_dims = config.infer_feature_dimensions(metadata, config)
        >>> simulator = create_simulator(
        ...     config, normalization_stats, feature_dims,
        ...     metadata['default_connectivity_radius'],
        ...     np.array(metadata['bounds']), 1.0, device)
    """
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=feature_dims['particle_dimensions'],
        nnode_in=feature_dims['nnode_in'],
        nedge_in=feature_dims['nedge_in'],
        latent_dim=128,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=connectivity_radius,
        boundaries=boundaries,
        normalization_stats=normalization_stats,
        nparticle_types=simulator_config.num_particle_types,
        particle_type_embedding_size=16,
        boundary_clamp_limit=boundary_clamp_limit,
        device=device)

    return simulator


def get_simulator(
        metadata: dict,
        acc_noise_std: float,
        vel_noise_std: float,
        simulator_config: config.SimulatorConfig,
        device: torch.device,
        rank: Optional[int] = None,
        is_distributed: bool = False) -> learned_simulator.LearnedSimulator:
    """Get a simulator instance, optionally wrapped with DDP for distributed training.

    Args:
        metadata: Dataset metadata containing normalization stats and dimensions.
        acc_noise_std: Acceleration noise standard deviation.
        vel_noise_std: Velocity noise standard deviation.
        simulator_config: Simulator configuration.
        device: PyTorch device 'cpu' or 'cuda'.
        rank: Process rank for distributed training (required if is_distributed=True).
        is_distributed: Whether to wrap simulator with DistributedDataParallel.

    Returns:
        LearnedSimulator instance, optionally wrapped with DDP.
    """
    # Build normalization statistics from metadata
    normalization_stats = config.build_normalization_stats(
        metadata, acc_noise_std, vel_noise_std, device)

    # Infer feature dimensions from metadata
    feature_dims = config.infer_feature_dimensions(metadata, simulator_config)

    # Extract boundary clamp limit (default to 1.0 if not specified)
    boundary_clamp_limit = metadata.get("boundary_augment", 1.0)

    # Create simulator
    simulator = create_simulator(
        simulator_config=simulator_config,
        normalization_stats=normalization_stats,
        feature_dims=feature_dims,
        connectivity_radius=metadata['default_connectivity_radius'],
        boundaries=np.array(metadata['bounds']),
        boundary_clamp_limit=boundary_clamp_limit,
        device=device)

    # Wrap with DDP if distributed
    if is_distributed:
        if rank is None:
            raise ValueError("rank must be provided when is_distributed=True")
        simulator = DDP(simulator.to(rank), device_ids=[rank], output_device=rank)

    return simulator


# ============================================================================
# Data Loading
# ============================================================================

def get_training_dataloader(
        data_path: str,
        split: str,
        input_sequence_length: int,
        batch_size: int,
        is_distributed: bool = False):
    """Get a data loader for training or validation.

    Args:
        data_path: Path to data directory.
        split: Data split ('train' or 'valid').
        input_sequence_length: Length of input position sequence.
        batch_size: Batch size.
        is_distributed: Whether to use distributed data loader.

    Returns:
        PyTorch DataLoader instance.
    """
    get_data_loader = (
        distribute.get_data_distributed_dataloader_by_samples
        if is_distributed
        else data_loader.get_data_loader_by_samples
    )

    dl = get_data_loader(
        path=f'{data_path}{split}.npz',
        input_length_sequence=input_sequence_length,
        batch_size=batch_size,
    )

    return dl


def infer_dataset_features(dataloader) -> int:
    """Infer the number of features in a dataset.

    Args:
        dataloader: PyTorch DataLoader instance.

    Returns:
        Number of features (2 or 3).
    """
    return len(dataloader.dataset._data[0])


# ============================================================================
# Training Orchestration
# ============================================================================

def run_training_loop(
        simulator: learned_simulator.LearnedSimulator,
        train_dl,
        valid_dl,
        optimizer: torch.optim.Optimizer,
        device_id: torch.device,
        rank: int,
        world_size: int,
        flags: dict,
        simulator_config: config.SimulatorConfig,
        model_path: str,
        is_distributed: bool,
        initial_step: int = 0,
        initial_epoch: int = 0,
        train_loss_hist: Optional[list] = None,
        valid_loss_hist: Optional[list] = None) -> None:
    """Run the complete training loop.

    This is the high-level orchestration function that manages the entire
    training process, including epochs, steps, validation, checkpointing,
    and learning rate scheduling.

    Args:
        simulator: Learned simulator model.
        train_dl: Training data loader.
        valid_dl: Validation data loader (can be None).
        optimizer: PyTorch optimizer.
        device_id: Device for training.
        rank: Process rank.
        world_size: Total number of processes.
        flags: Training configuration flags.
        simulator_config: Simulator configuration.
        model_path: Path for saving checkpoints.
        is_distributed: Whether using distributed training.
        initial_step: Starting step (for resuming training).
        initial_epoch: Starting epoch (for resuming training).
        train_loss_hist: Training loss history (for resuming training).
        valid_loss_hist: Validation loss history (for resuming training).
    """
    # Initialize training state
    step = initial_step
    epoch = initial_epoch
    steps_per_epoch = 0

    valid_loss = None
    epoch_train_loss = 0
    epoch_valid_loss = None

    if train_loss_hist is None:
        train_loss_hist = []
    if valid_loss_hist is None:
        valid_loss_hist = []

    # Infer number of features
    n_features = infer_dataset_features(train_dl)

    # Validate validation data if provided
    if valid_dl is not None:
        n_features_valid = infer_dataset_features(valid_dl)
        if n_features_valid != n_features:
            raise ValueError(
                f"n_features mismatch: train={n_features}, valid={n_features_valid}"
            )

    # Set simulator to training mode
    simulator.train()
    simulator.to(device_id)

    print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")

    try:
        while step < flags["ntraining_steps"]:
            if is_distributed:
                torch.distributed.barrier()

            for example in train_dl:
                steps_per_epoch += 1

                # Prepare batch
                batch = prepare_training_batch(example, n_features, device_id)

                # Add training noise
                sampled_noise, non_kinematic_mask = add_training_noise(
                    batch['position'],
                    batch['particle_type'],
                    flags["noise_std"],
                    simulator_config.kinematic_particle_id,
                    device_id
                )

                # Get device or rank for prediction
                device_or_rank = rank if is_distributed else device_id

                # Select the appropriate prediction function
                predict_accelerations = (
                    simulator.module.predict_accelerations if is_distributed
                    else simulator.predict_accelerations
                )

                # Training step
                train_loss = train_step(
                    predict_accelerations,
                    batch,
                    sampled_noise,
                    non_kinematic_mask,
                    optimizer,
                    device_or_rank
                )

                epoch_train_loss += train_loss

                # Validation
                if valid_dl is not None and flags["validation_interval"] is not None:
                    sampled_valid_example = next(iter(valid_dl))
                    if step > 0 and step % flags["validation_interval"] == 0:
                        valid_loss = validation_step(
                            simulator, sampled_valid_example, n_features,
                            flags["noise_std"], simulator_config, device_id, is_distributed
                        )
                        print(f"Validation loss at {step}: {valid_loss.item()}")

                # Update learning rate
                lr_new = update_learning_rate(
                    optimizer, step, flags["lr_init"], flags["lr_decay"],
                    flags["lr_decay_steps"], world_size
                )

                print(f'rank = {rank}, epoch = {epoch}, step = {step}/{flags["ntraining_steps"]}, loss = {train_loss}', flush=True)

                # Save checkpoint
                if step % flags["nsave_steps"] == 0:
                    save_checkpoint(
                        simulator, optimizer, step, epoch, train_loss, valid_loss,
                        train_loss_hist, valid_loss_hist, model_path, rank, is_distributed
                    )

                step += 1
                if step >= flags["ntraining_steps"]:
                    break

            # Epoch level statistics
            # Training loss at epoch
            epoch_train_loss /= steps_per_epoch
            epoch_train_loss = torch.tensor([epoch_train_loss]).to(device_id)
            if is_distributed:
                torch.distributed.reduce(epoch_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                epoch_train_loss /= world_size

            train_loss_hist.append((epoch, epoch_train_loss.item()))

            # Validation loss at epoch
            if valid_dl is not None and flags["validation_interval"] is not None:
                sampled_valid_example = next(iter(valid_dl))
                epoch_valid_loss = validation_step(
                    simulator, sampled_valid_example, n_features,
                    flags["noise_std"], simulator_config, device_id, is_distributed
                )
                if is_distributed:
                    torch.distributed.reduce(epoch_valid_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    epoch_valid_loss /= world_size

                valid_loss_hist.append((epoch, epoch_valid_loss.item()))

            # Print epoch statistics
            if rank == 0 or not is_distributed:
                print(f'Epoch {epoch}, training loss: {epoch_train_loss.item()}')
                if valid_dl is not None and flags["validation_interval"] is not None:
                    print(f'Epoch {epoch}, validation loss: {epoch_valid_loss.item()}')

            # Reset epoch training loss
            epoch_train_loss = 0
            if steps_per_epoch >= len(train_dl):
                epoch += 1
            steps_per_epoch = 0

            if step >= flags["ntraining_steps"]:
                break

    except KeyboardInterrupt:
        pass

    # Save model state on keyboard interrupt or completion
    save_checkpoint(
        simulator, optimizer, step, epoch, train_loss if 'train_loss' in locals() else 0.0,
        valid_loss, train_loss_hist, valid_loss_hist, model_path, rank, is_distributed
    )
