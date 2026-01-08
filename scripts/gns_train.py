"""Single-GPU training script for GNS - thin CLI wrapper.

This script provides a command-line interface for training, validation,
and rollout prediction using a single GPU or CPU.
"""

import os
import sys

import torch
from absl import app, flags

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import config, reading_utils, training, rollout

# ============================================================================
# Flags
# ============================================================================

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help='The path for saving checkpoints of the model.')
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('output_filename', 'rollout', help='Base name for saving the rollout')
flags.DEFINE_string('model_file', None, help='Model filename (.pt) to resume from. Can also use "latest".')
flags.DEFINE_string('train_state_file', 'train_state.pt', help='Train state filename (.pt) to resume from.')
flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('validation_interval', None, help='Validation interval. Set None if not needed')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')
flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed).")
flags.DEFINE_integer("n_gpus", 1, help="The number of GPUs to utilize for training.")

FLAGS = flags.FLAGS


# ============================================================================
# Training
# ============================================================================

def train_single_gpu(flags_dict: dict, device: torch.device):
    """Train the model on a single GPU or CPU.

    Args:
        flags_dict: Dictionary of configuration flags.
        device: PyTorch device for training.
    """
    rank = 0
    world_size = 1
    device_id = device

    # Read metadata and create configuration
    metadata = reading_utils.read_metadata(flags_dict["data_path"], "train")
    simulator_config = config.SimulatorConfig.from_metadata(metadata)

    # Get simulator
    simulator = training.get_simulator(
        metadata, flags_dict["noise_std"], flags_dict["noise_std"],
        simulator_config, device, is_distributed=False
    )

    # Create optimizer
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags_dict["lr_init"] * world_size)

    # Initialize training state
    initial_step = 0
    initial_epoch = 0
    train_loss_hist = []
    valid_loss_hist = []

    # Load checkpoint if specified
    if flags_dict["model_file"] is not None:
        checkpoint_data = training.load_checkpoint(
            simulator, optimizer, flags_dict["model_path"],
            flags_dict["model_file"], flags_dict["train_state_file"],
            device_id, is_distributed=False
        )
        initial_step = checkpoint_data['step']
        initial_epoch = checkpoint_data['epoch']
        train_loss_hist = checkpoint_data['train_loss_hist']
        valid_loss_hist = checkpoint_data['valid_loss_hist']

    # Get data loaders
    train_dl = training.get_training_dataloader(
        flags_dict["data_path"], 'train', simulator_config.input_sequence_length,
        flags_dict["batch_size"], is_distributed=False
    )

    valid_dl = None
    if flags_dict["validation_interval"] is not None:
        valid_dl = training.get_training_dataloader(
            flags_dict["data_path"], 'valid', simulator_config.input_sequence_length,
            flags_dict["batch_size"], is_distributed=False
        )

    # Run training loop
    training.run_training_loop(
        simulator, train_dl, valid_dl, optimizer, device_id,
        rank, world_size, flags_dict, simulator_config,
        flags_dict["model_path"], is_distributed=False,
        initial_step=initial_step, initial_epoch=initial_epoch,
        train_loss_hist=train_loss_hist, valid_loss_hist=valid_loss_hist
    )


# ============================================================================
# Prediction
# ============================================================================

def predict_rollouts(flags_dict: dict, device: torch.device):
    """Run rollout predictions.

    Args:
        flags_dict: Dictionary of configuration flags.
        device: PyTorch device for prediction.
    """
    # Read metadata and create configuration
    metadata = reading_utils.read_metadata(flags_dict["data_path"], "rollout")
    simulator_config = config.SimulatorConfig.from_metadata(metadata)

    # Get simulator
    simulator = training.get_simulator(
        metadata, flags_dict["noise_std"], flags_dict["noise_std"],
        simulator_config, device, is_distributed=False
    )

    # Load model
    model_filepath = flags_dict["model_path"] + flags_dict["model_file"]
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file not found: {model_filepath}")
    simulator.load(model_filepath)

    # Determine split
    split = 'test' if FLAGS.mode == 'rollout' else 'valid'

    # Run predictions
    rollout.predict_rollouts(
        simulator, flags_dict["data_path"], split, simulator_config, device,
        output_path=flags_dict["output_path"],
        output_filename=flags_dict["output_filename"],
        save_results=(FLAGS.mode == 'rollout')
    )


# ============================================================================
# Main
# ============================================================================

def main(_):
    """Main entry point."""
    # Convert flags to dictionary
    flags_dict = reading_utils.flags_to_dict(FLAGS)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if FLAGS.mode == 'train':
        # Create model directory if it doesn't exist
        if not os.path.exists(FLAGS.model_path):
            os.makedirs(FLAGS.model_path)

        # Train
        train_single_gpu(flags_dict, device)

    elif FLAGS.mode in ['valid', 'rollout']:
        # Set device if specified
        if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
            device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')

        # Run prediction
        predict_rollouts(flags_dict, device)


if __name__ == '__main__':
    app.run(main)
