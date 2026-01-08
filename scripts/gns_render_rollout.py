"""Render rollout predictions - thin CLI wrapper.

This script provides a command-line interface for rendering rollout
predictions as GIF animations or VTK files.
"""

import sys
import os
from absl import app, flags

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import render

# ============================================================================
# Flags
# ============================================================================

flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout `.pkl` file")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip")
flags.DEFINE_bool("change_yz", False, help="Change y and z axis (3D only)")
flags.DEFINE_enum("output_mode", "gif", ["gif", "vtk"], help="Render output type")

FLAGS = flags.FLAGS


# ============================================================================
# Main
# ============================================================================

def main(_):
    """Main entry point."""
    # Validate required flags
    if not FLAGS.rollout_dir:
        raise ValueError("A `rollout_dir` must be passed.")
    if not FLAGS.rollout_name:
        raise ValueError("A `rollout_name` must be passed.")

    # Render based on output mode
    if FLAGS.output_mode == "gif":
        render.render_gif_animation(
            rollout_dir=FLAGS.rollout_dir,
            rollout_name=FLAGS.rollout_name,
            point_size=1,
            timestep_stride=FLAGS.step_stride,
            vertical_camera_angle=20,
            viewpoint_rotation=0.3,
            change_yz=FLAGS.change_yz
        )
    elif FLAGS.output_mode == "vtk":
        render.write_vtk_trajectory(
            rollout_dir=FLAGS.rollout_dir,
            rollout_name=FLAGS.rollout_name
        )


if __name__ == '__main__':
    app.run(main)
