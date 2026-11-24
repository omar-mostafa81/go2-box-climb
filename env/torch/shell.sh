#!/bin/bash
# Open the Isaac Lab container on NYU Torch (run it on a compute node, never on a login node).
# Expects everything in /scratch/$USER/ocrl: isaaclab.ext3, IsaacLab/, go2-box-climb/, home/, nvidia_icd.json.
IMG=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
ICD=/scratch/$USER/ocrl/nvidia_icd.json   # Vulkan fix: without it Isaac Sim cannot render ("No device could be created")
apptainer exec --nv --fakeroot -B /scratch --home /scratch/$USER/ocrl/home \
  --env OCRL=/scratch/$USER/ocrl --env VK_ICD_FILENAMES=$ICD --env VK_DRIVER_FILES=$ICD \
  --overlay /scratch/$USER/ocrl/isaaclab.ext3:rw "$IMG" /bin/bash --rcfile /ext3/env.sh
