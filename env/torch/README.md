# Running on NYU Torch (Apptainer)

Torch runs everything inside an Apptainer container. Our Python environment
lives in an "overlay": one 50 GB file (`isaaclab.ext3`) that the container sees
as the folder `/ext3`. Everything sits in `/scratch/$USER/ocrl`.

| File | What it is |
| --- | --- |
| `shell.sh` | Opens the container on a compute node: GPU (`--nv`), the overlay, and the Vulkan fix |
| `env.sh` | Runs when the container opens: loads conda, activates `isaaclab`, accepts the Isaac Sim EULA. Lives at `/ext3/env.sh` inside the overlay |
| `nvidia_icd.json` | The Vulkan fix. The container's Vulkan only lists Mesa drivers, not NVIDIA's, so Isaac Sim's renderer fails with "No device could be created" (physics still runs, but no images and no `--video`). This file points Vulkan to the NVIDIA driver that `--nv` brings in |

## One-time setup

On a login node:

```bash
mkdir -p /scratch/$USER/ocrl/home && cd /scratch/$USER/ocrl
cp /share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
zcat overlay-50G-10M.ext3.gz | cp --sparse=always /dev/stdin isaaclab.ext3 && rm overlay-50G-10M.ext3.gz
git clone https://github.com/omar-mostafa81/go2-box-climb.git
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout 50c445b143ee40514090303923188f9536be85bc && cd ..
cp go2-box-climb/env/torch/shell.sh go2-box-climb/env/torch/nvidia_icd.json .
```

Get a CPU node (installing on login nodes is not allowed), wait until the prompt
shows the node name, then open the container:

```bash
srun --account=<ACCOUNT> --cpus-per-task=8 --mem=32GB --time=04:00:00 --pty /bin/bash
bash /scratch/$USER/ocrl/shell.sh
```

Inside the container, install Miniforge and `env.sh` into the overlay:

```bash
cd /ext3
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3 && rm Miniforge3-Linux-x86_64.sh
cp $OCRL/go2-box-climb/env/torch/env.sh /ext3/env.sh
source /ext3/env.sh
```

Then follow [docs/installation.md](../../docs/installation.md) to create the
`isaaclab` conda env and install the packages (`IsaacLab` is at `$OCRL/IsaacLab`).
Two additions on Torch: after `conda activate isaaclab`, also run
`conda install git cmake make compilers -y` (our working setup has these build
tools), and set `export UV_LINK_MODE=copy` before the `uv` install. Copying into the overlay is
slow, so that step can take a few hours; run it inside `tmux`.

## Every time

```bash
srun --account=<ACCOUNT> --gres=gpu:1 --constraint='l40s' --cpus-per-task=8 --mem=32GB --time=02:00:00 --pty /bin/bash
bash /scratch/$USER/ocrl/shell.sh
cd $OCRL/go2-box-climb
```

- Use L40S GPUs. The H200s have no RT cores, so Isaac Sim cannot render on them.
- Run `shell.sh` only after the prompt shows a compute node. Do not paste it
  together with `srun`. Otherwise it starts on the login node, and the overlay
  stays locked ("in use by another process").
- Only one container can open the overlay at a time (`:rw`).
- `--mode stream` does not work from a laptop: Torch's firewall blocks the
  compute nodes, even on the NYU VPN. Use `--video` instead (see the main
  [README](../../README.md)) and copy the MP4 to your computer with `scp`.
