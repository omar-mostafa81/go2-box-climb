# Installation

Follow these steps exactly. Everyone on the team must have the **same versions**,
otherwise trained policies and results may not match between machines.

## Tested versions

| Component | Version |
| --- | --- |
| OS | Ubuntu 24.04 (any Linux with GLIBC ≥ 2.35 should work) |
| GPU / driver | NVIDIA RTX GPU, driver 595.84 (tested) |
| Python | 3.12 (conda env named `isaaclab`) |
| Isaac Sim | 6.0.1.0 (pip install) |
| Isaac Lab | `develop` branch, commit `50c445b143ee40514090303923188f9536be85bc` (version 3.0.0) |
| PyTorch | 2.11.0 + CUDA 12.8 |
| RSL-RL | `rsl-rl-lib` 5.4.1 |

All other package versions are fixed by `env/isaaclab-requirements.lock.txt` in
this repository.

## Steps

**0. System packages** (once):

```bash
sudo apt install -y git cmake build-essential
```

You also need [Miniconda](https://docs.anaconda.com/miniconda/) installed.

**1. Create the Python environment:**

```bash
conda create -n isaaclab python=3.12 -y
conda activate isaaclab
pip install uv
```

**2. Clone Isaac Lab at the exact commit we use.** Put it next to this repository
(for example, both inside `~/projects/`):

```bash
cd ~/projects
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 50c445b143ee40514090303923188f9536be85bc
```

**3. Install Isaac Sim, Isaac Lab and all dependencies from our lock file.**
Run this **from inside the `IsaacLab` folder**, because the lock file installs
Isaac Lab from `./source/...`. Change the path to the lock file if your
`go2-box-climb` folder is somewhere else.

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" \
  --torch-backend cu128 \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow \
  -r ../go2-box-climb/env/isaaclab-requirements.lock.txt
```

This downloads several GB and can take a while.

**4. Apply the two version pins of the tested setup:**

```bash
python -m pip install --force-reinstall --no-deps "numpy==2.3.1" "coverage==7.7.1"
```

**5. Install this project's package** (editable, so code changes apply
immediately), from the `go2-box-climb` folder:

```bash
cd ../go2-box-climb
python -m pip install -e ".[dev]"
python -m pytest -q        # fast tests, no simulator needed; should all pass
```

## Check that it works

Accept the NVIDIA license once in every new terminal (or add it to `~/.bashrc`):

```bash
export OMNI_KIT_ACCEPT_EULA=yes
```

**A. Open an empty simulator window** (from the `IsaacLab` folder). The first
launch can take over 10 minutes while extensions are downloaded and cached.

```bash
python scripts/tutorials/00_sim/create_empty.py --viz kit
```

You should see a window with an empty (black) viewport.

**B. Run a tiny Go2 training** (3 iterations, no window). The Go2 robot model
is downloaded from NVIDIA's servers the first time, so you need internet.

```bash
python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 64 --max_iterations 3
```

It should print 3 training iterations and finish without errors. Logs go to
`IsaacLab/logs/rsl_rl/`.

**C. Confirm the versions:**

```bash
python -m pip list 2>/dev/null | grep -E "^(isaacsim|isaaclab|torch|rsl-rl-lib|numpy) "
```

Expected: `isaacsim 6.0.1.0`, `isaaclab 15.3.0`, `torch 2.11.0+cu128`,
`rsl-rl-lib 5.4.1`, `numpy 2.3.1`.

**D. Open our scene** (from the `go2-box-climb` folder): see "Viewing the
simulation" in the [README](../README.md).

## Common problems

- **`uv export --locked` fails / "lockfile needs to be updated".** Do not
  regenerate the lock from Isaac Lab's `uv.lock`. Always use our file in `env/`.
- **Window does not open / first start looks frozen.** The first start is slow
  (shader and extension caching). Wait, then try again.
- **"No such file ./source/isaaclab".** You ran step 3 outside the `IsaacLab`
  folder.
- **`undefined symbol: cudaGetDriverEntryPointByVersion` when streaming.**
  Torch was imported after Isaac Sim started. Always launch through
  `go2_box_climb.utils.launch`, which imports torch first.
