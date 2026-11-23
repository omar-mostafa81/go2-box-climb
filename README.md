# go2-box-climb

A Unitree Go2 in a warehouse climbs onto a 0.5 m staging platform by pushing a
0.25 m cardboard delivery box against it and jumping ground → box → platform. Simulated in Isaac Lab. Course project for
ROB-GY 6323.

- **Install:** [docs/installation.md](docs/installation.md)
- **How everything works:** [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)
- **Rules for everyone (and every AI agent):** [CLAUDE.md](CLAUDE.md)

## Viewing the simulation

Every script that starts Isaac Sim has the same `--mode` option:

| Mode | What happens | Use it for |
| --- | --- | --- |
| `--mode gui` (default) | Opens the Isaac Sim window on this computer | Working at the computer that runs the sim |
| `--mode headless` | No window at all; fastest | Training, batch evaluation |
| `--mode stream` | No local window; the view is streamed over WebRTC | Watching from another computer (laptop) |

Example (from the repository root, `conda activate isaaclab`, and
`export OMNI_KIT_ACCEPT_EULA=yes` once per terminal):

```bash
python scripts/view_scene.py --mode gui
python scripts/view_scene.py --mode stream
python scripts/view_scene.py --mode headless --duration 5
```

### Choosing the world (`--scene`)

| Scene | What it is |
| --- | --- |
| `--scene basic` (default) | Box and platform on a plain floor. Loads fast. |
| `--scene warehouse` | Same task inside a full warehouse: cardboard delivery box, staging platform with pallets, racks, forklift. The final test world. |

```bash
python scripts/view_scene.py --scene warehouse --mode gui
```

The first warehouse start downloads about 1,800 3D-model files from NVIDIA
(several minutes). They are cached in `/tmp`, so later starts are fast.

### Streaming to another computer

1. On the viewing computer, install the **Isaac Sim WebRTC Streaming Client**
   (download it from NVIDIA's Isaac Sim documentation, "Livestream Clients" page).
2. On the simulation computer, run a script with `--mode stream`. It prints:
   `[STREAM] ... connect to: <IP address>`. This is the IP of the computer
   running the simulation, detected automatically, so nobody needs to configure
   it. Each teammate simply uses the IP their own script prints.
3. Wait until the script prints `Scene running`, then open the client, enter
   that IP and connect.

Both computers must be on the same network (or VPN). If the client cannot
connect, check that the firewall on the simulation computer allows ports
**49100 (TCP)** and **47998 (UDP)**. Only one client can be connected at a time.

### Recording a video (`--video`)

When you cannot watch live (for example on a cluster), add `--video` to a
playback script. It saves what the camera sees as an MP4 in `videos/` (ignored
by git). It also works with `--mode headless`: no window is opened.

```bash
python scripts/locomotion/play_keyboard.py --mode headless --video --auto "0.3,0,0,8,0.27;0,0,0,2"
python scripts/locomotion/play.py --stage a --mode headless --video --duration 10 --checkpoint logs/rsl_rl/go2_walk/<run>/model_999.pt
```

The script prints `[VIDEO] Saved ... to: .../videos/<name>_<date_time>.mp4`.
Copy that file to your computer to watch it, for example with
`scp <user>@<cluster>:<path to repo>/videos/*.mp4 .`. Recording works in
`scripts/locomotion/play.py` and `play_keyboard.py` so far. Other scripts accept
the flag but do not record yet.

## Repository layout

See "Repository layout" in [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md).
