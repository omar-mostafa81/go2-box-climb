# Working Rules for This Project

Every team member's AI agent reads this file at the start of every session.
These rules are set by the team leader and apply to **everyone**. Follow all of
them. New rules are added to the end of the "Rules" list.

## Project in one line

A Unitree Go2 in Isaac Lab climbs a 0.5 m warehouse platform by pushing a 0.25 m box against
it (MPPI + PPO walking policy) and then jumping ground → box → platform
(trajectory optimization + PPO tracker, compared against RL from scratch), with a
value-based sequencer choosing the next skill. Full proposal:
`proposal/RL_OC_Project_Proposal.pdf`. Course: ROB-GY 6323, 19 Nov – 12 Dec 2026.

## Where things are

- How the project works, part by part: `docs/PROJECT_REPORT.md` (read it first).
- Installing the exact same software: `docs/installation.md`.
- Shared contracts every part must use: `go2_box_climb/common/` (robot settings,
  scene geometry, state format, skill interface, file formats).

## Rules

1. **The goal is the team member's understanding, not just finished code.** The
   person you work with must understand every part of the project and the
   repository. Explain what you do and why. Do not run ahead and build large
   pieces they did not ask for.

2. **Work one milestone at a time.** Only implement what was asked for in the
   current step. Wait for instructions before starting the next part.

3. **Keep the project report up to date.** The report lives at
   `docs/PROJECT_REPORT.md`. After every completed milestone, update the
   section for that part before saying the milestone is done. Only edit the
   sections of the part you are working on.

4. **What goes in the report: final results only.** Write the final design
   choices, how things work, and the results. Do **not** write the back-and-forth
   of discussions, abandoned ideas, or the history of changes.

5. **How the report is written.**
   - Simple, student-friendly wording. Short sentences. Explain any jargon the
     first time it appears.
   - Still include the high-level technical details: how each part works and
     why that design was chosen.

6. **What each part of the report must cover** (where it applies to that part):
   - What the part does and how it fits into the full task.
   - Design choices and the reason for each one.
   - Observations (what the policy/controller sees) and control inputs/actions
     (what it outputs), with units and dimensions.
   - Rewards / cost functions and termination conditions.
   - How training runs: algorithm, optimizer, learning rate, network size,
     number of environments, training steps, and other key hyperparameters.
   - Randomization (friction, pushes, box placement, etc.).
   - Plots: training losses, rewards, success rates, and study results. Save
     plot images in `docs/figures/` and link them from the report.
   - Results: numbers and a short explanation of what they mean.
   - How to run it: exact commands to train, evaluate, test, and visualize.
   - Where the code is: the files and folders that belong to this part.

7. **Keep the repository map in the report current.** When new folders or
   important files are added, update the "Repository layout" section of the
   report so everyone knows what each part of the directory is for.

8. **Never commit or push.** The agent never runs `git commit`, `git push`, or
   anything else that creates commits or changes the remote. The team member
   manages all commits and pushes. Instead:
   - When a piece of work is finished, remind them to commit and push it.
   - Keep track of what is pushed and what is not. Before reporting work as
     done (and when starting a session), check with `git status` and
     `git log origin/main..HEAD` (after `git fetch`). Warn about uncommitted
     changes or local commits that are not on GitHub yet.

9. **Never push training output; only final policies and references.** Logs, intermediate
   checkpoints, TensorBoard files, videos and other run output must always be
   in `.gitignore` (folders such as `logs/` and `outputs/`). Any new tool or
   script that writes output must write it to an ignored folder, and you add
   the folder to `.gitignore` if it is not covered yet. The only model files
   that go into git are **final, released policies** in `policies/`, each with
   a short note on how it was trained, and the **final jump reference
   trajectories** in `references/`.

10. **Respect the shared contracts.** Never redefine robot settings (control
    time step, PD gains, action scale, default joint pose, joint order) or scene
    geometry in your own files: import them from `go2_box_climb.common`. Do not
    change anything in `go2_box_climb/common/` without the team leader's
    approval, because every part depends on it.

11. **Stay inside your part's folders.** Only edit the folders that belong to
    the part you are working on. If you need a change in another part, tell the
    team member so they can coordinate with its owner.

12. **Every script that starts Isaac Sim supports three viewing modes**, using
    the shared helper `go2_box_climb.utils.launch`:
    `--mode headless` (no window), `--mode gui` (Isaac Sim window), and
    `--mode stream` (WebRTC livestream to view from another computer). Do not
    write a separate launcher in your own script.

13. **One robot model for everyone.** Trajectory optimization, IK, the MPPI
    model and the sequencer may import Isaac (`isaacsim`/`isaaclab`) or the USD
    library (`pxr`). Robot geometry (link lengths, joint offsets, foot size,
    head position) must come from the same Go2 USD the simulator uses, never
    typed in by hand: get it with `go2_geometry()` from
    `go2_box_climb.common.robot_geometry`, which reads that exact file with
    `pxr` without starting the simulator. If you need a value it does not
    provide yet, add it there (team leader approval, rule 10). Keep the math
    itself in plain functions that take numbers as inputs, so it can be
    unit-tested quickly.
