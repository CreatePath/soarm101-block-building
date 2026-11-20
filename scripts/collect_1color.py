"""Record the balanced single-candidate dataset described in docs/data/1color."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_env_so101.ik_control import solve_ik

if __package__:
    from .board_env import BLOCKS, BlockBoardEnv, HOME_RAD, IMAGE_HEIGHT, IMAGE_WIDTH, Layout
    from .collect_pilot import create_dataset, save_frames, simulate
else:
    from board_env import BLOCKS, BlockBoardEnv, HOME_RAD, IMAGE_HEIGHT, IMAGE_WIDTH, Layout
    from collect_pilot import create_dataset, save_frames, simulate

REPO_ID = "local/so101_1color"
GRID = np.array([4, 3])
LEFT = np.array([[.260, .095], [.350, .223]])
RIGHT = np.array([[.420, .140], [.615, .265]])
TARGET_EPISODES = 480


def schedule(seed):
    rng = np.random.default_rng(seed)
    cells = np.arange(np.prod(GRID))
    matching, idle = [], []
    for ref in BLOCKS:
        for match in (True, False):
            candidates = [ref] * 48 if match else [c for c in BLOCKS if c != ref for _ in range(4)]
            count = len(candidates) // len(cells)
            rows = zip(rng.permutation(candidates), rng.permutation(np.tile(cells, count)),
                       rng.permutation(np.tile(cells, count)), strict=True)
            (matching if match else idle).extend((ref, str(c), int(l), int(r)) for c, l, r in rows)
    rng.shuffle(matching)
    rng.shuffle(idle)
    result = []
    for i, negative in enumerate(idle):
        batch = matching[4*i:4*i+4] + [negative]
        rng.shuffle(batch)
        result.extend(batch)
    return result


def sample_xy(rng, bounds, cell):
    col_row = np.array([cell % GRID[0], cell // GRID[0]])
    return tuple(bounds[0] + (col_row + rng.random(2)) * (bounds[1] - bounds[0]) / GRID)


def verify_layout(env, renderer, qpos):
    m, d = env.model, env.data
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    addresses = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    block = m.geom("candidate_block").id
    paper = m.geom("paper").id
    if (np.max(np.abs(qpos[addresses] - HOME_RAD)) >= .03
            or any(block in (a, b) and paper not in (a, b) for a, b in d.contact.geom)):
        return False, [], []
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(d, camera="top")
        objects = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
    pixels, errors = [], []
    for name in ("reference_block", "candidate_block"):
        geom = m.geom(name).id
        pixels.append(int(np.count_nonzero((objects[:,:,0] == geom) & (objects[:,:,1] == mujoco.mjtObj.mjOBJ_GEOM))))
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        target = d.geom_xpos[geom].copy()
        target[2] = .014
        arm = solve_ik(m, d, env._ee_site_id, env._arm_dof_ids, target, seed_q=env._top_down_seed(),
                       point_offset=env._ik_point_offset, approach_axis=0, ik_damping=.001, ik_iterations=70)
        d.qpos[env._arm_dof_ids] = arm
        mujoco.mj_forward(m, d)
        errors.append(float(np.linalg.norm(env.ik_point_pos - target)))
    return min(pixels) >= 80 and max(errors) <= .003, pixels, errors


def record_one(dataset, index, assignment, seed):
    ref, candidate, left, right = assignment
    rng = np.random.default_rng([seed, index])
    for attempt in range(1, 101):
        layout = Layout(ref, candidate, sample_xy(rng, LEFT, left), sample_xy(rng, RIGHT, right))
        with BlockBoardEnv(layout) as env:
            if ref != candidate:
                # Test this candidate's graspability on an independent reset.
                # Only the subsequent stationary trajectory is recorded.
                _, _, probe = simulate(env, pick=True)
                if not probe["pinched_in_air"]:
                    continue
            valid, records, metrics = simulate(env)
            # A released block can slide a few mm away from its commanded
            # position. Retry the same layout before sampling another point.
            offset = 0.
            for _ in range(3):
                if valid or not metrics["pinched_in_air"] or metrics["line_error_m"] > .018:
                    break
                offset += env.data.geom("centre_ink").xpos[0] - metrics["final_xyz_m"][0]
                valid, records, metrics = simulate(env, release_offset_x=offset)
            if not valid:
                continue
            if np.linalg.norm(np.array(metrics["start_xyz_m"][:2]) - layout.candidate_xy) > .001:
                continue
            with mujoco.Renderer(env.model, IMAGE_HEIGHT, IMAGE_WIDTH) as renderer:
                usable, pixels, errors = verify_layout(env, renderer, records[0][0])
                if not usable:
                    continue
                save_frames(dataset, env, records, renderer)
            entry = dict(index=index, seed=seed, attempts=attempt, match=ref == candidate,
                         left_cell=left, right_cell=right, visible_pixels=pixels,
                         ik_error_m=errors, **asdict(layout), **metrics)
            return entry
    raise RuntimeError(f"No verified demonstration for {index}, {assignment}")


def collect(run: Path, seed: int):
    # Keep the original 240 assignments as a prefix when extending an existing run.
    plan = schedule(seed) + schedule(seed + 1)
    run.mkdir(parents=True, exist_ok=True)
    root = run / "dataset"
    manifest = run / "episodes.jsonl"
    if root.exists():
        previous = [json.loads(s) for s in manifest.read_text().splitlines()]
        if any(ep["seed"] != seed for ep in previous):
            raise ValueError("Use the original seed to resume this run.")
        if any(ep.get("home_command_rad") != HOME_RAD.tolist() for ep in previous):
            raise ValueError("The recorded home pose differs. Use a new run directory.")
        assignments = [(e["reference"], e["candidate"], e["left_cell"], e["right_cell"]) for e in previous]
        if assignments != plan[:len(previous)]:
            raise ValueError("The recorded assignments differ from the collection schedule.")
        dataset = LeRobotDataset.resume(REPO_ID, root=root, image_writer_threads=4)
        if len(previous) != dataset.meta.total_episodes:
            raise ValueError("Dataset and manifest episode counts differ.")
        start = len(previous)
    else:
        dataset = create_dataset(root, REPO_ID, image_writer_threads=4)
        start = 0
    try:
        with manifest.open("a", encoding="utf-8") as log:
            for index in range(start, len(plan)):
                entry = record_one(dataset, index, plan[index], seed)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(f"{index+1}/{len(plan)} {entry['reference']}/{entry['candidate']} attempt={entry['attempts']}", flush=True)
    finally:
        dataset.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--run", type=Path, help="Output directory, or a finalized partial run to resume.")
    args = parser.parse_args()
    run = args.run or Path("outputs/data/1color") / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(run.resolve(), flush=True)
    collect(run.resolve(), args.seed)
