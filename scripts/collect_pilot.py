"""Collect five colour-conditioned demonstrations from the simulated board."""

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_env_so101.ik_control import solve_ik

if __package__:
    from .board_env import BlockBoardEnv, FPS, IMAGE_HEIGHT, IMAGE_WIDTH, Layout, camera_frame
else:
    from board_env import BlockBoardEnv, FPS, IMAGE_HEIGHT, IMAGE_WIDTH, Layout, camera_frame

DATASET_ID = "local/so101_color_match"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASK = "If the right block matches the left block's color, place the right block on the center line. Otherwise hold still."


def pick_and_place(env, goal):
    model, data = env.model, env.data
    block = data.body("candidate")
    x, y = block.xpos[:2]
    home = data.ctrl.copy()

    def move(target, jaw_angle, frames):
        # Solve the grasp midpoint at each waypoint, then smoothly drive the six motors.
        arm = solve_ik(model, data, env._ee_site_id, env._arm_dof_ids, np.asarray(target),
                       seed_q=env._top_down_seed(), point_offset=env._ik_point_offset,
                       approach_axis=0, ik_damping=.001, ik_iterations=45)
        start = data.ctrl.copy()
        end = start.copy()
        end[env._arm_ctrl_ids] = arm
        end[model.actuator("gripper").id] = jaw_angle
        for step in range(1, frames + 1):
            t = step / frames
            yield start + (end - start) * t**2 * (3 - 2*t)

    yield from move((x, y, .082), .85, 32)
    yield from move((x, y, .014), .85, 32)
    yield from move((x, y, .014), .02, 22)
    yield from move((x, y, .067), .02, 30)
    across = env.ik_point_pos.copy()
    across[:2] += goal[:2] - block.xpos[:2]
    yield from move(across, .02, 44)
    lower = env.ik_point_pos + goal - block.xpos
    lower[2] += .001
    yield from move(lower, .02, 28)
    yield from move(lower, .85, 22)
    yield from move((*lower[:2], .084), .85, 28)
    raised = data.ctrl.copy()
    for step in range(1, 43):
        t = step / 42
        yield raised + (home - raised) * t**2 * (3 - 2*t)
    for _ in range(24):
        yield home.copy()


def simulate(env, *, pick=None, release_offset_x=0., controller=None):
    model, data = env.model, env.data
    env.reset()
    mujoco.mj_step(model, data, nstep=1000)  # Let the home pose settle before recording.
    mujoco.mj_forward(model, data)
    block = data.body("candidate")
    start = block.xpos.copy()
    goal = start.copy()
    line_x = data.geom("centre_ink").xpos[0]
    goal[0] = line_x + release_offset_x
    ref = data.geom("reference_block").xpos
    clearance_x = sum(model.geom(name).size[0] for name in ("candidate_block", "reference_block"))
    if goal[0] - ref[0] < clearance_x + .006 and abs(goal[1] - ref[1]) < .04:
        goal[1] = ref[1] + (.044 if ref[1] < .18 else -.044)
    joints = [model.joint(name).qposadr[0] for name in JOINTS]
    initial_joints = data.qpos[joints].copy()
    initial_ctrl = data.ctrl.copy()
    home_tip = data.site("gripperframe").xpos.copy()
    match = env.layout.reference == env.layout.candidate if pick is None else pick
    if controller is not None:
        commands = controller(env)
    else:
        commands = pick_and_place(env, goal) if match else (data.ctrl.copy() for _ in range(2 * FPS))
    records = []
    peak_z = start[2]
    max_joint_motion = max_block_motion = 0.
    block_id = model.geom("candidate_block").id
    paper_id = model.geom("paper").id
    robot_jaws = {model.body("gripper").id, model.body("moving_jaw_so101_v1").id}
    pinched_in_air = False
    for action in commands:
        records.append((data.qpos.copy(), action.copy()))
        data.ctrl[:] = action
        mujoco.mj_step(model, data, nstep=16)
        mujoco.mj_forward(model, data)
        contacts = {int(b if a == block_id else a) for a, b in data.contact.geom if block_id in (a, b)}
        touching_bodies = {model.geom_bodyid[g] for g in contacts}
        rise = block.xpos[2] - start[2]
        pinched_in_air |= rise > .03 and robot_jaws <= touching_bodies
        peak_z = max(peak_z, block.xpos[2])
        max_joint_motion = max(max_joint_motion, np.max(np.abs(data.qpos[joints] - initial_joints)))
        max_block_motion = max(max_block_motion, np.linalg.norm(block.xpos - start))

    final = block.xpos.copy()
    line_error = abs(final[0] - line_x)
    home_error = max(np.max(np.abs(data.qpos[joints] - initial_joints)),
                     np.max(np.abs(records[-1][0][joints] - initial_joints)))
    home_command = np.array_equal(data.ctrl, initial_ctrl)
    if match:
        placed = (pinched_in_air and line_error <= .003 and abs(final[2] - start[2]) < .001
                  and paper_id in contacts and robot_jaws.isdisjoint(touching_bodies)
                  and np.linalg.norm(data.qvel[model.joint("candidate_free").dofadr[0]:][:6]) < .005)
        valid = home_command and home_error < .01 and placed
    else:
        valid = max_joint_motion < 1e-4 and max_block_motion < 1e-5
    metrics = dict(start_xyz_m=start.tolist(), final_xyz_m=final.tolist(), lift_m=peak_z-start[2],
                   home_command_rad=initial_ctrl[[model.actuator(name).id for name in JOINTS]].tolist(),
                   home_tip_xyz_m=home_tip.tolist(),
                   pinched_in_air=bool(pinched_in_air),
                   line_error_m=line_error, max_joint_motion_rad=max_joint_motion,
                   home_error_rad=home_error, final_command_is_home=home_command,
                   max_block_motion_m=max_block_motion, frames=len(records))
    if controller is not None and match:
        metrics["released_on_line"] = bool(placed)
    return valid, records, metrics


def create_dataset(root, repo_id=DATASET_ID, image_writer_threads=0):
    image = dict(dtype="image", shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), names=["height", "width", "channels"])
    angles = dict(dtype="float32", shape=(len(JOINTS),), names=JOINTS)
    return LeRobotDataset.create(repo_id, fps=FPS, root=root, use_videos=False, robot_type="so101",
                                    image_writer_threads=image_writer_threads,
                                    features={"observation.images.top": image, "observation.images.gripper": image,
                                              "observation.state": angles, "action": angles})


def save_frames(dataset, env, records, renderer):
    joints = [env.model.joint(name).qposadr[0] for name in JOINTS]
    actuators = [env.model.actuator(name).id for name in JOINTS]
    for qpos, ctrl in records:
        env.data.qpos[:] = qpos
        mujoco.mj_forward(env.model, env.data)
        dataset.add_frame({
            "observation.images.top": camera_frame(renderer, env.data, "top"),
            "observation.images.gripper": camera_frame(renderer, env.data, "gripper_view"),
            "observation.state": qpos[joints].astype(np.float32),
            "action": ctrl[actuators].astype(np.float32), "task": TASK,
        })
    dataset.save_episode()


def record_episodes(root: Path, seed=2081):
    rng = np.random.default_rng(seed)
    dataset = create_dataset(root)
    episodes = []
    try:
        layouts = [Layout.sample(rng, match) for match in (True, False, True, False, True)]
        for index, layout in enumerate(layouts):
            match = layout.reference == layout.candidate
            for attempt in range(1, 31):
                with BlockBoardEnv(layout) as env:
                    valid, records, metrics = simulate(env)
                    if valid:
                        # Render the validated pre-action physics states, including idle frames.
                        with mujoco.Renderer(env.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH) as renderer:
                            save_frames(dataset, env, records, renderer)
                        episodes.append(dict(index=index, match=match, attempts=attempt, **asdict(layout), **metrics))
                        print(f"Saved episode {index}: {layout.reference}/{layout.candidate}, {len(records)} frames", flush=True)
                        break
                new = Layout.sample(rng, match)
                layout = replace(new, reference=layout.reference, candidate=layout.candidate)
            else:
                raise RuntimeError(f"No verified demonstration for episode {index}: {metrics}")
    finally:
        dataset.finalize()
    (root.parent / "episodes.json").write_text(json.dumps(dict(seed=seed, episodes=episodes), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2081)
    args = parser.parse_args()
    run = Path("outputs/pilot") / datetime.now().strftime("color-match-%Y%m%d-%H%M%S-%f")
    print(run.resolve(), flush=True)
    record_episodes(run.resolve() / "dataset", args.seed)
