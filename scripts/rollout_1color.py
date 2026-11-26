"""Replay one recorded 1color episode or run a local policy on its starting layout."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from .board_env import BlockBoardEnv, FPS, IMAGE_HEIGHT, IMAGE_WIDTH, Layout, camera_frame
from .collect_pilot import JOINTS, simulate


def recorded_commands(env, run, episode):
    table = pq.read_table(run / "dataset/data", columns=["frame_index", "action"],
                          filters=[("episode_index", "=", episode["index"])]).sort_by("frame_index")
    if table["frame_index"].to_pylist() != list(range(episode["frames"])):
        raise ValueError("Saved frame indices do not match the episode manifest")
    motors = [env.model.actuator(name).id for name in JOINTS]
    for angles in table["action"].to_pylist():
        command = env.data.ctrl.copy()
        command[motors] = angles
        yield command


def save_video(env, records, path):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (2 * IMAGE_WIDTH, IMAGE_HEIGHT))
    if not writer.isOpened():
        raise OSError(f"Cannot write video to {path}")
    try:
        with mujoco.Renderer(env.model, IMAGE_HEIGHT, IMAGE_WIDTH) as renderer:
            for qpos, _ in records:
                env.data.qpos[:] = qpos
                mujoco.mj_forward(env.model, env.data)
                views = [camera_frame(renderer, env.data, camera) for camera in ("top", "gripper_view")]
                writer.write(cv2.cvtColor(np.concatenate(views, axis=1), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Collection folder containing episodes.jsonl and dataset/")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, help="Local SmolVLA pretrained_model directory")
    parser.add_argument("--video", type=Path, help="Save top and gripper views side by side as MP4")
    args = parser.parse_args()
    with (args.run / "episodes.jsonl").open(encoding="utf-8") as manifest:
        episode = next((e for e in map(json.loads, manifest) if e["index"] == args.episode), None)
    if episode is None:
        parser.error(f"Episode {args.episode} is missing from the manifest")
    layout = Layout(**{key: episode[key] for key in ("reference", "candidate", "reference_xy", "candidate_xy")})
    with BlockBoardEnv(layout) as env:
        if args.checkpoint:
            import torch
            from .eval_1color import load_policy, policy_commands

            model = load_policy(args.checkpoint.resolve(strict=True))
            torch.manual_seed(episode["seed"] + args.episode)
            with torch.inference_mode(), mujoco.Renderer(env.model, IMAGE_HEIGHT, IMAGE_WIDTH) as renderer:
                valid, records, metrics = simulate(env, controller=lambda board: policy_commands(
                    board, renderer, *model, episode["frames"]))
        else:
            valid, records, metrics = simulate(env, controller=lambda board: recorded_commands(board, args.run, episode))
        success = (metrics["released_on_line"] and metrics["home_error_rad"] < .01) if episode["match"] else valid
        if args.video:
            save_video(env, records, args.video)
    print(json.dumps(dict(episode=args.episode, controller="policy" if args.checkpoint else "recorded_actions",
                          success=bool(success), **metrics), indent=2))


if __name__ == "__main__":
    main()
