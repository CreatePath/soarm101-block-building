"""Evaluate a local SmolVLA checkpoint on novel paired board layouts in MuJoCo."""

import argparse
import json
from dataclasses import asdict
from itertools import permutations
from pathlib import Path

import mujoco
import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .board_env import BLOCKS, FPS, IMAGE_HEIGHT, IMAGE_WIDTH, BlockBoardEnv, Layout, camera_frame
from .collect_1color import GRID, LEFT, RIGHT, sample_xy, verify_layout
from .collect_pilot import JOINTS, TASK, simulate
from .smolvla_640 import enable_640x480


def verified_pairs(seed):
    rng = np.random.default_rng(seed)
    cells = int(GRID.prod())
    left_cells, right_cells = rng.permutation(cells), rng.permutation(cells)
    for index, (candidate, other) in enumerate(permutations(BLOCKS, 2)):
        for _ in range(100):
            left = sample_xy(rng, LEFT, left_cells[index])
            right = sample_xy(rng, RIGHT, right_cells[index])
            layouts = [Layout(ref, candidate, left, right) for ref in (candidate, other)]
            lengths = []
            for layout in layouts:
                with BlockBoardEnv(layout) as env, mujoco.Renderer(env.model, IMAGE_HEIGHT, IMAGE_WIDTH) as renderer:
                    valid, records, _ = simulate(env)
                    if not valid or not verify_layout(env, renderer, records[0][0])[0]:
                        break
                    lengths.append(len(records))
            if len(lengths) == 2:
                yield layouts, lengths
                break
        else:
            raise RuntimeError(f"No valid layout pair for {candidate}/{other}")


def policy_commands(env, renderer, policy, preprocess, postprocess, horizon):
    joints = [env.model.joint(name).qposadr[0] for name in JOINTS]
    motors = [env.model.actuator(name).id for name in JOINTS]
    remaining = horizon
    while remaining:
        observation = {"task": TASK, "observation.state": torch.from_numpy(env.data.qpos[joints].astype("float32"))}
        for key, camera in (("top", "top"), ("gripper", "gripper_view")):
            rgb = torch.from_numpy(camera_frame(renderer, env.data, camera))
            observation[f"observation.images.{key}"] = rgb.permute(2, 0, 1).float().div_(255)
        batch = preprocess(observation)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk = policy.predict_action_chunk(batch)
        actions = postprocess(chunk)[0, :min(policy.config.n_action_steps, remaining)].float().cpu().numpy()
        if actions.shape[1:] != (len(JOINTS),) or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("Invalid joint targets returned by the checkpoint")
        for angles in actions:
            control = env.data.ctrl.copy()
            control[motors] = angles
            yield control
        remaining -= len(actions)


@torch.inference_mode()
def evaluate(checkpoint, seed):
    enable_640x480()
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint), strict=True).eval()
    preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    rows = []
    for index, (layouts, lengths) in enumerate(verified_pairs(seed)):
        for layout, horizon in zip(layouts, lengths, strict=True):
            match = layout.candidate == layout.reference
            torch.manual_seed(seed + index)
            policy.reset()
            preprocess.reset()
            postprocess.reset()
            with BlockBoardEnv(layout) as env, mujoco.Renderer(env.model, IMAGE_HEIGHT, IMAGE_WIDTH) as renderer:
                def commands(board):
                    return policy_commands(board, renderer, policy, preprocess, postprocess, horizon)

                strict_success, _, metrics = simulate(env, controller=commands)
            # A learned continuous controller can return to the physical home pose
            # without reproducing its last floating-point motor target bit for bit.
            success = (metrics["released_on_line"] and metrics["home_error_rad"] < .01) if match else strict_success
            rows.append(dict(pair=index, **asdict(layout), match=match, success=bool(success),
                             collection_success=bool(strict_success), **metrics))
            print(f"{len(rows):2d}/24 {layout.reference}/{layout.candidate} success={bool(success)} "
                  f"lift={metrics['lift_m']*1000:.1f} mm, max joint motion={np.rad2deg(metrics['max_joint_motion_rad']):.1f} deg",
                  flush=True)
    return dict(checkpoint=str(checkpoint), seed=seed, fps=FPS, n_action_steps=policy.config.n_action_steps,
                matches_succeeded=sum(r["success"] for r in rows if r["match"]),
                mismatches_succeeded=sum(r["success"] for r in rows if not r["match"]), episodes=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="The checkpoint's pretrained_model directory")
    parser.add_argument("--seed", type=int, default=902173)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint.resolve(strict=True), args.seed)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Match {result['matches_succeeded']}/12, mismatch {result['mismatches_succeeded']}/12")
