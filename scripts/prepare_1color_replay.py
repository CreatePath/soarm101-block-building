"""Export local 1color actions for lerobot-replay on a degree-mode SO-101 follower."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets import LeRobotDataset
from lerobot_env_so101.gripper import ctrl_to_normalized
from lerobot_env_so101.mujoco_env import ASSETS_DIR

from .collect_pilot import JOINTS, TASK


def prepare(run, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    root = run / "dataset"
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["robot_type"] != "so101" or info["features"]["action"]["names"] != JOINTS:
        raise ValueError("Expected a 1color simulation dataset with six named radian targets")
    with (run / "episodes.jsonl").open(encoding="utf-8") as src:
        episodes = list(map(json.loads, src))
    if len(episodes) != info["total_episodes"] or not episodes:
        raise ValueError("Episode count differs from dataset metadata or dataset is empty")
    model = ElementTree.parse(ASSETS_DIR / "so101_new_calib.xml")
    limits = np.array([model.find(f".//actuator/position[@name='{joint}']").get("ctrlrange").split()
                       for joint in JOINTS], dtype=float)
    features = {"action": {"dtype": "float32", "shape": (len(JOINTS),), "names": [f"{j}.pos" for j in JOINTS]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        dataset = LeRobotDataset.create("local/so101_1color_replay", fps=info["fps"], root=stage,
                                        robot_type="so101_follower", use_videos=False, features=features)
        try:
            for index, episode in enumerate(episodes):
                table = pq.read_table(root / "data", columns=["frame_index", "action"],
                                      filters=[("episode_index", "=", index)]).sort_by("frame_index")
                n = episode["frames"]
                if episode["index"] != index or n < 1 or table["frame_index"].to_pylist() != list(range(n)):
                    raise ValueError(f"Incomplete or misordered episode {index}")
                radians = np.array(table["action"].to_pylist())
                if (radians.shape != (n, len(JOINTS)) or not np.isfinite(radians).all()
                        or (radians < limits[:, 0] - 1e-5).any() or (radians > limits[:, 1] + 1e-5).any()):
                    raise ValueError(f"Invalid joint target in episode {index}")
                # MJCF ctrlrange endpoints are rounded to five decimals.
                radians = np.clip(radians, limits[:, 0], limits[:, 1])
                targets = np.rad2deg(radians).astype(np.float32)
                targets[:, -1] = 100 * ctrl_to_normalized(radians[:, -1], limits[-1])
                for target in targets:
                    dataset.add_frame({"action": target, "task": TASK})
                dataset.save_episode()
        finally:
            dataset.finalize()
        stage.rename(output)
    print(f"{len(episodes)} episodes at {info['fps']} FPS -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="1color run containing dataset/ and episodes.jsonl")
    parser.add_argument("output", type=Path, help="New LeRobot dataset directory")
    args = parser.parse_args()
    prepare(args.run, args.output)
