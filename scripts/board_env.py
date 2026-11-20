"""SO-101 experimental board. Run this module to open the interactive MuJoCo viewer."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from gymnasium import Env, spaces
from lerobot_env_so101.mujoco_env import ASSETS_DIR, GymRenderingSpec, SO101GymEnv
from lerobot_env_so101.policy import DEFAULT_ACTION_SCALE


FPS = 30
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SCENE = Path(__file__).resolve().parents[1] / "scenes" / "board.xml"
# Calibrated model angles for the folded pose in the experiment photograph.
# Order: pan, shoulder, elbow, wrist flex, wrist roll, jaw.
HOME_RAD = np.deg2rad([0., -94., 89.5, 65., 0., -9.])
BLOCKS = {
    "red": (.030, ".80 .055 .035 1"),
    "yellow": (.025, ".98 .66 .035 1"),
    "green": (.0225, ".055 .47 .17 1"),
    "blue": (.020, ".035 .15 .67 1"),
}


@dataclass(frozen=True)
class Layout:
    reference: str
    candidate: str
    reference_xy: tuple[float, float]
    candidate_xy: tuple[float, float]

    @classmethod
    def sample(cls, rng, match=None):
        colors = list(BLOCKS)
        reference = str(rng.choice(colors))
        if match is None:
            match = rng.random() < .5
        candidate = reference if match else str(rng.choice([c for c in colors if c != reference]))
        # Compact workspaces on either side of x=.3875, within 28 cm of the base.
        left = tuple(rng.uniform([.314, .145], [.345, .185]))
        right = tuple(rng.uniform([.434, .155], [.485, .205]))
        return cls(reference, candidate, left, right)


def write_scene(directory: Path, layout: Layout) -> Path:
    robot = ET.parse(ASSETS_DIR / "so101_new_calib.xml")
    robot.find("compiler").set("meshdir", (ASSETS_DIR / "assets").as_posix())
    base = robot.find(".//body[@name='base']")
    base.set("pos", ".5175 0 .004")
    base.set("quat", ".70710678 0 0 .70710678")
    for material in robot.findall("asset/material"):
        motor = "sts3215" in material.get("name")
        material.set("rgba", ".095 .095 .1 1" if motor else ".068 .07 .075 1")

    # Express the wrist mount in gripperframe coordinates so the lens moves with the jaws.
    gripper = robot.find(".//body[@name='gripper']")
    site = gripper.find("site[@name='gripperframe']")
    frame = ET.SubElement(gripper, "frame", pos=site.get("pos"), quat=site.get("quat"))
    ET.SubElement(frame, "geom", type="capsule", fromto="-.065 0 .012 -.087 -.068 .020",
                  size=".0045", material="housing", contype="0", conaffinity="0", mass="0")
    # Nominal 25-degree optical tilt of the SO-101 integrated 32 mm UVC mount,
    # mirrored to the photographed side. Translation remains a pilot estimate.
    mount = ET.SubElement(frame, "body", name="gripper_camera_housing", pos="-.08 -.06 .02",
                          xyaxes="0 0 1 -.422618262 .906307787 0")
    ET.SubElement(mount, "geom", type="box", pos="0 0 .014", size=".016 .012 .009",
                  material="housing", contype="0", conaffinity="0", mass="0")
    ET.SubElement(mount, "geom", type="cylinder", pos="0 0 .003", size=".006 .0028",
                  material="lens", contype="0", conaffinity="0", mass="0")
    ET.SubElement(mount, "camera", name="gripper_view", fovy="54", resolution="640 480")

    # The upstream IK controller loads MJCF by path. Build a private include next to the
    # scene so parallel runs can share the installed meshes without modifying the package.
    scene_path = directory / SCENE.name
    scene = ET.parse(SCENE)
    for name in ("reference", "candidate"):
        body = scene.find(f"worldbody/body[@name='{name}']")
        x, y = getattr(layout, name + "_xy")
        half_length, rgba = BLOCKS[getattr(layout, name)]
        body.set("pos", f"{x} {y} .01")
        body.find("geom").set("size", f"{half_length} .015 .01")
        body.find("geom").set("rgba", rgba)
    scene.write(scene_path, encoding="utf-8")
    robot.write(directory / "robot.xml", encoding="utf-8")
    return scene_path


class BlockBoardEnv(SO101GymEnv):
    def __init__(self, layout=None):
        self.layout = layout or Layout.sample(np.random.default_rng())
        with TemporaryDirectory() as tmp:
            super().__init__(
                xml_path=write_scene(Path(tmp), self.layout), control_dt=1 / FPS, physics_dt=1 / (FPS * 16),
                render_spec=GymRenderingSpec(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                action_scale=DEFAULT_ACTION_SCALE, jaw_pads=True, top_down_ik=True,
                home_position=HOME_RAD[:5].copy(),
                cartesian_bounds=np.array([[0., 0., 0.], [.775, .54, .45]]),
            )
        # The upstream render() selects its camera through this ID.
        self._camera_id = self.model.camera("top").id
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(-np.inf, np.inf, self.get_robot_state().shape, np.float32),
            "candidate_pos": spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })

    def observation(self):
        return {
            "agent_pos": self.get_robot_state().astype(np.float32),
            "candidate_pos": self.data.body("candidate").xpos.astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        Env.reset(self, seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.model.joint("gripper").qposadr[0]] = HOME_RAD[-1]
        self.data.ctrl[self.model.actuator("gripper").id] = HOME_RAD[-1]
        self.reset_robot()
        return self.observation(), {}

    def step(self, action):
        self.apply_action(action)
        return self.observation(), 0.0, False, False, {}


def camera_frame(renderer, data, name):
    renderer.update_scene(data, camera=name)
    return renderer.render().copy()


if __name__ == "__main__":
    import argparse
    import time
    import mujoco.viewer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("top", "gripper_view"), help="Start in a fixed camera view.")
    args = parser.parse_args()

    with BlockBoardEnv() as board:
        board.reset()
        with mujoco.viewer.launch_passive(
            board.model, board.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = board.model.stat.center
                viewer.cam.azimuth = board.model.vis.global_.azimuth
                viewer.cam.elevation = board.model.vis.global_.elevation
                viewer.cam.distance = 1.30
                if args.camera:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = board.model.camera(args.camera).id
            while viewer.is_running():
                start = time.monotonic()
                with viewer.lock():
                    mujoco.mj_step(board.model, board.data, nstep=16)
                viewer.sync()
                time.sleep(max(0, start + 1 / FPS - time.monotonic()))
