"""Training-matched SAPIEN cube-stacking environments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

import dacite
import numpy as np
import sapien
import torch
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from so101_sim.envs import KitSlowMixin, _OPTICAL_CONV
from so101_sim.robots.so101_base.so101 import (
    ARM_FORCE_LIMIT,
    REAL_HOME_ARM_DEG,
    REAL_HOME_GRIPPER_PCT,
    SO101,
    grip_rad_from_pct,
)
from so101_sim.tasks.base_random_env import (
    BaseRandomEnv,
    DefaultCameraEnv,
    DefaultRandomizationConfig,
)
from so101_smolvla.sapien_config import TASK_SPECS
from so101_smolvla.sapien_grip import grip_facts
from so101_smolvla.stack_initial_states import (
    TRAINING_INITIAL_STATES,
    draw_stack_initial_state,
)
from so101_smolvla.stack_layouts import TRAINING_CUBE_SIZES, draw_stack_layout
from so101_smolvla.stack_recording_rig import ARM_JOINTS, RECORDING_RIG
from so101_smolvla.table_view import table_seen_from


# Paired episode 33 calibration: its real frame-zero joints and wrist view
# independently recover the cube layout, then this offset aligns both cubes in
# the top view. The prior (+101, +124) image-shift conversion used the wrong
# reference direction and placed the same scene about 70 px right / 117 px low.
STACK_TOP_CAMERA_OFFSET = (0.0397, -0.0157, 0.0)

COLOR_CUBE_SIZE = 0.040
CUBE_COLORS = {
    "red": (0.718, 0.306, 0.302, 1.0),
    "green": (0.20, 0.65, 0.25, 1.0),
    "blue": (0.612, 0.733, 0.820, 1.0),
}
ITEM_DENSITY = 200.0
# Parts of the simulated kit that the recording rig did not have in view.
RECORDING_RIG_ABSENT_LINKS = (
    "kit_base_plate_link",
    "frame_extrusion_link",
    "top_camera_mount_link",
)
STACK_XY_TOLERANCE = 0.020
STACK_Z_TOLERANCE = 0.010
STATIC_SPEED_TOLERANCE = 0.020


def camera_projections(env, name: str) -> np.ndarray:
    """Pixels from robot-base coordinates for one camera of a scene, per scene."""
    camera = env._sensors[name].camera
    return (
        (camera.get_intrinsic_matrix() @ camera.get_extrinsic_matrix())
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _real_state_to_sim_qpos(state: np.ndarray) -> np.ndarray:
    qpos = np.deg2rad(np.asarray(state, dtype=np.float64))
    qpos[-1] = grip_rad_from_pct(float(state[-1]))
    return qpos


@dataclass
class StackRandomizationConfig(DefaultRandomizationConfig):
    """Settings consumed by the KIT robot mixins and stacking task."""

    robot_qpos_noise_std: float = np.deg2rad(5)
    item_friction_range: Sequence[float] = (0.5, 1.0)
    item_density_range: Sequence[float] = (ITEM_DENSITY, ITEM_DENSITY)


class TrainingStackCameraMixin:
    """Apply the measured top-camera mount used by the real stacking split."""

    def _initialize_episode(self, env_idx, options: dict):
        super()._initialize_episode(env_idx, options)
        self._sensors["top"].camera.set_local_pose(
            sapien.Pose(p=list(STACK_TOP_CAMERA_OFFSET), q=_OPTICAL_CONV)
        )


class RecordingRigMixin:
    """Show the scene and report the joints as the real stacking recordings did.

    Every measured value comes from ``RECORDING_RIG``: where the top camera
    stood and its lens, where the wrist camera sat on the hand and its lens,
    the table picture the top camera saw, how far the table lay below the
    robot base, and where each arm joint of the recording robot had its zero.
    Both cameras show that one recorded table: the top camera as it took the
    picture, the wrist camera from where the hand is.
    The model reads and commands the recording robot's joint numbers; the
    simulated robot stands at those numbers plus the zero points. The arm has
    its weight, and its joint drives give way under it as the recording
    robot's motors did, so a reading stays behind its command as recorded.
    Cubes, sentence, start poses and placement stay those of the task this is
    mixed into.
    """

    TABLE_SURFACE_Z = -RECORDING_RIG.table_below_robot_base

    def __init__(self, *args, obs_mode: str = "rgb", **kwargs):
        if obs_mode != "rgb":
            raise ValueError(
                f"the recording rig renders pictures only, not {obs_mode!r}"
            )
        super().__init__(
            *args,
            # Segmentation tells robot and cube pixels from the table.
            obs_mode="rgb+segmentation",
            domain_randomization_config={
                "apply_overlay": True,
                "rgb_overlay_path": str(RECORDING_RIG.table_picture),
            },
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        configs = super()._default_sensor_configs
        lenses = {
            "top": RECORDING_RIG.top_camera_vertical_fov_rad,
            "wrist": RECORDING_RIG.wrist_camera_vertical_fov_rad,
        }
        for config in configs:
            config.fov = lenses[config.uid]
        return configs

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        joints = [joint.name for joint in self.agent.robot.active_joints]
        if tuple(joints[: len(ARM_JOINTS)]) != ARM_JOINTS:
            raise RuntimeError(
                f"the zero points are measured for {ARM_JOINTS}, "
                f"the simulated robot has {joints}"
            )
        self._arm_zero_points = torch.as_tensor(
            RECORDING_RIG.arm_zero_points_rad,
            dtype=torch.float32,
            device=self.device,
        )
        # The simulator library takes the weight off every robot link, so its
        # arm stands exactly on the command. The recording robot's arm had
        # weight and motors that give way under it.
        for link in self.agent.robot.links:
            link.disable_gravity = False
        for joint, stiffness, damping in zip(
            self.agent.robot.active_joints,
            RECORDING_RIG.arm_drive_stiffness,
            RECORDING_RIG.arm_drive_damping,
        ):
            joint.set_drive_properties(
                stiffness, damping, force_limit=ARM_FORCE_LIMIT, mode="force"
            )
        links = self.agent.robot.links_map
        for name in RECORDING_RIG_ABSENT_LINKS:
            for obj in links[name]._objs:
                render_body = obj.entity.find_component_by_type(RenderBodyComponent)
                if render_body is not None:
                    render_body.visibility = 0.0

    def _initialize_episode(self, env_idx, options: dict):
        super()._initialize_episode(env_idx, options)
        mount = self.agent.robot.links_map["top_camera_optical_frame"].pose
        camera = Pose.create_from_pq(
            p=list(RECORDING_RIG.top_camera_position),
            q=list(RECORDING_RIG.top_camera_orientation_wxyz),
        )
        self._sensors["top"].camera.set_local_pose((mount.inv() * camera).sp)
        # The wrist camera is measured on the link it is mounted on.
        self._sensors["wrist"].camera.set_local_pose(
            sapien.Pose(
                p=list(RECORDING_RIG.wrist_camera_position),
                q=list(RECORDING_RIG.wrist_camera_orientation_wxyz),
            )
        )

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        # The simulator's overlay step paints one fixed picture into every
        # camera. The table picture was taken by the top camera, so that step
        # is passed over: the top picture gets the table picture as it is,
        # the wrist picture gets the same table seen from where the hand is.
        obs = super(BaseRandomEnv, self)._get_obs_sensor_data(
            apply_texture_transforms
        )
        top, wrist = obs["top"], obs["wrist"]
        top["rgb"] = self._green_screen_rgb(
            top["rgb"], top["segmentation"], self._rgb_overlay_image
        )
        wrist["rgb"] = self._green_screen_rgb(
            wrist["rgb"],
            wrist["segmentation"],
            self._table_in_wrist_picture(wrist["rgb"]),
        )
        return obs

    def _camera_projections(self, name: str) -> np.ndarray:
        return camera_projections(self, name)

    def _table_in_wrist_picture(self, rgb: torch.Tensor) -> torch.Tensor:
        """The recorded table as the wrist camera sees it from where it stands.

        Pixels that look past the table's plane keep what the scene draws.
        """
        table_picture = self._rgb_overlay_image.cpu().numpy()
        seen = rgb.clone()
        for scene_index, (top, wrist) in enumerate(
            zip(self._camera_projections("top"), self._camera_projections("wrist"))
        ):
            painted, on_table = table_seen_from(
                table_picture, top, wrist, self.TABLE_SURFACE_Z
            )
            on_table = torch.as_tensor(on_table, device=rgb.device)
            seen[scene_index][on_table] = torch.as_tensor(
                painted, dtype=rgb.dtype, device=rgb.device
            )[on_table]
        return seen

    def _simulated_qpos(self, state: np.ndarray) -> np.ndarray:
        qpos = super()._simulated_qpos(state)
        qpos[: len(ARM_JOINTS)] += RECORDING_RIG.arm_zero_points_rad
        return qpos

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos().clone()
        qpos[..., : len(ARM_JOINTS)] -= self._arm_zero_points
        return {"noisy_qpos": qpos}

    def step(self, action):
        action = common.to_tensor(action, device=self.device).clone()
        action[..., : len(ARM_JOINTS)] += self._arm_zero_points
        return super().step(action)

    def scene_facts(self) -> dict[str, object]:
        facts = super().scene_facts()
        camera = self._sensors["top"].camera
        extrinsic = camera.get_extrinsic_matrix()[0].cpu().numpy().astype(np.float64)
        facts["top_camera"] = {
            "position": (-extrinsic[:, :3].T @ extrinsic[:, 3]).tolist(),
            "focal_length_px": float(camera.get_intrinsic_matrix()[0, 1, 1]),
            "projection": self._camera_projections("top")[0].tolist(),
        }
        wrist = self._sensors["wrist"].camera
        on_hand = wrist.get_local_pose()
        facts["wrist_camera"] = {
            "position_on_optical_frame": on_hand.p[0].cpu().numpy().tolist(),
            "orientation_on_optical_frame_wxyz": on_hand.q[0].cpu().numpy().tolist(),
            "focal_length_px": float(wrist.get_intrinsic_matrix()[0, 1, 1]),
            "projection": self._camera_projections("wrist")[0].tolist(),
        }
        facts["simulated_arm_deg"] = np.rad2deg(
            self.agent.robot.get_qpos()[0, : len(ARM_JOINTS)].cpu().numpy()
        ).tolist()
        joints = self.agent.robot.active_joints[: len(ARM_JOINTS)]
        facts["arm_drives"] = {
            "has_weight": not any(
                bool(link.disable_gravity[0]) for link in self.agent.robot.links
            ),
            "stiffness": [float(joint.stiffness[0]) for joint in joints],
            "damping": [float(joint.damping[0]) for joint in joints],
        }
        return facts


class StackTask(DefaultCameraEnv):
    """Shared physics and state authority for two- and three-cube stacking."""

    # Height of the table surface in the robot base frame.
    TABLE_SURFACE_Z = 0.0
    SUPPORTED_ROBOTS = ["so101", "so101_kit_slow"]
    SUPPORTED_OBS_MODES = [
        "none",
        "state",
        "state_dict",
        "rgb",
        "rgb+segmentation",
        "rgb+state",
        "rgb+segmentation+state",
    ]
    agent: SO101

    def __init__(
        self,
        *args,
        task_mode: str,
        robot_uids: str = "so101",
        control_mode: str = "pd_joint_target_delta_pos",
        domain_randomization_config=None,
        domain_randomization: bool = False,
        **kwargs,
    ):
        self.task_mode = task_mode
        self.base_z_rot = 0.0
        self.rest_qpos = _real_state_to_sim_qpos(
            np.median(TRAINING_INITIAL_STATES, axis=0)
        ).tolist()
        merged = StackRandomizationConfig().dict()
        if domain_randomization_config is not None:
            incoming = (
                domain_randomization_config
                if isinstance(domain_randomization_config, dict)
                else domain_randomization_config.dict()
            )
            common.dict_merge(merged, incoming)
        self.domain_randomization_config = dacite.from_dict(
            data_class=StackRandomizationConfig,
            data=merged,
            config=dacite.Config(strict=True),
        )
        self.cube_sizes = (
            dict(TRAINING_CUBE_SIZES)
            if task_mode == "stack2-training"
            else {color: COLOR_CUBE_SIZE for color in ("red", "green", "blue")}
        )
        self.active_source = "red" if task_mode == "stack2-training" else None
        self.active_target = "blue" if task_mode == "stack2-training" else None
        self.initial_layout: dict[str, list[float]] = {}
        self.initial_state_real: list[float] = []
        self.initial_state_source_index: int | None = None
        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        cfg = self.domain_randomization_config
        friction = sum(cfg.item_friction_range) / 2
        density = sum(cfg.item_density_range) / 2
        material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0,
        )
        self.cubes: dict[str, Actor] = {}
        self.cube_half_sizes: dict[str, torch.Tensor] = {}
        for color, full_size in self.cube_sizes.items():
            half_size = full_size / 2
            actors = []
            for scene_index in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(
                    half_size=[half_size] * 3,
                    material=material,
                    density=density,
                )
                builder.add_box_visual(
                    half_size=[half_size] * 3,
                    material=sapien.render.RenderMaterial(
                        base_color=CUBE_COLORS[color]
                    ),
                )
                builder.initial_pose = sapien.Pose(
                    p=[0.2 + 0.06 * len(self.cubes), 0, half_size]
                )
                builder.set_scene_idxs([scene_index])
                actor = builder.build(name=f"{color}-cube-{scene_index}")
                actors.append(actor)
                self.remove_from_state_dict_registry(actor)
            cube = Actor.merge(actors, name=f"{color}-cube")
            self.cubes[color] = cube
            self.cube_half_sizes[color] = torch.full(
                (self.num_envs,), half_size, device=self.device
            )
            self.add_to_state_dict_registry(cube)
            if self.apply_greenscreen:
                self.remove_object_from_greenscreen(cube)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429 + self.TABLE_SURFACE_Z],
            q=euler2quat(0, 0, np.pi / 2),
        )
        self._load_camera_mount()
        self._randomize_robot_color()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.table_scene.initialize(env_idx)
        self.table_scene.table.set_pose(self.table_pose)
        self.agent.robot.set_pose(
            Pose.create_from_pq(
                p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)
            )
        )
        layouts: dict[int, dict[str, list[float]]] = {}
        state_indices: dict[int, int | None] = {}
        state_rows: list[np.ndarray] = []
        replay = options.get("stack_replay")
        if replay is not None and (self.num_envs != 1 or env_idx.tolist() != [0]):
            raise ValueError("stack replay requires one complete single-scene reset")
        for scene_index in env_idx.tolist():
            rng = self._batched_episode_rng[scene_index]
            state_index, state = self._start_state(
                int(replay["state_index"])
                if replay is not None
                else int(rng.randint(0, len(TRAINING_INITIAL_STATES)))
            )
            state_indices[scene_index] = state_index
            state_rows.append(state)
            if replay is None:
                layout_seed = int(rng.randint(0, np.iinfo(np.int32).max))
                layout = draw_stack_layout(self.task_mode, layout_seed)
            else:
                expected = set(self.cubes)
                if set(replay["layout"]) != expected:
                    raise ValueError(
                        f"stack replay colors {sorted(replay['layout'])} "
                        f"do not match {sorted(expected)}"
                    )
                layout = {
                    color: np.asarray(replay["layout"][color], dtype=np.float64).tolist()
                    for color in self.cubes
                }
                if any(
                    len(xy) != 2 or not np.isfinite(xy).all()
                    for xy in map(np.asarray, layout.values())
                ):
                    raise ValueError(f"invalid stack replay layout {layout}")
            layouts[scene_index] = layout
            for color, xy in layout.items():
                half_size = self.cube_sizes[color] / 2
                yaw = float(rng.uniform(-np.pi, np.pi))
                position = torch.tensor(
                    [[xy[0], xy[1], self.TABLE_SURFACE_Z + half_size]],
                    dtype=torch.float32,
                    device=self.device,
                )
                quaternion = torch.tensor(
                    np.asarray([euler2quat(0, 0, yaw)]),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.cubes[color].set_pose(
                    Pose.create_from_pq(position, quaternion)
                )
        qpos = np.stack([self._simulated_qpos(state) for state in state_rows])
        self.agent.robot.set_qpos(
            torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        )
        if self.num_envs == 1:
            self.initial_layout = layouts[0]
            self.initial_state_source_index = state_indices[0]
            self.initial_state_real = state_rows[0].tolist()

    def _start_state(self, drawn_index: int) -> tuple[int | None, np.ndarray]:
        """Return the source row and real-units robot state that start an episode."""
        return drawn_index, draw_stack_initial_state(drawn_index)

    def _simulated_qpos(self, state: np.ndarray) -> np.ndarray:
        """Return the simulated joint angles for one real-units robot state."""
        return _real_state_to_sim_qpos(state)

    def _get_obs_agent(self):
        return {"noisy_qpos": self.agent.robot.get_qpos()}

    def _get_obs_extra(self, info: dict):
        return {}

    def set_stack_goal(self, source: str, target: str) -> None:
        if source not in self.cubes or target not in self.cubes:
            raise ValueError(f"unknown cube pair {source!r} on {target!r}")
        if source == target:
            raise ValueError("source and target cube must differ")
        self.active_source = source
        self.active_target = target

    def _interface(self, source: str, target: str) -> dict[str, torch.Tensor]:
        source_cube = self.cubes[source]
        target_cube = self.cubes[target]
        offset = source_cube.pose.p - target_cube.pose.p
        xy_distance = torch.linalg.norm(offset[:, :2], axis=1)
        expected_z = (
            self.cube_half_sizes[source] + self.cube_half_sizes[target]
        )
        z_error = torch.abs(offset[:, 2] - expected_z)
        source_speed = torch.linalg.norm(source_cube.linear_velocity, axis=-1)
        supported = (
            (xy_distance <= STACK_XY_TOLERANCE)
            & (z_error <= STACK_Z_TOLERANCE)
            & (source_speed <= STATIC_SPEED_TOLERANCE)
        )
        return {
            "xy_distance": xy_distance,
            "z_error": z_error,
            "source_speed": source_speed,
            "supported": supported,
        }

    def evaluate(self):
        interfaces = {
            f"{source}_on_{target}": self._interface(source, target)
            for source, target in permutations(self.cubes, 2)
        }
        table_supported = {
            color: (
                torch.abs(
                    cube.pose.p[:, 2]
                    - self.TABLE_SURFACE_Z
                    - self.cube_half_sizes[color]
                )
                <= STACK_Z_TOLERANCE
            )
            & (
                torch.linalg.norm(cube.linear_velocity, axis=-1)
                <= STATIC_SPEED_TOLERANCE
            )
            for color, cube in self.cubes.items()
        }
        if self.active_source is None or self.active_target is None:
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            source_released = success
        else:
            interface = interfaces[
                f"{self.active_source}_on_{self.active_target}"
            ]
            source = self.cubes[self.active_source]
            source_released = ~self.agent.is_grasping(source)
            source_released &= ~self.agent.is_touching(source)
            success = (
                interface["supported"]
                & source_released
                & self.agent.is_static()
            )
        return {
            "success": success,
            "source_released": source_released,
            "interfaces": interfaces,
            "table_supported": table_supported,
        }

    def scene_facts(self) -> dict[str, object]:
        evaluated = self.evaluate()
        return {
            "task_mode": self.task_mode,
            "active_source": self.active_source,
            "active_target": self.active_target,
            "initial_layout": self.initial_layout,
            "initial_state_source_index": self.initial_state_source_index,
            "initial_state_real": self.initial_state_real,
            "cube_sizes": self.cube_sizes,
            "cube_positions": {
                color: cube.pose.p.detach().cpu().numpy()[0].tolist()
                for color, cube in self.cubes.items()
            },
            "interfaces": {
                name: {
                    key: value.detach().cpu().numpy()[0].item()
                    for key, value in facts.items()
                }
                for name, facts in evaluated["interfaces"].items()
            },
            "table_supported": {
                color: bool(value.detach().cpu().numpy()[0])
                for color, value in evaluated["table_supported"].items()
            },
            "grasped": {
                color: bool(
                    self.agent.is_grasping(cube).detach().cpu().numpy()[0]
                )
                for color, cube in self.cubes.items()
            },
            "grip": {
                color: grip_facts(self.agent, cube)
                for color, cube in self.cubes.items()
            },
            "source_released": bool(
                evaluated["source_released"].detach().cpu().numpy()[0]
            ),
            "robot_static": bool(
                self.agent.is_static().detach().cpu().numpy()[0]
            ),
            "success": bool(
                evaluated["success"].detach().cpu().numpy()[0]
            ),
        }

    def compute_dense_reward(self, obs, action, info):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)


@register_env(
    TASK_SPECS["stack2-training"].environment_id, max_episode_steps=600
)
class SO101Stack2Training(TrainingStackCameraMixin, KitSlowMixin, StackTask):
    """Random reachable 20 mm red and 40 mm blue cube stacking task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_mode="stack2-training", **kwargs)


@register_env(
    TASK_SPECS["stack2-sim-view"].environment_id, max_episode_steps=600
)
class SO101Stack2SimView(KitSlowMixin, StackTask):
    """Diagnostic twin of the two-cube task, viewed as in simulated training.

    Cubes, sentence and per-seed cube layout are those of ``stack2-training``.
    Two choices that follow the real stacking recordings are replaced by the
    pinned simulator's own: the top camera stays where its bin tasks put it
    (no ``TrainingStackCameraMixin``), and every episode starts from the
    simulator's rest pose instead of a recorded real start pose.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_mode="stack2-training", **kwargs)

    def _initialize_episode(self, env_idx, options: dict):
        if options.get("stack_replay") is not None:
            raise ValueError(
                "a training-episode replay needs the real start pose; "
                "use stack2-training"
            )
        super()._initialize_episode(env_idx, options)

    def _start_state(self, drawn_index: int) -> tuple[int | None, np.ndarray]:
        # The index was still drawn, so each seed keeps the cube layout it has
        # in stack2-training.
        del drawn_index
        return None, np.array(
            [*REAL_HOME_ARM_DEG, REAL_HOME_GRIPPER_PCT], dtype=np.float64
        )


@register_env(
    TASK_SPECS["stack2-recording-rig"].environment_id, max_episode_steps=600
)
class SO101Stack2RecordingRig(RecordingRigMixin, KitSlowMixin, StackTask):
    """The two-cube task as the rig of the real stacking recordings showed it.

    Cubes, sentence, start poses and per-seed cube layout are those of
    ``stack2-training``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_mode="stack2-training", **kwargs)


@register_env(TASK_SPECS["stack3-color"].environment_id, max_episode_steps=1200)
class SO101Stack3Color(TrainingStackCameraMixin, KitSlowMixin, StackTask):
    """Three equal 40 mm colored cubes in one persistent scene."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_mode="stack3-color", **kwargs)
