"""The three-cube bin scene: the trained 40 mm bin scene, two more cubes, a larger bin."""

from __future__ import annotations

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from so101_sim.envs import _OBJECTS_DIR, SO101PickPlaceCube40, _swap_visual_to_mesh
from so101_sim.tasks.place import PlaceRandomizationConfig
from so101_smolvla.bin_layouts import BIN_CUBE_COLORS, BIN_SCALE, draw_bin_layout
from so101_smolvla.sapien_config import OUTLINE_CODES, TASK_SPECS, THREE_CUBE_TASK
from so101_smolvla.sapien_grip import grip_facts
from so101_smolvla.sapien_stack_env import CUBE_COLORS, camera_projections

# The cube the trained scene already has. The simulator draws it pale blue.
SCENE_CUBE = "blue"
CAMERAS = ("top", "wrist")
# Height of the table surface in the robot base frame: the simulator's bin
# task stands its item on z = 0 (so101_sim/tasks/place.py:373).
TABLE_SURFACE_Z = 0.0
# The room the simulator's bin task adds around its item and again around its
# bin when it places them (so101_sim/tasks/place.py:364-365).
PLACEMENT_MARGIN = 0.010


def trained_bin_half_sizes() -> np.ndarray:
    """Half the length, width and depth of the bin the simulator's bin scenes have."""
    trained = PlaceRandomizationConfig()
    return np.array(
        [
            sum(trained.bin_half_size_x_range) / 2,
            sum(trained.bin_half_size_y_range) / 2,
            sum(trained.bin_half_size_z_range) / 2,
        ]
    )


def _render_bodies(struct) -> list[RenderBodyComponent]:
    bodies = []
    for obj in struct._objs:
        entity = obj.entity if hasattr(obj, "entity") else obj
        body = entity.find_component_by_type(RenderBodyComponent)
        if body is not None:
            bodies.append(body)
    return bodies


def _flag(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().numpy()[0])


@register_env(TASK_SPECS[THREE_CUBE_TASK].environment_id, max_episode_steps=400)
class SO101PickPlaceThreeCubes(SO101PickPlaceCube40):
    """Three 40 mm cubes, red, green and pale blue, and one bin that holds them.

    Robot, cameras, table, cube size, friction and weight, and the rule for
    "in the bin" are those of the trained 40 mm bin scene. The pale blue cube
    is that scene's own. The bin is ``BIN_SCALE`` times the trained one. A
    command names one cube with ``set_target``; ``success`` then speaks of
    that cube. Every observation comes with an outline map per camera.
    """

    def __init__(
        self,
        *args,
        obs_mode: str = "rgb",
        domain_randomization_config=None,
        **kwargs,
    ):
        if obs_mode != "rgb":
            raise ValueError(
                f"the three-cube scene renders pictures only, not {obs_mode!r}"
            )
        if domain_randomization_config is None:
            config = {}
        elif isinstance(domain_randomization_config, dict):
            config = dict(domain_randomization_config)
        else:
            config = domain_randomization_config.dict()
        self.trained_bin_half_sizes = trained_bin_half_sizes()
        half_sizes = self.trained_bin_half_sizes * np.asarray(BIN_SCALE)
        for axis, half_size in zip("xyz", half_sizes, strict=True):
            config[f"bin_half_size_{axis}_range"] = (float(half_size), float(half_size))
        # The outline maps are read from the segmentation. The simulator's
        # overlay step, which with segmentation on would paint its black
        # background picture over the table, stays off.
        config["apply_overlay"] = False
        self.target: str | None = None
        self.initial_layout: dict[str, object] = {}
        super().__init__(
            *args,
            obs_mode="rgb+segmentation",
            domain_randomization_config=config,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        if self.num_envs != 1:
            raise ValueError("the three-cube scene is built as one single scene")
        half_size = float(self.item_half_sizes[0])
        friction = float(self.item_frictions[0])
        density = float(self.item_densities[0])
        full_sizes = self.item_dimensions.cpu().numpy() * 2
        cubes = {SCENE_CUBE: self.item}
        added = [color for color in BIN_CUBE_COLORS if color != SCENE_CUBE]
        for index, color in enumerate(added):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(
                half_size=[half_size] * 3,
                material=sapien.pysapien.physx.PhysxMaterial(
                    static_friction=friction,
                    dynamic_friction=friction,
                    restitution=0,
                ),
                density=density,
            )
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(base_color=CUBE_COLORS[color]),
            )
            # Clear of the scene's cube and bin until the episode places it.
            builder.initial_pose = sapien.Pose(p=[0.2, 0.1 * (index + 1), half_size])
            builder.set_scene_idxs([0])
            actor = builder.build(name=f"{color}-cube-0")
            self.remove_from_state_dict_registry(actor)
            cube = Actor.merge([actor], name=f"{color}-cube")
            self.add_to_state_dict_registry(cube)
            # The same mesh as the scene's cube, in this cube's colour.
            _swap_visual_to_mesh(
                cube,
                _OBJECTS_DIR / f"{self.ITEM_MESH}_visual.glb",
                full_sizes,
                align="center",
            )
            for body in _render_bodies(cube):
                for shape in body.render_shapes:
                    for part in shape.parts:
                        part.material.set_base_color(list(CUBE_COLORS[color]))
            cubes[color] = cube
        self.cubes: dict[str, Actor] = {color: cubes[color] for color in BIN_CUBE_COLORS}

    def _after_reconfigure(self, options: dict):
        super()._after_reconfigure(options)
        shown = {
            "robot": torch.cat([link.per_scene_id for link in self.agent.robot.links]),
            "bin": self.bin.per_scene_id,
            **{color: cube.per_scene_id for color, cube in self.cubes.items()},
        }
        self._outline_ids = {
            OUTLINE_CODES[name]: ids.to(torch.int64) for name, ids in shown.items()
        }
        self._outline_maps: dict[str, np.ndarray] = {}

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if env_idx.tolist() != [0]:
            raise ValueError("the three-cube scene resets as one single scene")
        rng = self._batched_episode_rng[0]
        layout_seed = int(rng.randint(0, np.iinfo(np.int32).max))
        # The top of the bin's walls where the simulator's bin task sets the
        # bin down: walls twice the half depth high on an origin half the
        # floor's thickness up (so101_sim/tasks/place.py:290-291, 380).
        wall_top = 2 * float(self.bin_dimensions[0, 2]) + self.bin_thickness / 2
        cube_half = float(self.item_half_sizes[0])
        layout = draw_bin_layout(
            np.random.default_rng(layout_seed),
            self.bin_dimensions[0, :2].cpu().numpy(),
            # A corner of the bin stands no farther outside the reachable
            # workspace than a corner of the trained bin can.
            outreach=float(np.linalg.norm(self.trained_bin_half_sizes[:2])),
            # The closest a cube's centre comes to the bin's footprint when
            # the simulator's bin task places them: half the cube and the
            # margins around cube and bin.
            keep_clear=cube_half + 2 * PLACEMENT_MARGIN,
            # Bin and cubes stand clear of the resting arm. The gripper rests
            # lower than a cube and than the bin's walls at the near edge of
            # the workspace; a bin or a cube set down there starts pushed
            # into it and is thrown by the first step.
            keep_off=self._table_under_low_arm(
                max(wall_top, 2 * cube_half) + PLACEMENT_MARGIN
            ),
            bin_keep_off=PLACEMENT_MARGIN,
            # A cube's corner reaches half its diagonal from its centre.
            cube_keep_off=cube_half * np.sqrt(2.0) + PLACEMENT_MARGIN,
        )

        def pose(x: float, y: float, z: float, yaw: float) -> Pose:
            return Pose.create_from_pq(
                torch.tensor([[x, y, z]], dtype=torch.float32, device=self.device),
                torch.tensor(
                    np.asarray([euler2quat(0, 0, yaw)]),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )

        self.bin.set_pose(
            pose(*layout["bin_xy"], self.bin_thickness / 2, layout["bin_yaw"])
        )
        cube_height = float(self.item_half_sizes[0])
        for color in BIN_CUBE_COLORS:
            yaw = float(rng.uniform(-np.pi, np.pi))
            self.cubes[color].set_pose(pose(*layout["cubes"][color], cube_height, yaw))
        self.initial_layout = layout
        self.target = None

    def _table_under_low_arm(self, below: float) -> np.ndarray:
        """Table points under the parts of the arm, as it stands, lower than ``below``."""
        shape = self.agent.robot.get_first_collision_mesh(to_world_frame=True)
        if shape is None:
            raise RuntimeError("the arm has no collision shape to keep the bin clear of")
        points = np.asarray(shape.vertices, dtype=np.float64)
        return points[points[:, 2] < below][:, :2]

    def set_target(self, color: str) -> None:
        """Name the cube the next command moves into the bin."""
        if color not in self.cubes:
            raise ValueError(
                f"unknown cube {color!r}; expected one of {list(self.cubes)}"
            )
        self.target = color

    def _judge_cube(self, color: str) -> dict:
        """The simulator's own bin-task judgement, with this cube as its item."""
        scene_item = self.item
        self.item = self.cubes[color]
        try:
            return super().evaluate()
        finally:
            self.item = scene_item

    def evaluate(self):
        if self.target is None:
            judged = self._judge_cube(SCENE_CUBE)
            judged["success"] = torch.zeros_like(judged["success"])
            return judged
        return self._judge_cube(self.target)

    def compute_dense_reward(self, obs, action, info):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        obs = super()._get_obs_sensor_data(apply_texture_transforms)
        self._outline_maps = {
            name: self._outline_map(obs[name]["segmentation"]) for name in CAMERAS
        }
        return obs

    def _outline_map(self, segmentation: torch.Tensor) -> np.ndarray:
        shown = segmentation[0, ..., 0].to(torch.int64)
        codes = torch.full_like(shown, OUTLINE_CODES["other"], dtype=torch.uint8)
        for code, ids in self._outline_ids.items():
            codes[torch.isin(shown, ids.to(shown.device))] = code
        return codes.cpu().numpy()

    def outline_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """What every pixel of the last top and wrist picture shows."""
        return self._outline_maps["top"], self._outline_maps["wrist"]

    def pictures_without_cubes_and_robot(self) -> tuple[np.ndarray, np.ndarray]:
        """The top and wrist picture of the scene as it stands, cubes and robot not drawn.

        Table and bin are drawn. The outline maps of the last observation stay.
        """
        bodies = [
            body
            for struct in (*self.cubes.values(), *self.agent.robot.links)
            for body in _render_bodies(struct)
        ]
        shown = [body.visibility for body in bodies]
        try:
            for body in bodies:
                body.visibility = 0.0
            obs = super()._get_obs_sensor_data()
        finally:
            for body, visibility in zip(bodies, shown, strict=True):
                body.visibility = visibility
        top, wrist = (
            obs[name]["rgb"][0].cpu().numpy().astype(np.uint8) for name in CAMERAS
        )
        return top, wrist

    def scene_facts(self) -> dict[str, object]:
        judged = {color: self._judge_cube(color) for color in self.cubes}
        any_cube = judged[SCENE_CUBE]
        bin_quaternion = self.bin.pose.q[0].detach().cpu().numpy()
        return {
            "task_mode": THREE_CUBE_TASK,
            # Height of the table surface in the robot base frame.
            "table_height": TABLE_SURFACE_Z,
            "target": self.target,
            "initial_layout": self.initial_layout,
            "bin": {
                "position": self.bin.pose.p[0].detach().cpu().numpy().tolist(),
                "yaw": float(2.0 * np.arctan2(bin_quaternion[3], bin_quaternion[0])),
                "half_sizes": self.bin_dimensions[0].cpu().numpy().tolist(),
            },
            "cube_half_size": float(self.item_half_sizes[0]),
            "cubes": {
                color: {
                    "position": cube.pose.p[0].detach().cpu().numpy().tolist(),
                    # Inside the bin's footprint, by the simulator's rule.
                    "in_bin": _flag(judged[color]["is_item_above_bin"]),
                    # Its centre stands higher than on the table.
                    "above_table": _flag(judged[color]["item_lifted"]),
                    "grasped": _flag(judged[color]["is_item_grasped"]),
                    "touched_by_robot": _flag(judged[color]["robot_touching_item"]),
                    "speed": float(judged[color]["item_vel"][0]),
                }
                for color, cube in self.cubes.items()
            },
            "grip": {
                color: grip_facts(self.agent, cube)
                for color, cube in self.cubes.items()
            },
            "robot_static": _flag(any_cube["is_robot_static"]),
            "robot_touching_bin": _flag(any_cube["robot_touching_bin"]),
            "cameras": {
                name: camera_projections(self, name)[0].tolist() for name in CAMERAS
            },
            "success": _flag(self.evaluate()["success"]),
        }
