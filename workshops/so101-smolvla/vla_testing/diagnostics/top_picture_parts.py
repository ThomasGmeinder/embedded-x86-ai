"""Which part of the scene's top picture moves the model's aim away from the red cube?

Run from the workshop folder, in this order:
    top_picture_parts.py render    (.venv-sapien) assemble the mixed top pictures
    top_picture_parts.py predict   (.venv) the model's answers
    grasp_aim_check.py report top-picture-parts-predictions.npz top-picture-parts.json
                                   (.venv-sapien) where those answers put the gripper
PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim : ~/.cache/so101-stack-training
: <workshop>/vla_testing/diagnostics.

Moment: 25 frames before the human closes the gripper on the red cube, the
moment of grasp_aim_check.py, in the same recordings. A top picture is taken
apart into three parts: the table (with the shadows on it), the two cubes,
and the robot. Each part comes either from the real picture or from the
stack2-recording-rig scene drawn at the same joint readings, which gives
eight top pictures per recording. The wrist picture is the real one in all
of them, so only the top picture is under test.

Labels name the source of each part in the order table, cubes, robot:
"real table, scene cubes, real robot". The picture with three real parts is
the real picture itself and the one with three scene parts is the scene's
picture itself, pixel for pixel; both are checked.

Five more pictures change one thing in the real picture:
  the scene's cubes in the colours measured in the recordings
  the scene's robot with every part in the colour of its hand
  the scene's hand on the real arm
  the scene's arm with the real hand
  the real robot's pink cables painted over with table where they lie on the table

Three more separate a robot's outline from what is seen inside it:
  the real robot's outline filled with the scene's robot (beyond the scene's
    robot, with its nearest pixel)
  the real robot only where the scene's robot is too (the rest is table)
  the scene's robot, with the real robot where it reaches beyond the scene's

One more changes the posture of the scene's robot: its arm joints are turned
as the real robot's outline in this very picture asks for
(arm_pose_from_outline.py, analysis/arm-pose-from-outline.npz).

Writes analysis/top-picture-parts-pictures.npz, analysis/top-picture-parts-<n>.png
(render) and analysis/top-picture-parts-predictions.npz (predict).

The same pictures for a moment of the reach, <before> frames before the
closing (after early_reach_check.py has stored the real pictures of that
moment in analysis/early-reach-<before>-predictions.npz):
    top_picture_parts.py render <before>          (.venv-sapien)
    release_aim_check.py predict early-reach-<before>-top-parts-pictures.npz early-reach-<before>-top-parts-predictions.npz   (.venv)
    early_reach_check.py report <before> early-reach-<before>-top-parts   (.venv-sapien)
The posture picture is left out there (the outline fit exists for the grasp
moment only). Writes analysis/early-reach-<before>-top-parts-pictures.npz and
analysis/early-reach-<before>-top-parts-<n>.png.

For the reach, three more pictures ask how dark the robot is shown. The
recording robot is black all over and the recorded top pictures show it much
darker than the scene's top camera shows the scene's black parts
(wrist_picture_parts.py, analysis/early-reach-<before>-wrist-parts-black-levels.json):
  the scene's robot with every part in the grey that camera shows as dark as
    the recorded robot
  the real robot's outline filled with that dark scene robot
  the real robot's outline filled with one colour, the real robot's own black
A file stem as third argument keeps the earlier files of the moment:
    top_picture_parts.py render <before> <stem>     writes analysis/<stem>-pictures.npz, <stem>-<n>.png

For the reach, four more pictures ask about the pink servo cable that runs
along the recording robot's arm and that no drawn robot has. (The picture
"cables on the table painted over" paints every cable pixel with table, so
where the cable crosses the arm it leaves a light streak instead of a pink
one; it does not take the cable off the arm.)
  the real picture with the cable painted in the real robot's black
  the real robot's outline filled with that one black, the real cable kept
  the scene's robot as dark as the recorded robot, with the real cable
  the scene's robot, with the real cable
Words after the stem keep only the real picture and the pictures whose label
contains one of them:
    top_picture_parts.py render <before> <stem> cable

For the reach, more pictures ask which part of the robot it is. The robot is
five parts: base, shoulder, upper arm, forearm with the wrist joint, hand. A
pixel of the drawn robot belongs to the part the scene's segmentation
picture names there, a pixel of the real robot to the drawn part nearest to
it. Asked about: each part alone, and base and shoulder together.
  the real picture with that part drawn by the scene
  the scene's robot with that part taken from the real picture
All parts from the scene must be the picture "real table, real cubes, scene
robot" and none of them the real picture, pixel for pixel; both are checked.
    top_picture_parts.py render <before> <stem> taken
For the base, the shoulder and both together, which stand where the table
picture holds the recording robot's own base, three more each:
  the real picture with that part replaced by what the table picture shows
    there, nothing drawn
  the real picture with the scene's part drawn over it, nothing real taken away
  the scene's robot with that part not drawn, the table picture showing there
    top_picture_parts.py render <before> <stem> "table picture" "drawn over"
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
PICTURES = ANALYSIS / "top-picture-parts-pictures.npz"
SOURCES = ("real", "scene")
PARTS = ("table", "cubes", "robot")
RECORDED_CUBES = "real table, scene cubes in the recorded colours, real robot"
ONE_COLOUR_ROBOT = "real table, real cubes, scene robot in the colour of its hand"
SCENE_HAND = "real table, real cubes, scene hand on the real arm"
SCENE_ARM = "real table, real cubes, scene arm with the real hand"
NO_CABLES = "real picture, cables on the table painted over"
REAL_OUTLINE = "real robot's outline filled with the scene's robot"
REAL_CUT = "real robot only where the scene's robot is too"
SCENE_WIDENED = "scene robot, with the real robot where it reaches beyond"
POSED_BY_OUTLINE = "scene robot posed as the real robot's outline asks"
DARK_ROBOT = "real table, real cubes, scene robot with every part as dark as the recorded robot"
DARK_REAL_OUTLINE = "real robot's outline filled with the scene's robot as dark as the recorded robot"
FLAT_REAL_OUTLINE = "real robot's outline filled with the one black of the real robot"
CABLE_BLACK = "real picture, cable painted in the real robot's black"
FLAT_WITH_CABLE = "real robot's outline filled with its one black, real cable kept"
DARK_WITH_CABLE = "scene robot as dark as the recorded robot, with the real cable"
SCENE_WITH_CABLE = "real table, real cubes, scene robot, with the real cable"
# The drawn robot in five parts; the hand's links are sapien_config.HAND_LINKS.
HAND = "hand"
ARM_PARTS = {
    "base": ("base_link",),
    "shoulder": ("shoulder_link",),
    "upper arm": ("upper_arm_link",),
    "forearm and wrist joint": ("lower_arm_link", "wrist_link"),
}
ROBOT_PARTS = (*ARM_PARTS, HAND)
# What is asked about: base and shoulder together, and every part alone.
PART_GROUPS = {
    "base and shoulder": ("base", "shoulder"),
    "upper arm": ("upper arm",),
    "forearm and wrist joint": ("forearm and wrist joint",),
    HAND: (HAND,),
    "base": ("base",),
    "shoulder": ("shoulder",),
}
# The groups that stand where the table picture holds the recording robot's base.
STANDING = ("base and shoulder", "base", "shoulder")
PART_FROM_SCENE = "real robot, {} taken from the scene"
PART_FROM_REAL = "scene robot, {} taken from the real picture"
PART_AS_TABLE = "real robot, {} as the table picture shows it"
PART_DRAWN_OVER = "real robot, the scene's {} drawn over it"
SCENE_WITHOUT = "scene robot, {} not drawn, the table picture showing there"
# The stored outline fit belongs to a picture if its reading is the picture's, to this many degrees.
SAME_READING_DEG = 1e-6
# The real robot is black; nothing else in a top picture is this dark.
DARKEST_TABLE_VALUE = 70
# A cube's soft edge and the shadow at its foot go with the cube.
CUBE_MARGIN_PX = 6
ROBOT_MARGIN_PX = 2
# The real robot's thin pink cables lie this close to its dark parts.
CABLE_REACH_PX = 30
SHEET_RECORDINGS = (3, 20)


def label(sources: tuple[str, str, str]) -> str:
    return ", ".join(f"{source} {part}" for source, part in zip(sources, PARTS))


def render(before: str | None = None, stem: str | None = None, *only: str) -> None:
    import cv2
    import torch
    from mani_skill.utils.structs.pose import Pose
    from sapien.render import RenderBodyComponent
    from scipy.ndimage import distance_transform_edt

    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from grasp_aim_check import recordings
    from measure_objects import colour_signal, measure
    from so101_sim.envs import ROBOT_COLOR
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import HAND_LINKS, environment_arguments
    from so101_smolvla.sapien_env_worker import _configure_robot_appearance

    def grow(mask: np.ndarray, pixels: int) -> np.ndarray:
        size = 2 * pixels + 1
        return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)

    def real_cube(image: np.ndarray, colour: str) -> np.ndarray | None:
        """Pixels of one cube in a real picture in which the cube lies free."""
        found = measure(image, colour)
        if found is None:
            return None
        signal = colour_signal(image, colour)
        (cx, cy), reach = found["centre_px"], found["side_px"]
        ys, xs = np.mgrid[0:480, 0:640]
        box = (np.abs(xs - cx) < reach) & (np.abs(ys - cy) < reach)
        around = float(np.median(signal))
        peak = float(np.percentile(signal[box], 95))
        return box & (signal > around + 0.4 * (peak - around))

    def real_robot(image: np.ndarray, cubes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels of the real robot, and those of them that are pink cable.

        The cable pixels are the pink ones near the robot that are not dark:
        the cable where it lies on the table and where it crosses the arm.
        """
        dark = image.max(axis=2) < DARKEST_TABLE_VALUE
        redness = colour_signal(image, "red")
        cable = (
            (redness > float(np.median(redness)) + 10.0)
            & grow(dark, CABLE_REACH_PX)
            & ~grow(cubes, CUBE_MARGIN_PX)
        )
        return grow(dark | cable, ROBOT_MARGIN_PX), grow(cable, ROBOT_MARGIN_PX) & ~dark

    stored = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    recorded_paint = json.loads(
        (ANALYSIS / "objects" / "fitted-paint.json").read_text())["paint_base_colour"]
    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    mixes = [label(sources) for sources in product(SOURCES, repeat=3)]
    names = [*mixes, RECORDED_CUBES, ONE_COLOUR_ROBOT, SCENE_HAND, SCENE_ARM, NO_CABLES,
             REAL_OUTLINE, REAL_CUT, SCENE_WIDENED]
    if before is None:
        # The grasp moment: its real pictures and its outline fit are stored.
        names.append(POSED_BY_OUTLINE)
        outline_fit = np.load(ANALYSIS / "arm-pose-from-outline.npz")
        frames_before = 25
    else:
        frames_before = int(before)
        reach = np.load(ANALYSIS / f"early-reach-{frames_before}-predictions.npz")
        reach_episode, reach_frame = reach["episode"].tolist(), reach["frame"].tolist()
        reach_top = reach["real_top"]
        names += [DARK_ROBOT, DARK_REAL_OUTLINE, FLAT_REAL_OUTLINE,
                  CABLE_BLACK, FLAT_WITH_CABLE, DARK_WITH_CABLE, SCENE_WITH_CABLE]
        names += [form.format(group) for form in (PART_FROM_SCENE, PART_FROM_REAL)
                  for group in PART_GROUPS]
        names += [form.format(group) for form in (PART_AS_TABLE, PART_DRAWN_OVER, SCENE_WITHOUT)
                  for group in STANDING]
        # Per picture: pixels of each part, drawn and real; drawn robot pixels
        # at which the segmentation picture names no robot link (soft edges).
        part_pixels: list[list[list[int]]] = []
        edge_pixels: list[int] = []
        # The grey of those tried that the scene's top camera shows as dark
        # as the recorded top pictures show the recording robot.
        black = json.loads(
            (ANALYSIS / f"early-reach-{frames_before}-wrist-parts-black-levels.json").read_text()
        )["brightness_of_black_parts_0_to_255_median"]["top"]
        dark_grey = float(min(
            black["scene_per_grey_tried"],
            key=lambda tried: abs(black["scene_per_grey_tried"][tried] - black["recorded_pictures"])))
        dark_robot_shown: list[float] = []
        cable_pixels: list[int] = []
    # Every picture is made (some are built from others); `names` are the ones kept.
    made = list(names)
    if only:
        if before is None or stem is None:
            raise SystemExit("words that choose pictures need a moment of the reach and a file stem")
        names = [name for name in made
                 if name == label(("real",) * 3) or any(word in name for word in only)]
    rows: dict[str, list] = {
        "episode": [], "frame": [], "state": [], "red_centre_px": [],
        "red_cube": [], "table_height": [],
    }
    rows.update({name: [] for name in made})
    skipped: list[int] = []
    shown_cube_rgb: dict[str, list] = {"red": [], "blue": []}
    try:
        _configure_robot_appearance(env)
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)

        def render_bodies(holder) -> list:
            found = []
            for obj in holder._objs:
                entity = obj.entity if hasattr(obj, "entity") else obj
                body = entity.find_component_by_type(RenderBodyComponent)
                # Parts the scene already hides stay hidden.
                if body is not None and body.visibility > 0.0:
                    found.append(body)
            return found

        def materials(bodies: list) -> list:
            return [part.material for body in bodies
                    for shape in body.render_shapes for part in shape.parts]

        links = inner.agent.robot.links
        robot_bodies = [b for link in links for b in render_bodies(link)]
        hand_bodies = [b for link in links if link.name in HAND_LINKS for b in render_bodies(link)]
        cube_bodies = {colour: render_bodies(cube) for colour, cube in inner.cubes.items()}
        all_cube_bodies = [b for bodies in cube_bodies.values() for b in bodies]
        if not hand_bodies:
            raise RuntimeError(f"no drawn part of the hand links {sorted(HAND_LINKS)}")
        part_links = {**ARM_PARTS, HAND: tuple(sorted(HAND_LINKS))}
        drawn_links = sorted(link.name for link in links if render_bodies(link))
        if sorted(name for chosen in part_links.values() for name in chosen) != drawn_links:
            raise RuntimeError(f"the parts {part_links} are not the drawn links {drawn_links}")
        part_ids = {part: [int(link.per_scene_id[0]) for link in links if link.name in chosen]
                    for part, chosen in part_links.items()}

        def drawn_ids() -> np.ndarray:
            """The body the top camera shows at each pixel, from the scene's segmentation picture."""
            return inner.get_obs()["sensor_data"]["top"]["segmentation"][0, ..., 0].cpu().numpy()

        def show(group: list, visible: bool) -> None:
            for body in group:
                body.visibility = 1.0 if visible else 0.0

        def pose_robot(reading: np.ndarray) -> np.ndarray:
            qpos = inner._simulated_qpos(reading)
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0)

        def stand(color: str, xy: np.ndarray) -> None:
            height = inner.TABLE_SURFACE_Z + inner.cube_sizes[color] / 2
            inner.cubes[color].set_pose(Pose.create_from_pq(p=torch.tensor(
                [[float(xy[0]), float(xy[1]), height]],
                dtype=torch.float32, device=inner.device)))

        def top_picture() -> np.ndarray:
            return env._format_raw_obs(inner.get_obs())["pixels"]["top"].copy()

        def repainted(parts: list, colours: list, matte: bool) -> np.ndarray:
            """The top picture with these materials in these colours; the scene is put back."""
            before = [(m.base_color, m.roughness, m.metallic, m.specular) for m in parts]
            for material, colour in zip(parts, colours):
                material.set_base_color([*colour, 1.0])
                if matte:
                    material.roughness, material.metallic, material.specular = 1.0, 0.0, 0.0
            picture = top_picture()
            for material, (colour, roughness, metallic, specular) in zip(parts, before):
                material.set_base_color(colour)
                material.roughness, material.metallic, material.specular = (
                    roughness, metallic, specular)
            return picture

        robot_materials = materials(robot_bodies)
        cube_materials = [(colour, m) for colour, bodies in cube_bodies.items()
                          for m in materials(bodies)]

        for number, close, release in recordings():
            mine = stored["episode"] == number
            first = mine & (stored["moment"] == 0)
            start = stored["top"][first][0]
            if before is None:
                real = stored["top"][mine & (stored["moment"] == 1)][0]
                frame = int(stored["frame"][mine & (stored["moment"] == 1)][0])
            else:
                at = reach_episode.index(number)
                real, frame = reach_top[at], int(reach_frame[at])
            if frame != close - frames_before:
                raise RuntimeError(
                    f"recording {number}: stored frame {frame} is not {close - frames_before}")
            free_cubes = [real_cube(start, colour) for colour in ("red", "blue")]
            if any(mask is None for mask in free_cubes):
                skipped.append(number)
                continue
            real_cubes = grow(free_cubes[0] | free_cubes[1], CUBE_MARGIN_PX)
            real_robot_mask, cable_on_table = real_robot(real, free_cubes[0] | free_cubes[1])

            state = data["state"][data["episode"] == number].astype(np.float64)
            stand("red", pose_robot(state[min(close + 10, release - 1)]))
            stand("blue", pose_robot(state[release]))
            pose_robot(state[frame])
            show(robot_bodies, True), show(all_cube_bodies, True)
            scene = top_picture()
            one_colour_scene = repainted(
                robot_materials, [ROBOT_COLOR] * len(robot_materials), matte=False)
            if before is not None:
                dark_scene = repainted(
                    robot_materials, [(dark_grey,) * 3] * len(robot_materials), matte=False)
            show(hand_bodies, False)
            scene_without_hand = top_picture()
            show(robot_bodies, False)
            scene_cubes_on_table = top_picture()
            recorded_cubes_on_table = repainted(
                [m for _, m in cube_materials],
                [recorded_paint[colour] for colour, _ in cube_materials], matte=True)
            for colour in shown_cube_rgb:
                show(cube_bodies[colour], False)
                this_cube = np.any(scene_cubes_on_table != top_picture(), axis=2)
                show(cube_bodies[colour], True)
                if this_cube.any():
                    shown_cube_rgb[colour].append(
                        np.median(recorded_cubes_on_table[this_cube], axis=0))
            show(all_cube_bodies, False)
            scene_table = top_picture()
            show(robot_bodies, True), show(all_cube_bodies, True)
            scene_robot_mask = np.any(scene != scene_cubes_on_table, axis=2)
            scene_cube_mask = np.any(scene_cubes_on_table != scene_table, axis=2)
            scene_hand_mask = np.any(scene != scene_without_hand, axis=2)
            scene_arm_mask = scene_robot_mask & ~scene_hand_mask

            # The real cubes as they lie free: where the arm covers them now,
            # the first frame of the recording shows them.
            free_real_cubes = np.where(real_robot_mask[..., None], start, real)

            def without_real_robot(gone: np.ndarray) -> np.ndarray:
                """The real picture with these robot pixels replaced by what lies under them."""
                picture = real.copy()
                picture[gone] = scene_table[gone]
                uncovered = real_cubes & gone
                picture[uncovered] = start[uncovered]
                return picture

            for sources in product(SOURCES, repeat=3):
                table, cubes, robot = sources
                if table == "real":
                    picture = real.copy()
                    gone = np.zeros((480, 640), dtype=bool)
                    if robot == "scene":
                        gone |= real_robot_mask
                    if cubes == "scene":
                        gone |= grow(real_cubes, 2)
                    picture[gone] = scene_table[gone]
                    if robot == "scene" and cubes == "real":
                        uncovered = real_cubes & real_robot_mask
                        picture[uncovered] = start[uncovered]
                else:
                    picture = scene_table.copy()
                    if cubes == "real":
                        picture[real_cubes] = free_real_cubes[real_cubes]
                if cubes == "scene":
                    picture[scene_cube_mask] = scene_cubes_on_table[scene_cube_mask]
                if robot == "scene":
                    picture[scene_robot_mask] = scene[scene_robot_mask]
                elif table == "scene" or cubes == "scene":
                    picture[real_robot_mask] = real[real_robot_mask]
                rows[label(sources)].append(picture)
            if not np.array_equal(rows[label(("real",) * 3)][-1], real):
                raise RuntimeError(f"recording {number}: three real parts are not the real picture")
            if not np.array_equal(rows[label(("scene",) * 3)][-1], scene):
                raise RuntimeError(f"recording {number}: three scene parts are not the scene picture")

            picture = real.copy()
            gone = grow(real_cubes, 2)
            picture[gone] = scene_table[gone]
            picture[scene_cube_mask] = recorded_cubes_on_table[scene_cube_mask]
            picture[real_robot_mask] = real[real_robot_mask]
            rows[RECORDED_CUBES].append(picture)

            picture = without_real_robot(real_robot_mask)
            picture[scene_robot_mask] = one_colour_scene[scene_robot_mask]
            rows[ONE_COLOUR_ROBOT].append(picture)

            # The real hand with its cables: the real robot near the drawn
            # hand, apart from where the drawn arm is.
            real_hand = real_robot_mask & grow(scene_hand_mask, CABLE_REACH_PX) & ~scene_arm_mask
            picture = without_real_robot(real_hand)
            picture[scene_hand_mask] = scene[scene_hand_mask]
            rows[SCENE_HAND].append(picture)

            picture = without_real_robot(real_robot_mask & ~real_hand)
            picture[scene_arm_mask] = scene[scene_arm_mask]
            picture[real_hand] = real[real_hand]
            rows[SCENE_ARM].append(picture)

            rows[NO_CABLES].append(without_real_robot(cable_on_table))

            # Outline against inside. Where the real robot reaches beyond the
            # scene's, the scene robot's nearest pixel stands in.
            beyond_scene = real_robot_mask & ~scene_robot_mask
            _, (nearest_y, nearest_x) = distance_transform_edt(
                ~scene_robot_mask, return_indices=True)
            picture = real.copy()
            picture[real_robot_mask] = scene[nearest_y, nearest_x][real_robot_mask]
            rows[REAL_OUTLINE].append(picture)

            rows[REAL_CUT].append(without_real_robot(beyond_scene))

            picture = rows[label(("real", "real", "scene"))][-1].copy()
            picture[beyond_scene] = real[beyond_scene]
            rows[SCENE_WIDENED].append(picture)

            if before is not None:
                # How dark the robot is shown.
                picture = without_real_robot(real_robot_mask)
                picture[scene_robot_mask] = dark_scene[scene_robot_mask]
                rows[DARK_ROBOT].append(picture)
                dark_robot_shown.append(
                    float(np.median(dark_scene[scene_robot_mask].mean(axis=1))))

                picture = real.copy()
                picture[real_robot_mask] = dark_scene[nearest_y, nearest_x][real_robot_mask]
                rows[DARK_REAL_OUTLINE].append(picture)

                real_black = np.median(real[real.max(axis=2) < DARKEST_TABLE_VALUE], axis=0)
                picture = real.copy()
                picture[real_robot_mask] = real_black
                rows[FLAT_REAL_OUTLINE].append(picture)

                # The cable along the arm, taken off the real robot and put on the others.
                picture = real.copy()
                picture[cable_on_table] = real_black
                rows[CABLE_BLACK].append(picture)

                picture = rows[FLAT_REAL_OUTLINE][-1].copy()
                picture[cable_on_table] = real[cable_on_table]
                rows[FLAT_WITH_CABLE].append(picture)

                picture = rows[DARK_ROBOT][-1].copy()
                picture[cable_on_table] = real[cable_on_table]
                rows[DARK_WITH_CABLE].append(picture)

                picture = rows[label(("real", "real", "scene"))][-1].copy()
                picture[cable_on_table] = real[cable_on_table]
                rows[SCENE_WITH_CABLE].append(picture)
                cable_pixels.append(int(cable_on_table.sum()))

                # Which part of the robot. The scene is fully shown and posed here.
                ids = drawn_ids()
                named = np.stack([np.isin(ids, part_ids[part]) for part in ROBOT_PARTS])
                nearest = np.stack([
                    distance_transform_edt(~shown) if shown.any()
                    else np.full(shown.shape, np.inf) for shown in named]).argmin(axis=0)
                scene_part = {part: scene_robot_mask & (nearest == index)
                              for index, part in enumerate(ROBOT_PARTS)}
                real_part = {part: real_robot_mask & (nearest == index)
                             for index, part in enumerate(ROBOT_PARTS)}

                def mixed(taken_away: tuple[str, ...], drawn: tuple[str, ...]) -> np.ndarray:
                    """The real picture without these parts of the real robot, with these drawn parts on it."""
                    gone = np.zeros_like(real_robot_mask)
                    for part in taken_away:
                        gone |= real_part[part]
                    picture = without_real_robot(gone)
                    for part in drawn:
                        picture[scene_part[part]] = scene[scene_part[part]]
                    return picture

                if not np.array_equal(mixed(ROBOT_PARTS, ROBOT_PARTS),
                                      rows[label(("real", "real", "scene"))][-1]):
                    raise RuntimeError(
                        f"recording {number}: all scene parts are not the scene's robot in the real picture")
                if not np.array_equal(mixed((), ()), real):
                    raise RuntimeError(f"recording {number}: no scene part is not the real picture")
                for group, parts in PART_GROUPS.items():
                    others = tuple(part for part in ROBOT_PARTS if part not in parts)
                    rows[PART_FROM_SCENE.format(group)].append(mixed(parts, parts))
                    rows[PART_FROM_REAL.format(group)].append(mixed(others, others))
                    if group in STANDING:
                        rows[PART_AS_TABLE.format(group)].append(mixed(parts, ()))
                        rows[PART_DRAWN_OVER.format(group)].append(mixed((), parts))
                        rows[SCENE_WITHOUT.format(group)].append(mixed(ROBOT_PARTS, others))
                part_pixels.append([[int(scene_part[part].sum()), int(real_part[part].sum())]
                                    for part in ROBOT_PARTS])
                edge_pixels.append(int((scene_robot_mask & ~named.any(axis=0)).sum()))

            if before is None:
                # Posture. Moment 1 of the outline fit is this picture.
                fitted = (outline_fit["episode"] == number) & (outline_fit["moment"] == 1)
                arm = outline_fit["reading_deg"].shape[1]
                if not np.allclose(outline_fit["reading_deg"][fitted][0], state[frame][:arm],
                                   atol=SAME_READING_DEG):
                    raise RuntimeError(f"recording {number}: the outline fit is of another reading")
                turned = state[frame].copy()
                turned[:arm] += outline_fit["turned_by_deg"][fitted][0]
                pose_robot(turned)
                posed = top_picture()
                posed_robot_mask = np.any(posed != scene_cubes_on_table, axis=2)
                picture = without_real_robot(real_robot_mask)
                picture[posed_robot_mask] = posed[posed_robot_mask]
                rows[POSED_BY_OUTLINE].append(picture)
                pose_robot(state[frame])

            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(state[frame])
            rows["red_centre_px"].append(stored["red_centre_px"][first][0])
            rows["red_cube"].append(inner.cubes["red"].pose.p[0].cpu().numpy().copy())
            rows["table_height"].append(float(inner.TABLE_SURFACE_Z))
    finally:
        env.close()
    if before is None:
        sheets = "top-picture-parts"
        np.savez_compressed(PICTURES, **{name: np.array(values) for name, values in rows.items()})
    else:
        # The form early_reach_check.py report and release_aim_check.py predict
        # read: every "top: <what>" is asked together with the real wrist picture.
        sheets = stem or f"early-reach-{frames_before}-top-parts"
        np.savez_compressed(
            ANALYSIS / f"{sheets}-pictures.npz",
            **{(f"top: {name}" if name in names else name): np.array(values)
               for name, values in rows.items() if name in names or name not in made},
        )
        print(f"pink cable pixels per picture, smallest / median / largest: "
              f"{min(cable_pixels)} / {int(np.median(cable_pixels))} / {max(cable_pixels)}",
              flush=True)
        for part, (drawn, real_count) in zip(ROBOT_PARTS, np.median(part_pixels, axis=0).astype(int)):
            print(f"{part}: {drawn} drawn pixels, {real_count} real pixels (median per picture)",
                  flush=True)
        print(f"drawn robot pixels at which the segmentation names no robot link, "
              f"median / largest: {int(np.median(edge_pixels))} / {max(edge_pixels)}", flush=True)
        print(f"the scene's robot with every part in grey {dark_grey} shows at brightness "
              f"{np.median(dark_robot_shown):.1f} of 255 in the top picture; the recorded "
              f"robot: {black['recorded_pictures']}", flush=True)
    print(f"assembled {len(rows['episode'])} recordings; a cube was not found free in the "
          f"first frame of recordings {skipped}", flush=True)
    measured = json.loads(
        (ANALYSIS / "objects" / "fitted-paint.json").read_text())["target_picture_rgb"]
    for colour, values in shown_cube_rgb.items():
        print(f"{colour} cube in the recorded colours as the scene's top camera shows it: "
              f"{np.median(values, axis=0).round().tolist()}; measured in the recordings: "
              f"{measured[colour]}", flush=True)

    # All pictures of two recordings, cut out around the red cube, to look at.
    for sheet, index in enumerate(SHEET_RECORDINGS):
        cx, cy = rows["red_centre_px"][index]
        x0 = int(np.clip(cx - 130, 0, 640 - 240))
        y0 = int(np.clip(cy - 110, 0, 480 - 180))
        tiles = []
        for name in names:
            tile = np.ascontiguousarray(rows[name][index][y0:y0 + 180, x0:x0 + 240])
            tile = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            for line, text in enumerate(name.split(", ")):
                cv2.putText(tile, text, (4, 16 + 16 * line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            tiles.append(tile)
        tiles.extend([np.full_like(tiles[0], 255)] * (-len(tiles) % 4))
        rows_of_tiles = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(ANALYSIS / f"{sheets}-{sheet}.png"),
                    cv2.cvtColor(np.concatenate(rows_of_tiles, axis=0), cv2.COLOR_RGB2BGR))


def predict() -> None:
    import glob
    import time

    import pandas as pd

    from compare_real_vs_sim_images import predict as answer
    from measure_objects import FPS, real_first_frame
    from so101_smolvla.policy_env import PolicyObservation
    from so101_smolvla.sapien_config import task_spec
    from so101_smolvla.sapien_rollout import _load_policy

    stored = np.load(PICTURES)
    names = [name for name in stored.files if stored[name].ndim == 4]
    pictures = {name: stored[name] for name in names}
    episodes = pd.concat(
        pd.read_parquet(path)
        for path in sorted(glob.glob(str(ROOT / "meta" / "episodes" / "chunk-000" / "*.parquet")))
    ).set_index("episode_index")
    column = "videos/observation.images.wrist/from_timestamp"
    real_wrist = [
        real_first_frame("wrist", float(episodes.loc[number][column]) + frame / FPS)
        for number, frame in zip(stored["episode"].tolist(), stored["frame"].tolist())
    ]
    text = task_spec("stack2-recording-rig").instruction
    print(f"task sentence: {text!r}", flush=True)
    checkpoint, policy, pre, post = _load_policy()
    print(f"checkpoint revision {checkpoint.name}", flush=True)

    started = time.monotonic()
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for index, state in enumerate(stored["state"]):
        for name in names:
            observation = PolicyObservation(
                state=state.astype(np.float32), top=pictures[name][index], wrist=real_wrist[index],
            )
            chunks[name].append(answer(policy, pre, post, observation, text, 0))
        print(f"recording {int(stored['episode'][index])} answered", flush=True)
    count = len(stored["state"]) * len(names)
    print(f"{count} answers, {(time.monotonic() - started) / count:.2f} s per answer", flush=True)
    np.savez_compressed(
        ANALYSIS / "top-picture-parts-predictions.npz",
        episode=stored["episode"], frame=stored["frame"],
        **{name: np.stack(values) for name, values in chunks.items()},
    )


if __name__ == "__main__":
    {"render": render, "predict": predict}[sys.argv[1]](*sys.argv[2:])
