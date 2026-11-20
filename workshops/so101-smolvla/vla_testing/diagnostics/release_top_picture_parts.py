"""Which part of the scene's top picture makes the model let the red cube go beside the large cube?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after release_aim_check.py render and predict:
    release_top_picture_parts.py
then, with file names in analysis/:
    release_aim_check.py predict release-top-parts-pictures.npz release-top-parts-predictions.npz
    release_aim_check.py report  release-top-parts-pictures.npz release-top-parts-predictions.npz release-top-parts.json

Moment and recordings are those of release_aim_check.py: 25 frames before the
human lets the red cube go. The top picture is taken apart into three parts:
the table (with the shadows on it), the two cubes (the large one on the table
and the red one in the hand), and the robot. Each part comes either from the
real picture or from the stack2-recording-rig scene at the same joint
readings, which gives eight top pictures per recording, named in the order
table, cubes, robot. The one with three real parts is the real picture itself
and the one with three scene parts is the scene's picture itself, pixel for
pixel; both are checked. Where the real robot is taken away, the recorded
table picture and the large cube of the recording's first frame show what lay
under it.

Four more pictures change one thing in the real picture: only the scene's
large cube, only the scene's hand, only the scene's arm, and the scene's
robot with every part in the colour of its hand. Three more tell the robot's
outline from its look: the real robot painted in one flat colour (its own
median colour), the scene's robot painted in that same flat colour, and the
scene's robot moved in the picture to where it overlaps the real robot best
around the hand. Three more are about the pink cables on the recording robot,
which the scene's robot does not have: the real picture with the cables
painted over from their surroundings, the real picture with only the robot's
dark parts in the flat colour and the cables left, and the scene's robot with
the real cables laid over it. Every top picture is asked about together with
the real wrist picture.

With a lens:
    release_top_picture_parts.py <vertical lens in degrees> <pictures .npz in analysis/>
the scene's top camera is first moved along its line of sight to where that
lens shows the table at the same size (top_camera_height_scan.py), and only
three top pictures are kept and asked about: the real picture (which must
give the stored answers again), the real picture with the scene's robot as
that camera draws it, and that camera's picture of the whole scene.

Writes analysis/release-top-parts-pictures.npz, or the named file, and
analysis/<that name>-<n>.png, and refuses to overwrite the pictures file.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from sapien.render import RenderBodyComponent

from cube_places_from_pictures import coloured
from measure_objects import colour_signal
from release_aim_check import recording_rig_scene, scene_facts, scenes_before_release
from so101_sim.envs import ROBOT_COLOR
from so101_smolvla.sapien_config import HAND_LINKS
from top_camera_height_scan import TopCameraOnItsLineOfSight

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
SOURCES = ("real", "scene")
PARTS = ("table", "cubes", "robot")
SCENE_LARGE_CUBE = "top: real picture with the scene's large cube"
SCENE_HAND = "top: real picture with the scene's hand on the real arm"
SCENE_ARM = "top: real picture with the scene's arm and the real hand"
ONE_COLOUR_ROBOT = "top: real table, real cubes, scene robot in the colour of its hand"
FLAT_REAL_ROBOT = "top: real picture with the real robot in one flat colour"
FLAT_SCENE_ROBOT = "top: real table, real cubes, scene robot in that flat colour"
MOVED_SCENE_ROBOT = "top: real table, real cubes, scene robot moved onto the real robot"
NO_CABLES = "top: real picture with the robot's cables painted over"
FLAT_DARK_PARTS = "top: real picture with the robot's dark parts in one flat colour"
SCENE_ROBOT_REAL_CABLES = "top: real table, real cubes, scene robot with the real cables"
# The two robot outlines are compared this far around the hand, and the
# scene's robot is tried at every whole-pixel shift up to this far.
OUTLINE_WINDOW_PX = 160
LARGEST_SHIFT_PX = 16
# The real robot is black; nothing else in a top picture is this dark.
DARKEST_TABLE_VALUE = 70
# A cube's soft edge and the shadow at its foot go with the cube.
CUBE_MARGIN_PX = 6
ROBOT_MARGIN_PX = 2
# The real robot's thin pink cables lie this close to its dark parts.
CABLE_REACH_PX = 30
SHEET_RECORDINGS = (2, 20)


def label(sources: tuple[str, str, str]) -> str:
    return "top: " + ", ".join(f"{source} {part}" for source, part in zip(sources, PARTS))


def grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    size = 2 * pixels + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)


def moved(picture: np.ndarray, right: int, down: int) -> np.ndarray:
    """A picture or a mask moved by whole pixels; what moves in from outside is empty."""
    shift = np.float32([[1, 0, right], [0, 1, down]])
    height, width = picture.shape[:2]
    out = cv2.warpAffine(picture.astype(np.uint8), shift, (width, height), flags=cv2.INTER_NEAREST)
    return out.astype(picture.dtype)


def best_overlap_shift(real: np.ndarray, drawn: np.ndarray, window: np.ndarray) -> tuple[int, int, float, float]:
    """Whole-pixel shift (right, down) of the drawn outline that overlaps the real one best.

    Overlap is the shared area over the joint area, inside the window. Also
    returns the overlap without a shift and with the best one.
    """
    def overlap(candidate: np.ndarray) -> float:
        joint = ((real | candidate) & window).sum()
        return float((real & candidate & window).sum() / joint) if joint else 0.0

    reach = range(-LARGEST_SHIFT_PX, LARGEST_SHIFT_PX + 1)
    scores = {(right, down): overlap(moved(drawn, right, down)) for right in reach for down in reach}
    right, down = max(scores, key=scores.get)
    return right, down, scores[0, 0], scores[right, down]


def main(lens: str | None = None, output: str = "release-top-parts-pictures.npz") -> None:
    pictures_file = ANALYSIS / output
    if pictures_file.exists():
        raise SystemExit(f"{pictures_file} exists; choose another name")
    real_pictures = np.load(ANALYSIS / "release-aim-predictions.npz")
    real_top_of = dict(zip(real_pictures["episode"].tolist(), real_pictures["real_top"]))
    first_frames = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    start_of = {
        int(number): picture
        for number, moment, picture in zip(
            first_frames["episode"], first_frames["moment"], first_frames["top"])
        if moment == 0
    }
    names = [label(sources) for sources in product(SOURCES, repeat=3)]
    names += [SCENE_LARGE_CUBE, SCENE_HAND, SCENE_ARM, ONE_COLOUR_ROBOT]
    names += [FLAT_REAL_ROBOT, FLAT_SCENE_ROBOT, MOVED_SCENE_ROBOT]
    names += [NO_CABLES, FLAT_DARK_PARTS, SCENE_ROBOT_REAL_CABLES]
    rows: dict[str, list] = {
        "episode": [], "frame": [], "state": [],
        "large_cube": [], "red_cube": [], "table_height": [], "hand_px": [],
        "scene_robot_shift_px": [], "robot_overlap": [], "cable_px": [], "scene_robot_px": [],
    }
    rows.update({name: [] for name in names})
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        # Made before anything is moved: it keeps the camera as configured.
        camera = TopCameraOnItsLineOfSight(inner) if lens is not None else None

        def render_bodies(holder) -> list:
            found = []
            for obj in holder._objs:
                entity = obj.entity if hasattr(obj, "entity") else obj
                body = entity.find_component_by_type(RenderBodyComponent)
                # Parts the scene already hides stay hidden.
                if body is not None and body.visibility > 0.0:
                    found.append(body)
            return found

        links = inner.agent.robot.links
        robot_bodies = [b for link in links for b in render_bodies(link)]
        hand_bodies = [b for link in links if link.name in HAND_LINKS for b in render_bodies(link)]
        cube_bodies = {colour: render_bodies(cube) for colour, cube in inner.cubes.items()}
        all_cube_bodies = [b for bodies in cube_bodies.values() for b in bodies]
        robot_materials = [part.material for body in robot_bodies
                           for shape in body.render_shapes for part in shape.parts]

        def show(group: list, visible: bool) -> None:
            for body in group:
                body.visibility = 1.0 if visible else 0.0

        def top_picture() -> np.ndarray:
            return env._format_raw_obs(inner.get_obs())["pixels"]["top"].copy()

        def in_hand_colour() -> np.ndarray:
            """The top picture with the whole robot in the colour of its hand; the scene is put back."""
            before = [material.base_color for material in robot_materials]
            for material in robot_materials:
                material.set_base_color([*ROBOT_COLOR[:3], 1.0])
            picture = top_picture()
            for material, colour in zip(robot_materials, before):
                material.set_base_color(colour)
            return picture

        for number, frame, reading in scenes_before_release(env):
            real, start = real_top_of[number], start_of[number]
            if camera is not None:
                # Every recording is replayed from a reset; the lens is set again after it.
                height = camera.use(float(lens))
                if not rows["episode"]:
                    print(f"{lens} degree lens, top camera {1000.0 * height:.0f} mm above the robot base",
                          flush=True)
            show(robot_bodies, True), show(all_cube_bodies, True)
            scene = top_picture()
            one_colour_scene = in_hand_colour()
            show(hand_bodies, False)
            scene_without_hand = top_picture()
            show(robot_bodies, False)
            scene_cubes_on_table = top_picture()
            show(cube_bodies["red"], False)
            scene_large_on_table = top_picture()
            show(cube_bodies["blue"], False)
            scene_table = top_picture()
            show(robot_bodies, True), show(all_cube_bodies, True)
            scene_robot_mask = np.any(scene != scene_cubes_on_table, axis=2)
            scene_cube_mask = np.any(scene_cubes_on_table != scene_table, axis=2)
            scene_large_mask = np.any(scene_large_on_table != scene_table, axis=2)
            scene_hand_mask = np.any(scene != scene_without_hand, axis=2)
            scene_arm_mask = scene_robot_mask & ~scene_hand_mask

            # Real parts. The large cube lies where it lay in the first frame;
            # the red cube is in the hand, near the scene's hand.
            large_free = coloured(start, "blue", free=True)
            if large_free is None:
                print(f"recording {number}: no large cube in the first top picture", flush=True)
                continue
            dark = real.max(axis=2) < DARKEST_TABLE_VALUE
            redness = colour_signal(real, "red")
            reddish = redness > float(np.median(redness)) + 10.0
            near_hand = grow(scene_hand_mask, CABLE_REACH_PX)
            found_red = coloured(np.where(near_hand[..., None], real, scene_table), "red", free=False)
            real_red = found_red if found_red is not None else np.zeros_like(dark)
            real_large = grow(large_free, CUBE_MARGIN_PX)
            real_cubes = real_large | grow(real_red, 2)
            cable = reddish & grow(dark, CABLE_REACH_PX) & ~real_cubes
            real_robot = grow(dark | cable, ROBOT_MARGIN_PX) & ~grow(real_red, 2)
            # The real large cube as it lies free: where the arm covers it
            # now, the first frame of the recording shows it.
            free_real = np.where((real_robot & real_large)[..., None], start, real)

            def without_real_robot(gone: np.ndarray) -> np.ndarray:
                """The real picture with these robot pixels replaced by what lies under them."""
                picture = real.copy()
                picture[gone] = scene_table[gone]
                uncovered = real_large & gone
                picture[uncovered] = start[uncovered]
                return picture

            for sources in product(SOURCES, repeat=3):
                table, cubes, robot = sources
                if table == "real":
                    gone = np.zeros(real.shape[:2], dtype=bool)
                    if robot == "scene":
                        gone |= real_robot
                    picture = without_real_robot(gone)
                    if cubes == "scene":
                        away = grow(real_cubes, 2)
                        picture[away] = scene_table[away]
                else:
                    picture = scene_table.copy()
                    if cubes == "real":
                        picture[real_cubes] = free_real[real_cubes]
                if cubes == "scene":
                    picture[scene_cube_mask] = scene_cubes_on_table[scene_cube_mask]
                if robot == "scene":
                    picture[scene_robot_mask] = scene[scene_robot_mask]
                elif table == "scene" or cubes == "scene":
                    picture[real_robot] = real[real_robot]
                rows[label(sources)].append(picture)
            if not np.array_equal(rows[label(("real",) * 3)][-1], real):
                raise RuntimeError(f"recording {number}: three real parts are not the real picture")
            if not np.array_equal(rows[label(("scene",) * 3)][-1], scene):
                raise RuntimeError(f"recording {number}: three scene parts are not the scene picture")

            picture = real.copy()
            away = grow(real_large, 2)
            picture[away] = scene_table[away]
            picture[scene_large_mask] = scene_large_on_table[scene_large_mask]
            in_front = real_robot | grow(real_red, 2)
            picture[in_front] = real[in_front]
            rows[SCENE_LARGE_CUBE].append(picture)

            # The real hand with its cables: the real robot near the drawn
            # hand, apart from where the drawn arm is.
            real_hand = real_robot & near_hand & ~scene_arm_mask
            picture = without_real_robot(real_hand)
            picture[scene_hand_mask] = scene[scene_hand_mask]
            rows[SCENE_HAND].append(picture)

            picture = without_real_robot(real_robot & ~real_hand)
            picture[scene_arm_mask] = scene[scene_arm_mask]
            picture[real_hand] = real[real_hand]
            rows[SCENE_ARM].append(picture)

            picture = without_real_robot(real_robot)
            picture[scene_robot_mask] = one_colour_scene[scene_robot_mask]
            rows[ONE_COLOUR_ROBOT].append(picture)

            # Outline against look. The flat colour is the real robot's own.
            flat_colour = np.median(real[dark], axis=0).astype(real.dtype)
            picture = real.copy()
            picture[real_robot] = flat_colour
            rows[FLAT_REAL_ROBOT].append(picture)

            picture = without_real_robot(real_robot)
            picture[scene_robot_mask] = flat_colour
            rows[FLAT_SCENE_ROBOT].append(picture)

            hand_rows, hand_columns = np.nonzero(scene_hand_mask)
            hand_px = (float(hand_columns.mean()), float(hand_rows.mean()))
            rows_px, columns_px = np.indices(dark.shape)
            around_hand = np.hypot(columns_px - hand_px[0], rows_px - hand_px[1]) < OUTLINE_WINDOW_PX
            right, down, as_drawn, at_best = best_overlap_shift(
                real_robot, scene_robot_mask, around_hand)
            picture = without_real_robot(real_robot)
            moved_mask = moved(scene_robot_mask, right, down)
            picture[moved_mask] = moved(scene, right, down)[moved_mask]
            rows[MOVED_SCENE_ROBOT].append(picture)
            rows["scene_robot_shift_px"].append((right, down))
            rows["robot_overlap"].append((as_drawn, at_best))

            # The cables, with their soft edge.
            cables = grow(cable, 1)
            rows[NO_CABLES].append(
                cv2.inpaint(real, cables.astype(np.uint8), 3, cv2.INPAINT_TELEA))
            picture = real.copy()
            picture[grow(dark, ROBOT_MARGIN_PX) & ~cables] = flat_colour
            rows[FLAT_DARK_PARTS].append(picture)
            picture = rows[label(("real", "real", "scene"))][-1].copy()
            picture[cables] = real[cables]
            rows[SCENE_ROBOT_REAL_CABLES].append(picture)
            rows["cable_px"].append(int(cable.sum()))
            rows["scene_robot_px"].append(int(scene_robot_mask.sum()))

            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            for name, value in scene_facts(env).items():
                rows[name].append(value)
            rows["hand_px"].append(hand_px)
            print(f"recording {number}: pictures assembled", flush=True)
    finally:
        env.close()
    stored = {name: np.array(values) for name, values in rows.items()}
    if lens is not None:
        kept = [label(("real", "real", "real")), label(("real", "real", "scene")),
                label(("scene", "scene", "scene"))]
        for name in names:
            if name not in kept:
                del stored[name]
        names = kept
        # Read by release_aim_check.pairs_in: label, top picture, wrist picture.
        stored["pairs"] = np.array([(f"{name[5:]} + real wrist", name, "real_wrist") for name in kept])
    np.savez_compressed(pictures_file, **stored)
    print(f"assembled {len(rows['episode'])} recordings", flush=True)
    print(f"scene robot in the top picture, pixels (10 %, median, 90 %): "
          f"{np.percentile(rows['scene_robot_px'], [10, 50, 90])}", flush=True)
    shifts, overlaps = np.array(rows["scene_robot_shift_px"]), np.array(rows["robot_overlap"])
    print("scene robot moved onto the real robot, px right and down (10 %, median, 90 %): "
          f"{np.percentile(shifts[:, 0], [10, 50, 90])} and {np.percentile(shifts[:, 1], [10, 50, 90])}; "
          f"overlap as drawn {np.median(overlaps[:, 0]):.3f}, moved {np.median(overlaps[:, 1]):.3f}",
          flush=True)
    print(f"cable pixels per picture (10 %, median, 90 %): "
          f"{np.percentile(rows['cable_px'], [10, 50, 90])}", flush=True)

    # All pictures of two recordings, cut out around the hand, to look at.
    for sheet, index in enumerate(SHEET_RECORDINGS):
        cx, cy = rows["hand_px"][index]
        x0 = int(np.clip(cx - 120, 0, 640 - 240))
        y0 = int(np.clip(cy - 90, 0, 480 - 180))
        tiles = []
        for name in names:
            tile = np.ascontiguousarray(rows[name][index][y0:y0 + 180, x0:x0 + 240])
            tile = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            for line, text in enumerate(name.split(", ")):
                cv2.putText(tile, text, (4, 16 + 16 * line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            tiles.append(tile)
        tiles.extend([np.full_like(tiles[0], 255)] * (-len(tiles) % 4))
        cv2.imwrite(
            str(ANALYSIS / f"{pictures_file.stem}-{sheet}.png"),
            cv2.cvtColor(np.concatenate(
                [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)], axis=0),
                cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(*sys.argv[1:])
