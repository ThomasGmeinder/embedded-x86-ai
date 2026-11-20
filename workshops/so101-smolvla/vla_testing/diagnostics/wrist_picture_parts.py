"""Which part of the scene's wrist picture moves the model's answer during the reach, and does it matter that the scene's black parts are lighter than the recording robot's?

Run from the workshop folder, for one moment (a number of frames before the
human closes the gripper on the red cube), after early_reach_check.py render
<before> and its predict step (their answers file holds the real pictures of
the moment):
    wrist_picture_parts.py <before>                                  (.venv-sapien)
    release_aim_check.py predict early-reach-<before>-wrist-parts-pictures.npz early-reach-<before>-wrist-parts-predictions.npz   (.venv)
    early_reach_check.py report <before> early-reach-<before>-wrist-parts          (.venv-sapien)
PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim : ~/.cache/so101-stack-training
: <workshop>/vla_testing/diagnostics.

The wrist picture is taken apart into three parts: the table, the two cubes,
and the hand. Each part comes either from the real picture or from the
stack2-recording-rig scene at the same joint readings, which gives eight
wrist pictures per recording, named in the order table, cubes, hand. The one
with three real parts is the real picture itself and the one with three scene
parts is the scene's picture itself, pixel for pixel; both are checked. Where
a real part is taken away, the real table is continued over the gap from its
surroundings. release_picture_parts.py uses the same taking apart for the
moment before the release.

Black parts: the recording robot is black all over. The scene's hand and
motors are black too, but the cameras show them lighter than the recorded
pictures show the recording robot. Both pictures are therefore drawn a second
time with the scene's black parts in the one grey, of a list tried, at which
the two cameras together come closest to the recorded black; what the cameras
show for every grey tried is written to the black-levels file. The
orange printed parts of the arm are not touched.

Writes analysis/early-reach-<before>-wrist-parts-pictures.npz,
analysis/early-reach-<before>-wrist-parts-black-levels.json and
analysis/early-reach-<before>-wrist-parts-<n>.png.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
SOURCES = ("real", "scene")
PARTS = ("table", "cubes", "hand")
CUBE_COLOURS = ("red", "blue")
# The real hand is black; nothing else in a wrist picture is this dark.
DARKEST_TABLE_VALUE = 70
HAND_MARGIN_PX = 2
CUBE_MARGIN_PX = 3
# A picture's black level is read this far inside its dark area, away from the edge.
BLACK_INSIDE_PX = 3
SMALLEST_BLACK_AREA_PX = 500
# Greys the scene's black parts are drawn in to find the one the cameras show
# as dark as the recorded black, lightest first; 0.05 is what the scene builds.
GREYS_TRIED = (0.05, 0.03, 0.02, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0)
SHEET_RECORDINGS = (2, 20)
HAND_AS_RECORDED = "wrist: real table, real cubes, scene hand as dark as recorded"
SCENE_AS_RECORDED = "wrist: scene picture with its black parts as dark as recorded"
TOP_AS_RECORDED = "top: scene picture with its black parts as dark as recorded"


def label(sources: tuple[str, str, str]) -> str:
    return "wrist: " + ", ".join(f"{source} {part}" for source, part in zip(sources, PARTS))


def grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    size = 2 * pixels + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)


def continued(picture: np.ndarray, gap: np.ndarray) -> np.ndarray:
    """The picture with the gap filled from what surrounds it."""
    return cv2.inpaint(picture, gap.astype(np.uint8), 5, cv2.INPAINT_TELEA)


def render_bodies(holder) -> list:
    """The drawn bodies of a robot link or a cube that the scene shows."""
    from sapien.render import RenderBodyComponent

    found = []
    for obj in holder._objs:
        entity = obj.entity if hasattr(obj, "entity") else obj
        body = entity.find_component_by_type(RenderBodyComponent)
        # Parts the scene already hides stay hidden.
        if body is not None and body.visibility > 0.0:
            found.append(body)
    return found


def show(group: list, visible: bool) -> None:
    for body in group:
        body.visibility = 1.0 if visible else 0.0


@dataclass(frozen=True)
class WristParts:
    """One moment's wrist picture, the real one and the scene's, taken apart."""

    real: np.ndarray
    scene: np.ndarray
    # The scene without its robot, and without robot and cubes.
    scene_cubes_on_table: np.ndarray
    scene_table: np.ndarray
    scene_hand_mask: np.ndarray
    scene_cube_mask: np.ndarray
    real_hand: np.ndarray
    # Per cube colour, where the real picture shows that cube (no pixel if it shows none).
    real_cube: dict[str, np.ndarray]
    real_cubes: np.ndarray
    # The real picture as it would show without the hand in front.
    real_without_hand: np.ndarray

    def mixed(
        self, sources: tuple[str, str, str],
        scene_cubes: np.ndarray | None = None, cube_mask: np.ndarray | None = None,
        scene_hand: np.ndarray | None = None,
    ) -> np.ndarray:
        """One wrist picture with table, cubes and hand from the named sources.

        "scene_cubes" is another picture of the scene's cubes on its table and
        "cube_mask" says where it shows a cube; "scene_hand" is another picture
        of the scene with its hand in the same place.
        """
        table, cubes, hand = sources
        scene_cubes = self.scene_cubes_on_table if scene_cubes is None else scene_cubes
        cube_mask = self.scene_cube_mask if cube_mask is None else cube_mask
        scene_hand = self.scene if scene_hand is None else scene_hand
        if table == "real":
            gone = np.zeros(self.real.shape[:2], dtype=bool)
            if cubes == "scene":
                gone |= self.real_cubes
            if hand == "scene":
                gone |= self.real_hand
            picture = continued(self.real, gone) if gone.any() else self.real.copy()
            if cubes == "real" and hand == "scene":
                uncovered = self.real_cubes & self.real_hand
                picture[uncovered] = self.real_without_hand[uncovered]
        else:
            picture = self.scene_table.copy()
            if cubes == "real":
                picture[self.real_cubes] = self.real_without_hand[self.real_cubes]
        if cubes == "scene":
            picture[cube_mask] = scene_cubes[cube_mask]
        if hand == "scene":
            picture[self.scene_hand_mask] = scene_hand[self.scene_hand_mask]
        elif table == "scene" or cubes == "scene":
            picture[self.real_hand] = self.real[self.real_hand]
        return picture


class SceneInParts:
    """The scene's cameras, with the robot and the cubes shown or hidden."""

    def __init__(self, env) -> None:
        self.env = env
        self.inner = env._env.unwrapped
        self.robot_bodies = [
            body for link in self.inner.agent.robot.links for body in render_bodies(link)]
        self.cube_bodies = {
            colour: render_bodies(cube) for colour, cube in self.inner.cubes.items()}
        self.all_cube_bodies = [body for bodies in self.cube_bodies.values() for body in bodies]

    def pictures(self) -> dict[str, np.ndarray]:
        seen = self.env._format_raw_obs(self.inner.get_obs())["pixels"]
        return {camera: seen[camera].copy() for camera in ("top", "wrist")}

    def taken_apart(self, real: np.ndarray) -> WristParts:
        """The wrist picture of the scene as it stands, against the real one of the same moment."""
        from cube_places_from_pictures import coloured

        show(self.robot_bodies, True), show(self.all_cube_bodies, True)
        scene = self.pictures()["wrist"]
        show(self.robot_bodies, False)
        scene_cubes_on_table = self.pictures()["wrist"]
        show(self.all_cube_bodies, False)
        scene_table = self.pictures()["wrist"]
        show(self.robot_bodies, True), show(self.all_cube_bodies, True)

        real_hand = grow(real.max(axis=2) < DARKEST_TABLE_VALUE, HAND_MARGIN_PX)
        no_pixels = np.zeros(real.shape[:2], dtype=bool)
        real_cube = {}
        for colour in CUBE_COLOURS:
            found = coloured(real, colour, free=False)
            real_cube[colour] = grow(found, CUBE_MARGIN_PX) if found is not None else no_pixels
        return WristParts(
            real=real, scene=scene,
            scene_cubes_on_table=scene_cubes_on_table, scene_table=scene_table,
            scene_hand_mask=np.any(scene != scene_cubes_on_table, axis=2),
            scene_cube_mask=np.any(scene_cubes_on_table != scene_table, axis=2),
            real_hand=real_hand, real_cube=real_cube,
            real_cubes=np.any(list(real_cube.values()), axis=0),
            real_without_hand=continued(real, real_hand),
        )


def black_materials(inner) -> list:
    """The materials of every drawn robot part that is not painted orange: hand and motors."""
    from sapien.render import RenderBodyComponent

    from so101_smolvla.sapien_config import ROBOT_ORANGE
    from so101_smolvla.sapien_env_worker import _robot_part_color

    found = []
    for link in inner.agent.robot.links:
        for obj in link._objs:
            body = obj.entity.find_component_by_type(RenderBodyComponent)
            if body is None:
                continue
            for shape in body.render_shapes:
                if _robot_part_color(link.name, shape.filename) != ROBOT_ORANGE:
                    found.extend(part.material for part in shape.parts)
    if not found:
        raise RuntimeError("the scene's robot has no black part")
    return found


def recorded_black_level(picture: np.ndarray) -> float | None:
    """Median brightness well inside the dark area of a real picture, or None if it has none."""
    dark = (picture.max(axis=2) < DARKEST_TABLE_VALUE).astype(np.uint8)
    size = 2 * BLACK_INSIDE_PX + 1
    inside = cv2.erode(dark, np.ones((size, size), np.uint8)).astype(bool)
    if inside.sum() < SMALLEST_BLACK_AREA_PX:
        return None
    return float(np.median(picture[inside].mean(axis=1)))


def level_in(picture: np.ndarray, mask: np.ndarray) -> float | None:
    if mask.sum() < SMALLEST_BLACK_AREA_PX:
        return None
    return float(np.median(picture[mask].mean(axis=1)))


def median_of(levels: list[float | None]) -> float:
    kept = [level for level in levels if level is not None]
    if not kept:
        raise RuntimeError("no picture shows a black part large enough to measure")
    return float(np.median(kept))


def main(before: str) -> None:
    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from grasp_aim_check import scenes_before_closing
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments
    from so101_smolvla.sapien_env_worker import _configure_robot_appearance

    stem = f"early-reach-{int(before)}-wrist-parts"
    recorded = np.load(ANALYSIS / f"early-reach-{int(before)}-predictions.npz")
    real_of = {
        (number, frame): (top, wrist)
        for number, frame, top, wrist in zip(
            recorded["episode"].tolist(), recorded["frame"].tolist(),
            recorded["real_top"], recorded["real_wrist"])
    }
    mixes = [label(sources) for sources in product(SOURCES, repeat=3)]
    wrist_names = [*mixes, HAND_AS_RECORDED, SCENE_AS_RECORDED]
    pairs = [(f"real top + {name[7:]}", "real_top", name) for name in wrist_names]
    pairs += [
        ("scene top + real wrist", "top", "real_wrist"),
        ("scene top with its black parts as dark as recorded + real wrist",
         TOP_AS_RECORDED, "real_wrist"),
        ("scene top + scene wrist", "top", label(("scene",) * 3)),
        ("scene top + scene wrist, black parts as dark as recorded in both",
         TOP_AS_RECORDED, SCENE_AS_RECORDED),
    ]
    rows: dict[str, list] = {
        "episode": [], "frame": [], "state": [], "red_cube": [], "table_height": [],
        "top": [], TOP_AS_RECORDED: [],
    }
    rows.update({name: [] for name in wrist_names})

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    try:
        _configure_robot_appearance(env)
        env._env.reset(seed=0)
        scene = SceneInParts(env)
        inner = scene.inner
        materials = black_materials(inner)
        as_built = [material.base_color for material in materials]
        grey_as_built = float(np.mean([colour[:3] for colour in as_built]))

        def paint(grey: float | None) -> None:
            """The black parts in one grey, or as the scene builds them."""
            for material, colour in zip(materials, as_built):
                material.set_base_color(colour if grey is None else [grey, grey, grey, 1.0])

        # First pass: how bright both cameras show the scene's black parts in
        # each grey tried, and how bright the real pictures of the same
        # moments show the recording robot's black.
        shown: dict[float, dict[str, list]] = {
            tried: {"top": [], "wrist": []} for tried in GREYS_TRIED}
        as_built_shown: dict[str, list] = {"top": [], "wrist": []}
        real_shown: dict[str, list] = {"top": [], "wrist": []}
        for number, frame, _ in scenes_before_closing(env, int(before)):
            seen = scene.pictures()
            drawn = {}
            for tried in GREYS_TRIED:
                paint(tried)
                drawn[tried] = scene.pictures()
            paint(None)
            for camera, real in zip(("top", "wrist"), real_of[(number, frame)]):
                # The black parts are where the picture changes with their grey.
                black = np.any(drawn[GREYS_TRIED[0]][camera] != drawn[GREYS_TRIED[-1]][camera], axis=2)
                as_built_shown[camera].append(level_in(seen[camera], black))
                real_shown[camera].append(recorded_black_level(real))
                for tried in GREYS_TRIED:
                    shown[tried][camera].append(level_in(drawn[tried][camera], black))
        as_built_level = {camera: median_of(levels) for camera, levels in as_built_shown.items()}
        recorded_level = {camera: median_of(levels) for camera, levels in real_shown.items()}
        tried_level = {
            tried: {camera: median_of(levels) for camera, levels in cameras.items()}
            for tried, cameras in shown.items()}
        # One grey for both cameras: the scene has one robot.
        grey = min(GREYS_TRIED, key=lambda tried: sum(
            (tried_level[tried][camera] - recorded_level[camera]) ** 2
            for camera in ("top", "wrist")))

        # Second pass: the pictures.
        for number, frame, reading in scenes_before_closing(env, int(before)):
            real = real_of[(number, frame)][1]
            parts = scene.taken_apart(real)
            top = scene.pictures()["top"]
            paint(grey)
            dark = scene.pictures()
            paint(None)
            for sources in product(SOURCES, repeat=3):
                rows[label(sources)].append(parts.mixed(sources))
            if not np.array_equal(rows[label(("real",) * 3)][-1], real):
                raise RuntimeError(f"recording {number}: three real parts are not the real picture")
            if not np.array_equal(rows[label(("scene",) * 3)][-1], parts.scene):
                raise RuntimeError(f"recording {number}: three scene parts are not the scene picture")
            rows[HAND_AS_RECORDED].append(
                parts.mixed(("real", "real", "scene"), scene_hand=dark["wrist"]))
            rows[SCENE_AS_RECORDED].append(dark["wrist"])
            rows["top"].append(top)
            rows[TOP_AS_RECORDED].append(dark["top"])
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            rows["red_cube"].append(inner.cubes["red"].pose.p[0].cpu().numpy().copy())
            rows["table_height"].append(float(inner.TABLE_SURFACE_Z))
            print(f"recording {number}: pictures assembled", flush=True)
        if not np.array_equal(scene.pictures()["top"], rows["top"][-1]):
            raise RuntimeError("the black parts were not put back to the scene's own colour")
    finally:
        env.close()
    if rows["episode"] != recorded["episode"].tolist():
        raise RuntimeError("the real pictures are not of these recordings")

    levels = {
        "frames_before_closing": int(before),
        "recordings": len(rows["episode"]),
        "black_parts_base_grey_as_built_mean": round(grey_as_built, 4),
        "black_parts_base_grey_chosen": grey,
        "brightness_of_black_parts_0_to_255_median": {
            camera: {
                "recorded_pictures": round(recorded_level[camera], 1),
                "scene_as_built": round(as_built_level[camera], 1),
                "scene_with_chosen_grey": round(tried_level[grey][camera], 1),
                "scene_per_grey_tried": {
                    str(tried): round(tried_level[tried][camera], 1) for tried in GREYS_TRIED},
            } for camera in ("top", "wrist")
        },
    }
    print(json.dumps(levels, indent=1), flush=True)
    (ANALYSIS / f"{stem}-black-levels.json").write_text(
        json.dumps(levels, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(
        ANALYSIS / f"{stem}-pictures.npz", pairs=np.array(pairs),
        **{name: np.array(values) for name, values in rows.items()},
    )
    print(f"assembled {len(rows['episode'])} recordings, {int(before)} frames before "
          f"the human closes the gripper", flush=True)

    for sheet, index in enumerate(SHEET_RECORDINGS):
        tiles = []
        for name in [*wrist_names, "top", TOP_AS_RECORDED]:
            tile = cv2.resize(rows[name][index], (320, 240), interpolation=cv2.INTER_AREA)
            for line, text in enumerate(name.split(", ")):
                cv2.putText(tile, text, (4, 14 + 14 * line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(tile)
        tiles.extend([np.full_like(tiles[0], 255)] * (-len(tiles) % 4))
        cv2.imwrite(
            str(ANALYSIS / f"{stem}-{sheet}.png"),
            cv2.cvtColor(np.concatenate(
                [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)], axis=0),
                cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(*sys.argv[1:])
