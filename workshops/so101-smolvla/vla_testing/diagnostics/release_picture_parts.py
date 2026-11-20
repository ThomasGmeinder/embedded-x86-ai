"""Which part of the scene's wrist picture makes the model let the red cube go too high?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after release_aim_check.py render and predict:
    release_picture_parts.py
then, with file names in analysis/:
    release_aim_check.py predict release-picture-parts-pictures.npz release-picture-parts-predictions.npz
    release_aim_check.py report  release-picture-parts-pictures.npz release-picture-parts-predictions.npz release-picture-parts.json

Moment and recordings are those of release_aim_check.py: 25 frames before the
human lets the red cube go. The wrist picture is taken apart into three
parts: the table, the two cubes, and the hand. Each part comes either from the
real picture or from the stack2-recording-rig scene at the same joint
readings, which gives eight wrist pictures per recording, named in the order
table, cubes, hand. The one with three real parts is the real picture itself
and the one with three scene parts is the scene's picture itself, pixel for
pixel; both are checked. Where a real part is taken away, the real table is
continued over the gap from its surroundings.

Two more wrist pictures and one top picture show the scene's cubes in the
colours measured in the recordings. Five more change one thing in the real
wrist picture: only the large cube from the scene, only the red cube from the
scene, the scene's large cube put where and turned as the recording's first
top picture shows it (cube_places_from_pictures.py), and the scene's cubes
under light from all sides, in the scene's and in the recorded colours. Three
more put the scene's red cube where a cube squeezed between the jaws sits:
midway between them instead of half its size from the fixed jaw. Every
wrist picture is asked about together with the real top picture, the top
picture with the real wrist picture.

Writes analysis/release-picture-parts-pictures.npz and
analysis/release-picture-parts-<n>.png.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from release_aim_check import recording_rig_scene, scene_facts, scenes_before_release
# The taking apart of a wrist picture into table, cubes and hand lives in
# wrist_picture_parts.py, which uses it for the reach.
from wrist_picture_parts import SOURCES, SceneInParts, continued, label, show

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
RECORDED_CUBES_IN_REAL = "wrist: real table, scene cubes in the recorded colours, real hand"
RECORDED_CUBES_IN_SCENE = "wrist: scene table, scene cubes in the recorded colours, scene hand"
RECORDED_CUBES_TOP = "top: scene top with cubes in the recorded colours"
SCENE_LARGE_CUBE = "wrist: real picture with the scene's large cube"
SCENE_RED_CUBE = "wrist: real picture with the scene's red cube"
PLACED_LARGE_CUBE = "wrist: real picture with the scene's large cube placed as in the top picture"
SQUEEZED_RED_CUBE = "wrist: real picture with the scene's red cube where a squeezed cube sits"
SQUEEZED_CUBES = "wrist: real table, scene cubes with the red one where a squeezed cube sits, real hand"
SQUEEZED_SCENE = "wrist: scene picture with the red cube where a squeezed cube sits"
SOFT_LIGHT_CUBES = "wrist: real picture with the scene's cubes under soft light"
SOFT_LIGHT_RECORDED_CUBES = (
    "wrist: real picture with the scene's cubes in the recorded colours under soft light")
# Soft light: strength of the light from all sides and of each of the scene's
# two lamps. Not measured; it only makes a cube's faces nearly equally bright.
SOFT_AMBIENT = 0.9
SOFT_LAMP = 0.25
SHEET_RECORDINGS = (2, 20)


def main() -> None:
    real_pictures = np.load(ANALYSIS / "release-aim-predictions.npz")
    real_wrist_of = dict(zip(real_pictures["episode"].tolist(), real_pictures["real_wrist"]))
    recorded_paint = json.loads(
        (ANALYSIS / "objects" / "fitted-paint.json").read_text())["paint_base_colour"]
    places = np.load(ANALYSIS / "cube-places-from-pictures.npz")
    top_places = {
        int(number): (place / 1000.0, float(turn))
        for number, place, turn in zip(
            places["episode"], places["large_place_mm"], places["large_turn_deg"])
    }
    unplaced: list[int] = []
    names = [label(sources) for sources in product(SOURCES, repeat=3)]
    names += [RECORDED_CUBES_IN_REAL, RECORDED_CUBES_IN_SCENE, RECORDED_CUBES_TOP,
              SCENE_LARGE_CUBE, SCENE_RED_CUBE, PLACED_LARGE_CUBE,
              SOFT_LIGHT_CUBES, SOFT_LIGHT_RECORDED_CUBES,
              SQUEEZED_RED_CUBE, SQUEEZED_CUBES, SQUEEZED_SCENE]
    rows: dict[str, list] = {
        "episode": [], "frame": [], "state": [],
        "large_cube": [], "red_cube": [], "table_height": [],
    }
    rows.update({name: [] for name in names})
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        scene_in_parts = SceneInParts(env)
        robot_bodies, cube_bodies = scene_in_parts.robot_bodies, scene_in_parts.cube_bodies
        pictures = scene_in_parts.pictures
        cube_materials = [
            (colour, part.material)
            for colour, bodies in cube_bodies.items()
            for body in bodies for shape in body.render_shapes for part in shape.parts
        ]

        @contextmanager
        def soft_light():
            """Light from all sides instead of two lamps, to see whether shading matters."""
            systems = [scene.render_system for scene in inner.scene.sub_scenes]
            before = [(system.ambient_light, [light.color for light in system.lights])
                      for system in systems]
            for system in systems:
                system.ambient_light = [SOFT_AMBIENT] * 3
                for light in system.lights:
                    light.color = [SOFT_LAMP] * 3
            try:
                yield
            finally:
                for system, (ambient, colours) in zip(systems, before):
                    system.ambient_light = ambient
                    for light, colour in zip(system.lights, colours):
                        light.color = colour

        def in_recorded_colours() -> dict[str, np.ndarray]:
            """Both pictures with the cubes in the recorded colours; the scene is put back."""
            before = [(m.base_color, m.roughness, m.metallic, m.specular) for _, m in cube_materials]
            for colour, material in cube_materials:
                material.set_base_color([*recorded_paint[colour], 1.0])
                material.roughness, material.metallic, material.specular = 1.0, 0.0, 0.0
            seen = pictures()
            for (_, material), (colour, roughness, metallic, specular) in zip(cube_materials, before):
                material.set_base_color(colour)
                material.roughness, material.metallic, material.specular = (
                    roughness, metallic, specular)
            return seen

        for number, frame, reading in scenes_before_release(env):
            real = real_wrist_of[number]
            parts = scene_in_parts.taken_apart(real)
            mixed = parts.mixed
            scene_cubes_on_table, scene_table = parts.scene_cubes_on_table, parts.scene_table
            real_hand, real_without_hand = parts.real_hand, parts.real_without_hand
            recorded = in_recorded_colours()
            show(robot_bodies, False)
            recorded_cubes_on_table = in_recorded_colours()["wrist"]
            show(robot_bodies, True)

            for sources in product(SOURCES, repeat=3):
                rows[label(sources)].append(mixed(sources))
            if not np.array_equal(rows[label(("real",) * 3)][-1], real):
                raise RuntimeError(f"recording {number}: three real parts are not the real picture")
            if not np.array_equal(rows[label(("scene",) * 3)][-1], parts.scene):
                raise RuntimeError(f"recording {number}: three scene parts are not the scene picture")
            rows[RECORDED_CUBES_IN_REAL].append(
                mixed(("real", "scene", "real"), recorded_cubes_on_table))
            rows[RECORDED_CUBES_IN_SCENE].append(recorded["wrist"])
            rows[RECORDED_CUBES_TOP].append(recorded["top"])
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            for name, value in scene_facts(env).items():
                rows[name].append(value)

            # One cube at a time, in the real picture.
            real_red, real_large = parts.real_cube["red"], parts.real_cube["blue"]
            show(robot_bodies, False), show(cube_bodies["red"], False)
            large_on_table = pictures()["wrist"]
            show(cube_bodies["red"], True)
            scene_large_mask = np.any(large_on_table != scene_table, axis=2)
            scene_red_mask = np.any(scene_cubes_on_table != large_on_table, axis=2)

            def with_scene_large_cube(large_alone: np.ndarray, where: np.ndarray) -> np.ndarray:
                picture = continued(real, real_large)
                picture[where] = large_alone[where]
                in_front = real_red | real_hand
                picture[in_front] = real[in_front]
                return picture

            rows[SCENE_LARGE_CUBE].append(with_scene_large_cube(large_on_table, scene_large_mask))
            picture = continued(real_without_hand, real_red)
            picture[scene_red_mask] = scene_cubes_on_table[scene_red_mask]
            picture[real_hand] = real[real_hand]
            rows[SCENE_RED_CUBE].append(picture)

            # The recorded red cube gives way in the grip (red_cube_in_hand.py,
            # gripper_from_wrist_pictures.py): it sits midway between the jaws
            # as they stand at the recorded reading. The scene's rigid cube
            # sits half its size from the fixed jaw, where the replayed grasp
            # left it. Here it is moved towards the fixed jaw by the difference.
            red = inner.cubes["red"]
            rigid_place, rigid_turn = red.pose.p.clone(), red.pose.q.clone()
            fixed, moving = (
                finger.pose.p[0].cpu().numpy().astype(np.float64)
                for finger in (inner.agent.finger1_tip, inner.agent.finger2_tip))
            between_jaws = float(np.linalg.norm(moving - fixed))
            squeeze = max(0.0, (float(inner.cube_sizes["red"]) - between_jaws) / 2)
            towards_fixed_jaw = squeeze * (fixed - moving) / between_jaws
            red.set_pose(Pose.create_from_pq(
                p=rigid_place + torch.as_tensor(
                    towards_fixed_jaw[None], dtype=torch.float32, device=inner.device),
                q=rigid_turn))
            squeezed_cubes = pictures()["wrist"]
            squeezed_red_mask = np.any(squeezed_cubes != large_on_table, axis=2)
            picture = continued(real_without_hand, real_red)
            picture[squeezed_red_mask] = squeezed_cubes[squeezed_red_mask]
            picture[real_hand] = real[real_hand]
            rows[SQUEEZED_RED_CUBE].append(picture)
            rows[SQUEEZED_CUBES].append(mixed(
                ("real", "scene", "real"), squeezed_cubes,
                np.any(squeezed_cubes != scene_table, axis=2)))
            show(robot_bodies, True)
            rows[SQUEEZED_SCENE].append(pictures()["wrist"])
            show(robot_bodies, False)
            red.set_pose(Pose.create_from_pq(p=rigid_place, q=rigid_turn))

            with soft_light():
                rows[SOFT_LIGHT_CUBES].append(
                    mixed(("real", "scene", "real"), pictures()["wrist"]))
                rows[SOFT_LIGHT_RECORDED_CUBES].append(
                    mixed(("real", "scene", "real"), in_recorded_colours()["wrist"]))

            # The large cube where, and turned as, the first top picture shows it.
            place = top_places.get(number)
            if place is None:
                rows[PLACED_LARGE_CUBE].append(rows[SCENE_LARGE_CUBE][-1])
                unplaced.append(number)
            else:
                (x, y), turn = place
                inner.cubes["blue"].set_pose(Pose.create_from_pq(
                    p=torch.tensor(
                        [[x, y, inner.TABLE_SURFACE_Z + inner.cube_sizes["blue"] / 2]],
                        dtype=torch.float32, device=inner.device),
                    q=torch.tensor(
                        np.asarray([euler2quat(0, 0, np.deg2rad(turn))]),
                        dtype=torch.float32, device=inner.device)))
                show(cube_bodies["red"], False)
                placed = pictures()["wrist"]
                show(cube_bodies["red"], True)
                rows[PLACED_LARGE_CUBE].append(
                    with_scene_large_cube(placed, np.any(placed != scene_table, axis=2)))
            show(robot_bodies, True)
            print(f"recording {number}: pictures assembled", flush=True)
        print(f"no place in the top picture for the large cube of recordings {unplaced}; "
              f"their picture keeps the scene's place", flush=True)
    finally:
        env.close()
    np.savez_compressed(
        ANALYSIS / "release-picture-parts-pictures.npz",
        **{name: np.array(values) for name, values in rows.items()},
    )
    print(f"assembled {len(rows['episode'])} recordings", flush=True)

    for sheet, index in enumerate(SHEET_RECORDINGS):
        tiles = []
        for name in names:
            tile = cv2.resize(rows[name][index], (320, 240), interpolation=cv2.INTER_AREA)
            for line, text in enumerate(name.split(", ")):
                cv2.putText(tile, text, (4, 14 + 14 * line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(tile)
        tiles.extend([np.full_like(tiles[0], 255)] * (-len(tiles) % 4))
        cv2.imwrite(
            str(ANALYSIS / f"release-picture-parts-{sheet}.png"),
            cv2.cvtColor(np.concatenate(
                [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)], axis=0),
                cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
