# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""YOLOv26s object detection on the Ryzen AI NPU - voice-command grounding.

Same structure as :class:`vla_pipeline.vision.yolo_pose_npu.PoseEstimator`
(letterbox → VitisAI/CPU session → de-letterboxed outputs) but for the
80-class COCO detection head (raw ``[1, 84, 8400]``: xywh, 80 class scores,
run through NMS; end-to-end ``[1, 300, 6]``: xyxy, score, class). Used by:

- ``pick_place`` - locate the object named in the voice command ("pick up
  the **ball**") in the arm/mount camera frame;
- ``grip`` - optional verification that an object is present between the jaws.

Spoken names are matched through a synonym table (``"ball" → sports ball``)
so natural phrasing works without retraining.

Independent test
----------------
    python -m vla_pipeline.vision.yolo_detect_npu                  # webcam overlay
    python -m vla_pipeline.vision.yolo_detect_npu --find ball
    python -m vla_pipeline.vision.yolo_detect_npu --device cpu
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from vla_pipeline.utils.config import load_config
from vla_pipeline.utils.npu_session import active_provider, build_npu_session, run_session, size_npu
from vla_pipeline.vision.yolo_pose_npu import LetterboxInfo, head_rows, letterbox, nms_keep, unmap_boxes, xywh_to_xyxy

logger = logging.getLogger(__name__)

# A detection has no measured size. These are the class priors, in metres, for the
# objects this arm is asked to pick: the collision sizes in the twin's scene.
CLASS_SIZE = {
    "sports ball": (0.04, 0.04, 0.04),
    "bottle": (0.04, 0.04, 0.085),
    "book": (0.06, 0.09, 0.012),
    "toothbrush": (0.024, 0.12, 0.02),
}

COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Spoken-word → COCO class synonyms (lowercased, applied after exact match).
SYNONYMS = {
    "ball": "sports ball",
    "sports_ball": "sports ball",
    "soccer": "sports ball",
    "soccer ball": "sports ball",
    "football": "sports ball",
    "foot ball": "sports ball",
    "cube": None,
    "block": None,
    "phone": "cell phone",
    "mobile": "cell phone",
    "glass": "wine glass",
    "mug": "cup",
    "pen": None,
    "pencil": None,
    "brush": "toothbrush",
    "drink": "bottle",
    "water bottle": "bottle",
    "computer": "laptop",
    "plant": "potted plant",
    "teddy": "teddy bear",
    "bear": "teddy bear",
    "controller": "remote",
}
# Classes with value None aren't in COCO-80; pick_place reports "can't
# recognize that object" rather than guessing. The twin uses four real COCO
# classes: sports ball, book, bottle, and toothbrush.


@dataclass
class Detection:
    """One detected object: class name, confidence, and pixel-space box."""

    name: str
    score: float
    box: np.ndarray  # xyxy, original pixels

    @property
    def center(self) -> tuple[float, float]:
        """Center of the box in pixels."""
        return (
            float(self.box[0] + self.box[2]) / 2.0,
            float(self.box[1] + self.box[3]) / 2.0,
        )

    @property
    def area(self) -> float:
        """Area of the box in square pixels."""
        return float(
            max(self.box[2] - self.box[0], 0.0) * max(self.box[3] - self.box[1], 0.0)
        )


def resolve_class(spoken: str) -> str | None:
    """Map a spoken object name to a COCO class, or None if unknown."""
    s = (spoken or "").strip().lower()
    if s in COCO_NAMES:
        return s
    if s in SYNONYMS:
        return SYNONYMS[s]
    # Loose containment ("the red ball" → ball → sports ball).
    for word in s.split():
        if word in COCO_NAMES:
            return word
        if word in SYNONYMS and SYNONYMS[word]:
            return SYNONYMS[word]
    for name in COCO_NAMES:
        if name in s:
            return name
    return None


def decode_detections(outputs, info: LetterboxInfo, orig_shape, conf: float) -> list[Detection]:
    """Decode a YOLOv26s-detect output into :class:`Detection` s in original pixels."""
    preds, raw = head_rows(outputs[0], 6, 84)
    if raw:
        boxes = xywh_to_xyxy(preds[:, :4].astype(np.float32))
        scores, classes = preds[:, 4:].max(axis=1), preds[:, 4:].argmax(axis=1)
    else:
        boxes, scores, classes = preds[:, :4].astype(np.float32), preds[:, 4], preds[:, 5]
    mask = scores > conf
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
    if raw and len(boxes):
        keep = nms_keep(boxes, scores)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    boxes = unmap_boxes(boxes, info, orig_shape) if len(boxes) else boxes
    out: list[Detection] = []
    for box, score, cls in zip(boxes, scores, classes):
        idx = int(cls)
        name = COCO_NAMES[idx] if 0 <= idx < len(COCO_NAMES) else f"cls{idx}"
        out.append(Detection(name=name, score=float(score), box=box))
    return out


def draw_detections(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw labelled detection boxes; returns a copy."""
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d.box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 160, 255), 2)
        cv2.putText(out, f"{d.name} {d.score:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 2)
    return out


class ObjectDetector:
    """YOLOv26s detection ONNX inference (raw head [1, 84, 8400] + NMS; end-to-end head [1, 300, 6])."""

    def __init__(self, cfg: dict, device: str | None = None):
        y = cfg["yolo_detect"]
        self.imgsz: int = int(y["imgsz"])
        size_npu(cfg)
        self.session = build_npu_session(
            y["onnx"],
            device or y["device"],
            vitisai_config=y.get("vitisai_config", "config/vaiep_config.json"),
            cache_dir=cfg.get("system", {}).get("cache_dir", "cache"),
            cache_key=y["cache_key"],
        )
        self.provider = active_provider(self.session)
        on_npu = self.provider == "VitisAIExecutionProvider"
        self.conf_threshold: float = float(
            y.get("npu_conf_threshold", y["conf_threshold"]) if on_npu else y["conf_threshold"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(
        self, frame_bgr: np.ndarray, conf: float | None = None
    ) -> list[Detection]:
        """Run detection on one frame and return all detections above the confidence threshold."""
        conf = self.conf_threshold if conf is None else conf
        nchw, info = letterbox(frame_bgr, self.imgsz)
        return decode_detections(run_session(self.session, None, {self.input_name: nchw}), info, frame_bgr.shape, conf)

    def find(
        self, frame_bgr: np.ndarray, spoken_name: str, conf: float | None = None
    ) -> tuple[list[Detection], str | None]:
        """All detections matching a spoken object name.

        Returns ``(matches, resolved_class)``; ``resolved_class`` is ``None``
        when the spoken name doesn't map to any known class.
        """
        cls = resolve_class(spoken_name)
        if cls is None:
            return [], None
        dets = [d for d in self.infer(frame_bgr, conf) if d.name == cls]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets, cls

    def draw(self, frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
        """Draw detection boxes and labels onto a copy of ``frame``."""
        return draw_detections(frame, dets)


# =============================================================================
# Standalone component test (live overlay)
# =============================================================================


def table_point(pixel, camera_info, optical_in_base, table_z: float):
    """Where one pixel's ray meets the plane ``z = table_z``, or ``None``.

    ``pixel`` is ``(u, v)``. ``optical_in_base`` is ``(rotation, translation)``
    of the camera's optical frame (x right, y down, z forward) in the arm's
    base frame. The ray comes from ``image_geometry`` and the camera info.
    """
    from image_geometry import PinholeCameraModel

    model = PinholeCameraModel()
    model.from_camera_info(camera_info)
    centre = (float(pixel[0]), float(pixel[1]))
    ray = np.asarray(model.project_pixel_to_3d_ray(centre), float)
    rotation, origin = optical_in_base
    direction = np.asarray(rotation, float) @ ray
    origin = np.asarray(origin, float)
    if abs(direction[2]) < 1e-9:
        return None
    scale = (float(table_z) - origin[2]) / direction[2]
    if scale <= 0.0:
        return None
    return origin + direction * scale


def object_axis(frame, box, min_elongation):
    """The long axis of the object inside ``box``, as two full-frame pixels, or ``None``.

    The box is axis-aligned, so its width and height cannot tell a 30° brush
    from a −30° one. An Otsu threshold separates the object from the crop, the
    class that touches fewer border pixels is the object, and ``minAreaRect``
    fits that outline. A round object (long side under ``min_elongation`` times
    the short side) has no axis.
    """
    height, width = frame.shape[:2]
    x0 = max(0, int(round(float(box[0]))))
    y0 = max(0, int(round(float(box[1]))))
    x1 = min(width, int(round(float(box[2]))))
    y1 = min(height, int(round(float(box[3]))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _level, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def _border(binary) -> int:
        return int(np.count_nonzero(binary[0, :]) + np.count_nonzero(binary[-1, :])
                   + np.count_nonzero(binary[:, 0]) + np.count_nonzero(binary[:, -1]))

    if _border(mask) > _border(cv2.bitwise_not(mask)):
        mask = cv2.bitwise_not(mask)
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(contour)
    long_side, short_side = max(rect_w, rect_h), min(rect_w, rect_h)
    if short_side <= 0.0 or long_side / short_side < float(min_elongation):
        return None
    theta = math.radians(angle)
    if rect_h > rect_w:
        theta += math.pi / 2
    half = 0.5 * long_side
    dx, dy = half * math.cos(theta), half * math.sin(theta)
    return (x0 + cx - dx, y0 + cy - dy), (x0 + cx + dx, y0 + cy + dy)


def locate_detection(box, camera_info, optical_in_base, table_z, class_name: str,
                     frame=None, min_elongation=None):
    """Where a wrist-camera detection sits: the box centre's ray at the class centre.

    ``image_geometry`` casts the ray from this frame's camera info. The class
    size puts the centre half a height above the table, and the ray meets that
    plane. Meeting the table and then lifting that point leaves the centre
    past the object: 12.3 mm for the ball and 44.4 mm for the bottle. Returns
    the locate dict, or ``None``.

    The yaw puts the box's X along the detection box's vertical, from its
    top edge's middle to its bottom edge's middle, on that plane. The wrist
    camera's image vertical is the jaws' closing axis, so the jaws close
    between those edges. A class whose footprint is longer one way
    (``CLASS_SIZE`` X differs from Y) takes the yaw of its outline's long
    axis instead, when :func:`object_axis` finds one. A round object's
    outline in this oblique view is its side, which has no footprint axis.
    """
    if class_name not in CLASS_SIZE:
        return None
    size = CLASS_SIZE[class_name]
    centre_z = float(table_z) + 0.5 * float(size[2])
    centre_x = 0.5 * (float(box[0]) + float(box[2]))
    centre_px = (centre_x, 0.5 * (float(box[1]) + float(box[3])))
    hit = table_point(centre_px, camera_info, optical_in_base, centre_z)
    if hit is None:
        return None
    top = table_point((centre_x, float(box[1])), camera_info, optical_in_base, centre_z)
    bottom = table_point((centre_x, float(box[3])), camera_info, optical_in_base, centre_z)
    if top is None or bottom is None:
        return None
    yaw = math.atan2(float(bottom[1] - top[1]), float(bottom[0] - top[0]))
    if size[0] != size[1] and frame is not None and min_elongation is not None:
        ends = object_axis(frame, box, min_elongation)
        if ends is not None:
            start = table_point(ends[0], camera_info, optical_in_base, centre_z)
            end = table_point(ends[1], camera_info, optical_in_base, centre_z)
            if start is not None and end is not None:
                # CLASS_SIZE puts a long object's length on its Y axis.
                yaw = math.atan2(float(end[1] - start[1]), float(end[0] - start[0])) - math.pi / 2
    return {
        "xyz": np.array(hit, float),
        "width": float(min(size[0], size[1])),
        "height": float(size[2]),
        "shape": class_name,
        "yaw": yaw,
    }


def collision_object(name: str, centre, size, frame: str = "base_link", yaw: float = 0.0):
    """A box at ``centre`` for the planning scene. ``size`` is ``(x, y, z)`` metres.

    ``yaw`` is the rotation about the vertical, so the box's Y axis follows a
    long object. Zero leaves the box aligned with ``frame``.
    """
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [float(v) for v in size]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in centre)
    half = 0.5 * float(yaw)
    pose.orientation.z = math.sin(half)
    pose.orientation.w = math.cos(half)
    obj = CollisionObject()
    obj.id = name
    obj.header.frame_id = frame
    obj.operation = CollisionObject.ADD
    obj.primitives.append(primitive)
    obj.primitive_poses.append(pose)
    return obj


def main() -> None:
    """CLI entry point: live camera overlay of detections, optionally filtered by --find."""
    from vla_pipeline.vision.camera import make_camera

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="YOLOv26s detection on Ryzen AI NPU - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--role", choices=["mount", "arm"], default="arm")
    parser.add_argument(
        "--find", default=None, help="highlight matches for a spoken name"
    )
    parser.add_argument("--device", choices=["npu", "cpu"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    det = ObjectDetector(cfg, device=args.device)
    cam = make_camera(cfg, args.role)
    fps_window: deque[float] = deque(maxlen=30)
    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            if args.find:
                dets, cls = det.find(frame, args.find)
                label = f"find '{args.find}' → {cls}: {len(dets)}"
            else:
                dets = det.infer(frame)
                label = f"{len(dets)} objects"
            fps_window.append(1.0 / max(time.perf_counter() - t0, 1e-6))
            out = det.draw(frame, dets)
            cv2.putText(
                out,
                f"{label}  {sum(fps_window)/len(fps_window):5.1f} FPS",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("yolo26s detect", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
