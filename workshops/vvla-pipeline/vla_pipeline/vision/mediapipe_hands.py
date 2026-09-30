# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""MediaPipe Hands on CPU - pinch, hand size, and roll for gesture-mimic.

Several signals come out of one hand detection, driving three parts of the
SO-101 in gesture-mimic (gripper, forward/back, and wrist-roll):

- **pinch** - thumb-tip↔index-tip distance normalized by hand size
  (wrist→middle-MCP). Invariant to camera distance: ``0`` = fingers touching,
  ``~1+`` = wide open → drives the gripper.
- **hand_size** - apparent span of the hand (wrist→middle-MCP) as a fraction of
  the frame height. Grows as the hand nears the camera, so it is the
  "how close is the hand" signal that drives the robot forward/back.
- **roll** - in-plane angle of the hand (wrist→middle-MCP vector), degrees,
  ``0`` = fingers pointing straight up, ``+`` = tilted toward image-right. This
  is the *fallback* wrist-roll cue (stable, always available while a hand is
  seen).
- **roll_ti** - in-plane orientation of the **thumb-tip→index-tip vector**,
  degrees in the full ``(-180, 180]`` (``0`` = pointing toward image-right). It
  is kept *directed* (thumb→index, not a folded line) so it never sign-flips when
  the hand merely tilts past vertical - the earlier ``(-90,90]`` fold was the
  cause of wrist_roll flipping without a real rotation. This is the *primary*
  wrist-roll cue: the gripper follows the line between index finger and thumb.
- **roll_ref** - in-plane orientation of the **knuckle line** (index-MCP →
  pinky-MCP), degrees in ``(-180, 180]``. This is a STABLE reference landmark
  that doesn't move when the fingers pinch, used to disambiguate/steady the roll
  so it only changes when the hand truly rotates.
- **roll_ti_weight** - reliability of ``roll_ti`` in ``[0, 1]``: the thumb↔index
  pixel separation scaled by ``roll_ti_full_px``. When the fingers pinch nearly
  shut the line is too short to give a trustworthy angle, so the weight falls and
  the mapper leans on ``roll_ref`` / ``roll``.

Independent test
----------------
    python -m vla_pipeline.vision.mediapipe_hands           # webcam overlay
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
# Knuckle line across the palm - a roll REFERENCE that doesn't change when the
# fingers pinch (unlike the thumb↔index line). index-MCP → pinky-MCP.
INDEX_MCP, PINKY_MCP = 5, 17


@dataclass
class HandObservation:
    """One frame's hand signals: pinch, position, and the various roll cues, for the gesture-mimic mapper."""

    pinch: float  # normalized thumb↔index distance (gripper)
    thumb_px: tuple[int, int]  # pixel coords
    index_px: tuple[int, int]
    handedness: str  # "Left" / "Right" (as seen by MediaPipe)
    hand_size: float = 0.0  # wrist→middle-MCP span / frame height (depth cue)
    roll: float = 0.0  # in-plane hand angle, deg (0 = fingers up) - fallback roll
    roll_ti: float = 0.0  # thumb→index vector angle, deg in (-180,180] - primary roll
    roll_ref: float = 0.0  # knuckle-line angle, deg in (-180,180] - stable reference
    roll_ti_weight: float = 0.0  # reliability of roll_ti in [0,1] (line length based)
    wrist_px: tuple[int, int] = (0, 0)  # wrist landmark, for the roll overlay


class HandTracker:
    """Thin wrapper around ``mediapipe.solutions.hands`` (CPU)."""

    def __init__(self, cfg: dict | None = None):
        import mediapipe as mp  # heavy import, keep local

        m = (cfg or {}).get("mediapipe", {})
        # Thumb↔index separation (px) at which roll_ti is fully trusted (weight 1).
        # Below it the line is too short to give a reliable angle, so the weight
        # ramps down linearly and the mapper leans on the wrist→knuckle `roll`.
        self._roll_ti_full_px = float(m.get("roll_ti_full_px", 40.0))
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(m.get("max_num_hands", 1)),
            min_detection_confidence=float(m.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(m.get("min_tracking_confidence", 0.5)),
        )

    def process(self, frame_bgr: np.ndarray) -> tuple[HandObservation | None, object]:
        """Return (observation for the first detected hand, raw results)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, results

        lm = results.multi_hand_landmarks[0].landmark
        thumb = np.array([lm[THUMB_TIP].x, lm[THUMB_TIP].y])
        index = np.array([lm[INDEX_TIP].x, lm[INDEX_TIP].y])
        wrist = np.array([lm[WRIST].x, lm[WRIST].y])
        mcp = np.array([lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y])

        # Pinch normalizes by the hand size in *normalized* coords so it stays
        # distance-invariant (unchanged behavior).
        hand_size_norm = float(np.linalg.norm(mcp - wrist)) or 1e-6
        pinch = float(np.linalg.norm(thumb - index)) / hand_size_norm

        # Depth cue + roll are computed in *pixels* so aspect ratio doesn't skew
        # them. hand_size = wrist→MCP span as a fraction of frame height (grows
        # as the hand nears the camera); roll = the same vector's in-plane angle
        # (0 = fingers point straight up, + = tilted toward image-right).
        wrist_px = np.array([wrist[0] * w, wrist[1] * h])
        mcp_px = np.array([mcp[0] * w, mcp[1] * h])
        v = mcp_px - wrist_px
        hand_size = float(np.linalg.norm(v)) / float(h or 1)
        roll = float(np.degrees(np.arctan2(v[0], -v[1])))

        # Primary roll cue: the DIRECTED thumb→index vector. Because thumb and
        # index are distinct points the vector has an unambiguous head/tail, so
        # atan2 gives a full (-180,180] with NO sign-flip as the hand tilts past
        # vertical (the old (-90,90] fold flipped here - the bug we're fixing).
        # 0° = pointing toward image-right.
        thumb_px = np.array([thumb[0] * w, thumb[1] * h])
        index_px = np.array([index[0] * w, index[1] * h])
        ti = index_px - thumb_px
        ti_len = float(np.linalg.norm(ti))
        roll_ti = float(np.degrees(np.arctan2(-ti[1], ti[0])))  # (-180,180], 0 = right
        roll_ti_weight = float(
            np.clip(ti_len / max(self._roll_ti_full_px, 1e-6), 0.0, 1.0)
        )

        # Reference roll cue: the knuckle line (index-MCP → pinky-MCP). It barely
        # moves when the fingers pinch, so it's a steady anchor the mapper can
        # fall back to (and use to keep continuity) when the thumb↔index line is
        # short/unreliable. Same directed (-180,180] convention.
        imcp_px = np.array([lm[INDEX_MCP].x * w, lm[INDEX_MCP].y * h])
        pmcp_px = np.array([lm[PINKY_MCP].x * w, lm[PINKY_MCP].y * h])
        ref = pmcp_px - imcp_px
        roll_ref = float(np.degrees(np.arctan2(-ref[1], ref[0])))

        handedness = "Unknown"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        obs = HandObservation(
            pinch=pinch,
            thumb_px=(int(thumb[0] * w), int(thumb[1] * h)),
            index_px=(int(index[0] * w), int(index[1] * h)),
            handedness=handedness,
            hand_size=hand_size,
            roll=roll,
            roll_ti=roll_ti,
            roll_ref=roll_ref,
            roll_ti_weight=roll_ti_weight,
            wrist_px=(int(wrist_px[0]), int(wrist_px[1])),
        )
        return obs, results

    def draw(
        self, frame: np.ndarray, obs: HandObservation | None, results
    ) -> np.ndarray:
        """Draw the MediaPipe landmarks and a pinch/roll HUD overlay onto frame."""
        out = frame.copy()
        if results and results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    out, hand_lms, self._mp_hands.HAND_CONNECTIONS
                )
        if obs:
            cv2.line(out, obs.thumb_px, obs.index_px, (0, 0, 255), 3)
            mid = (
                (obs.thumb_px[0] + obs.index_px[0]) // 2,
                (obs.thumb_px[1] + obs.index_px[1]) // 2,
            )
            cv2.putText(
                out,
                f"pinch {obs.pinch:.2f} ({obs.handedness})",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                out,
                f"size {obs.hand_size:.3f}  roll {obs.roll:+.0f}",
                (mid[0], mid[1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                out,
                f"roll_ti {obs.roll_ti:+.0f} (w {obs.roll_ti_weight:.2f}) ref {obs.roll_ref:+.0f}",
                (mid[0], mid[1] + 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
        return out

    def close(self) -> None:
        """Release the MediaPipe Hands model."""
        self.hands.close()


# =============================================================================
# Standalone component test (live overlay)
# =============================================================================


def main() -> None:
    """CLI entry point: live webcam overlay of hand tracking output."""
    from vla_pipeline.utils.config import load_config

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="MediaPipe hand/pinch tracking (CPU) - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--cam", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tracker = HandTracker(cfg)
    cap = cv2.VideoCapture(args.cam if args.cam is not None else cfg["camera"]["index"])
    if not cap.isOpened():
        raise SystemExit("ERROR: could not open camera")

    fps_window: deque[float] = deque(maxlen=30)
    print("Show your hand. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        obs, results = tracker.process(frame)
        fps_window.append(1.0 / max(time.perf_counter() - t0, 1e-6))

        out = tracker.draw(frame, obs, results)
        cv2.putText(
            out,
            f"MediaPipe Hands (CPU)  {sum(fps_window)/len(fps_window):5.1f} FPS",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imshow("MediaPipe Hands", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
