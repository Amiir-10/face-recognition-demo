"""OpenCV drawing -- bounding boxes, labels, HUD overlays."""
import cv2
import numpy as np

from config import (
    BOX_COLOR_KNOWN,
    BOX_COLOR_UNKNOWN,
    FONT_SCALE,
    BOX_THICKNESS,
    UNKNOWN_LABEL,
    LABEL_BG_COLOR,
)


class DisplayRenderer:
    """Draws face detection results and UI elements on frames."""

    FONT = cv2.FONT_HERSHEY_DUPLEX

    def draw_results(
        self,
        frame: np.ndarray,
        face_locations: list[tuple],
        results: list[tuple[str, float]],
    ) -> np.ndarray:
        """Draw bounding boxes and name labels on the frame.

        Args:
            frame: BGR image to annotate.
            face_locations: List of (top, right, bottom, left).
            results: List of (name, confidence) matching face_locations order.

        Returns:
            Annotated frame.
        """
        for (top, right, bottom, left), (name, confidence) in zip(face_locations, results):
            is_known = name != UNKNOWN_LABEL
            color = BOX_COLOR_KNOWN if is_known else BOX_COLOR_UNKNOWN

            cv2.rectangle(frame, (left, top), (right, bottom), color, BOX_THICKNESS)

            if is_known:
                label = f"{name.title()} - {confidence:.0f}%"
            else:
                label = f"{UNKNOWN_LABEL} - Press R to register"

            text_size = cv2.getTextSize(label, self.FONT, FONT_SCALE, 1)[0]
            label_top = top - text_size[1] - 10
            if label_top < 0:
                label_top = bottom + 5

            cv2.rectangle(
                frame,
                (left, label_top - 5),
                (left + text_size[0] + 10, label_top + text_size[1] + 5),
                LABEL_BG_COLOR,
                cv2.FILLED,
            )

            cv2.putText(
                frame,
                label,
                (left + 5, label_top + text_size[1]),
                self.FONT,
                FONT_SCALE,
                color,
                1,
            )

        return frame

    def draw_hud(self, frame: np.ndarray, fps: float, face_count: int) -> np.ndarray:
        """Draw FPS counter and registered face count."""
        hud_text = f"FPS: {fps:.1f} | Registered: {face_count}"
        cv2.putText(frame, hud_text, (10, 30), self.FONT, 0.6, (255, 255, 255), 1)

        controls = "R:Register  I:Import  L:List  D:Delete  Q:Quit"
        h = frame.shape[0]
        cv2.putText(frame, controls, (10, h - 15), self.FONT, 0.5, (200, 200, 200), 1)

        return frame

    def draw_registration_overlay(self, frame: np.ndarray, message: str) -> np.ndarray:
        """Draw a registration status message centered on frame."""
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(message, self.FONT, 1.0, 2)[0]
        x = (w - text_size[0]) // 2
        y = (h + text_size[1]) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        cv2.putText(frame, message, (x, y), self.FONT, 1.0, (0, 255, 255), 2)
        return frame
