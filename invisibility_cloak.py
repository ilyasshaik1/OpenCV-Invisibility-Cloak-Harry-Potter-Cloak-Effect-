import cv2
import numpy as np
import os
import time
from collections import deque

# ----------------------- Utility -----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ----------------------- Color Presets -----------------------

# HSV ranges are given as (lower, upper). Hue is [0,179] in OpenCV.
# RED wraps around the hue circle, so we use two ranges and combine.
COLOR_PRESETS = {
    "RED": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([179, 255, 255])),
    ],
    "BLUE": [
        (np.array([94, 80, 2]), np.array([126, 255, 255])),
    ],
    "GREEN": [
        (np.array([35, 80, 40]), np.array([85, 255, 255])),
    ],
}


# ----------------------- Trackbar (Calibration) -----------------------

class HSVCalibrator:
    def __init__(self, window_name="HSV Calibration"):
        self.window = window_name
        self.visible = False
        self.values = {
            "H_low": 0, "S_low": 120, "V_low": 70,
            "H_high": 10, "S_high": 255, "V_high": 255,
        }

    def _on_change(self, _):
        for k in list(self.values.keys()):
            self.values[k] = cv2.getTrackbarPos(k, self.window)

    def show(self, initial_lower=(0, 120, 70), initial_upper=(10, 255, 255)):
        if self.visible:
            return
        self.visible = True
        cv2.namedWindow(self.window)
        cv2.createTrackbar("H_low", self.window, int(initial_lower[0]), 179, self._on_change)
        cv2.createTrackbar("S_low", self.window, int(initial_lower[1]), 255, self._on_change)
        cv2.createTrackbar("V_low", self.window, int(initial_lower[2]), 255, self._on_change)
        cv2.createTrackbar("H_high", self.window, int(initial_upper[0]), 179, self._on_change)
        cv2.createTrackbar("S_high", self.window, int(initial_upper[1]), 255, self._on_change)
        cv2.createTrackbar("V_high", self.window, int(initial_upper[2]), 255, self._on_change)

    def hide(self):
        if self.visible:
            cv2.destroyWindow(self.window)
            self.visible = False

    def get_range(self):
        if not self.visible:
            return None
        v = self.values
        lower = np.array([v["H_low"], v["S_low"], v["V_low"]])
        upper = np.array([v["H_high"], v["S_high"], v["V_high"]])
        return (lower, upper)


# ----------------------- Background Capture -----------------------

def capture_background(cap, num_frames=60, delay_ms=15):
    """Capture a smooth background by averaging several frames while the scene is empty."""
    grabbed_frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)  # mirror for a natural webcam feel
        grabbed_frames.append(frame.astype(np.float32))
        cv2.waitKey(delay_ms)
    if not grabbed_frames:
        return None
    avg = np.mean(grabbed_frames, axis=0).astype(np.uint8)
    return avg


# ----------------------- Masking -----------------------

def build_mask(hsv, target_color: str, calibrator: HSVCalibrator):
    """Return a cleaned binary mask for the target cloak color."""
    mask = None

    # If calibrator is visible, use its single range; otherwise use presets (may be two ranges for RED)
    if calibrator.visible and calibrator.get_range() is not None:
        lower, upper = calibrator.get_range()
        mask = cv2.inRange(hsv, lower, upper)
    else:
        ranges = COLOR_PRESETS[target_color]
        for lower, upper in ranges:
            part = cv2.inRange(hsv, lower, upper)
            mask = part if mask is None else (mask | part)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


# ----------------------- Main -----------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Try a different index in VideoCapture(0/1/2)...")
        return

    target_color = "RED"
    calibrator = HSVCalibrator()
    bg = None

    ensure_dir("frames")

    # Rolling FPS estimator for smoother display
    times = deque(maxlen=30)

    print("\nControls: b=background  1=RED 2=BLUE 3=GREEN  c=calibrate  s=snapshot  q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Timestamp for FPS
        t = time.time()
        times.append(t)
        fps = 0.0
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0] + 1e-6)

        display = frame.copy()

        if bg is None:
            # Prompt to capture background
            bar = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
            text(bar, "Press 'b' to capture background (stand aside)", (10, 40), 0.8, (255, 255, 255), 2)
            display = np.vstack([bar, display])
            cv2.imshow("Invisibility Cloak", display)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = build_mask(hsv, target_color, calibrator)
            mask_inv = cv2.bitwise_not(mask)

            # Keep non-cloak areas from the current frame
            foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)
            # Fill cloak area with background
            cloak_area = cv2.bitwise_and(bg, bg, mask=mask)
            output = cv2.addWeighted(foreground, 1, cloak_area, 1, 0)

            # HUD
            text(output, f"Target: {target_color}  |  FPS: {fps:4.1f}", (10, 25), 0.7, (0, 0, 0), 3)
            text(output, f"Target: {target_color}  |  FPS: {fps:4.1f}", (10, 25), 0.7, (255, 255, 255), 1)
            text(output, "Keys: b(bg) 1/2/3(color) c(calib) s(snap) q(quit)", (10, 50))

            cv2.imshow("Invisibility Cloak", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # Count-down overlay before background capture
            for sec in [3, 2, 1]:
                temp = frame.copy()
                text(temp, f"Capturing background in {sec}...", (10, 40), 0.9, (0, 0, 0), 4)
                text(temp, f"Capturing background in {sec}...", (10, 40), 0.9, (0, 255, 255), 2)
                cv2.imshow("Invisibility Cloak", temp)
                cv2.waitKey(450)
            bg = capture_background(cap, num_frames=60, delay_ms=10)
            if bg is None:
                print("Background capture failed. Try again.")
        elif key == ord('1'):
            target_color = "RED"
        elif key == ord('2'):
            target_color = "BLUE"
        elif key == ord('3'):
            target_color = "GREEN"
        elif key == ord('c'):
            if calibrator.visible:
                calibrator.hide()
            else:
                # Set reasonable defaults for the selected color when opening
                if target_color == "RED":
                    lo, hi = (0, 120, 70), (10, 255, 255)
                elif target_color == "BLUE":
                    lo, hi = (94, 80, 2), (126, 255, 255)
                else:
                    lo, hi = (35, 80, 40), (85, 255, 255)
                calibrator.show(lo, hi)
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join("frames", f"frame_{ts}.png")
            # If we have output window, try to grab it; else save raw frame
            if bg is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = build_mask(hsv, target_color, calibrator)
                mask_inv = cv2.bitwise_not(mask)
                foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)
                cloak_area = cv2.bitwise_and(bg, bg, mask=mask)
                output = cv2.addWeighted(foreground, 1, cloak_area, 1, 0)
                cv2.imwrite(path, output)
            else:
                cv2.imwrite(path, frame)
            print(f"Saved snapshot: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
