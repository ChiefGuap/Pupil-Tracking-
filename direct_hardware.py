# -*- coding: utf-8 -*-
"""
Direct USB Camera Benchmark (Bypassing Pupil Labs Software)
============================================================

PURPOSE:
  This script connects DIRECTLY to the Pupil Labs Eye Camera hardware via
  USB, completely bypassing Pupil Capture and Pupil Service. The goal is to
  measure the absolute maximum frame rate and minimum latency achievable
  when we remove the Pupil Labs software "middleman."

WHY THIS MATTERS:
  Pupil Capture/Service adds overhead (network serialization, plugin pipeline,
  GUI rendering). By going straight to the USB camera with OpenCV, we can
  determine the raw hardware limit and see how much speed we lose by using
  the Pupil Labs stack.

HOW IT WORKS:
  1. Opens the USB camera directly using OpenCV's VideoCapture.
  2. For each frame, runs a simple PUPIL detection algorithm:
       - Convert to grayscale
       - Threshold (inverted) to isolate the dark pupil
       - Compute centroid via image moments
  3. Measures and logs:
       - Hardware grab time (how long it takes to read a frame from USB)
       - Algorithm processing time (how long our pupil detection takes)
       - Total latency per frame (grab + processing)
       - Rolling FPS (frames per second over the entire session)
  4. Displays a live video feed with a crosshair on the detected pupil center.

IMPORTANT:
  You MUST close Pupil Capture and Pupil Service before running this script,
  otherwise they will hold exclusive access to the camera device.

REQUIREMENTS:
  - Python packages: opencv-python, numpy
  - Pupil Labs eye camera connected via USB
  - Pupil Capture/Service must NOT be running
"""

# ============================================================================
# IMPORTS
# ============================================================================
import cv2          # OpenCV — camera access, image processing, GUI display
import time         # Timing functions (perf_counter for high-precision benchmarks)
import numpy as np  # NumPy — numerical operations (not heavily used here but available)
import logging      # Python's logging framework (replaces print statements)
import os           # File system operations (create log directory)
import sys          # System info (Python version for session header)
from datetime import datetime  # Human-readable timestamps for log filenames

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Create a 'logs/' directory next to this script if it doesn't already exist.
# This is the same directory used by bridge_code.py — both scripts share it.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a unique log filename using the current date and time.
# Example: logs/direct_hw_20260323_143000.log
log_filename = os.path.join(
    LOG_DIR,
    f"direct_hw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Configure the logger with two outputs:
#   1. File handler — captures EVERYTHING (DEBUG level) to the log file
#   2. Console handler — shows INFO and above on the terminal
logger = logging.getLogger("direct_hw")
logger.setLevel(logging.DEBUG)

# File handler — detailed log for post-run analysis.
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

# Console handler — cleaner output for real-time monitoring.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
# CAMERA_INDEX: Which video device to open.
#   0 = usually the laptop's built-in webcam
#   1 = usually the first external USB camera (Pupil Labs eye cam)
#   2 = try this if 1 doesn't work
# Change this if the camera isn't detected on index 1.
CAMERA_INDEX = 1

# PUPIL_THRESHOLD: Brightness cutoff for dark-pupil detection.
#   The pupil is the darkest region in an eye image. Any pixel with a
#   grayscale value BELOW this threshold is considered part of the pupil.
#   50 is a reasonable default — lower = stricter (only very dark pixels),
#   higher = more inclusive (may pick up eyelashes/shadows).
PUPIL_THRESHOLD = 50


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """
    Main entry point. Connects to the USB camera, runs pupil detection on
    every frame, and reports timing metrics (FPS, grab latency, processing
    speed) to both the console and a log file.
    """

    # ------------------------------------------------------------------
    # LOG SESSION HEADER — record system info at the start of every run
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("DIRECT HARDWARE BENCHMARK — SESSION START")
    logger.info("=" * 70)
    logger.info(f"Timestamp        : {datetime.now().isoformat()}")
    logger.info(f"Python Version   : {sys.version}")
    logger.info(f"OpenCV Version   : {cv2.__version__}")
    logger.info(f"Camera Index     : {CAMERA_INDEX}")
    logger.info(f"Pupil Threshold  : {PUPIL_THRESHOLD}")
    logger.info(f"Log File         : {log_filename}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: CONNECT TO USB CAMERA
    # ------------------------------------------------------------------
    logger.info(f"Attempting to connect directly to USB Camera #{CAMERA_INDEX}...")

    # cv2.VideoCapture opens a connection to the camera hardware.
    # On macOS/Linux it uses V4L2 or AVFoundation backends automatically.
    # On Windows, cv2.CAP_DSHOW can be passed for faster DirectShow access.
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Check if the camera was successfully opened.
    if not cap.isOpened():
        logger.error(
            "❌ Could not open camera. Is Pupil Service still running? (Close it!) "
            f"Or try changing CAMERA_INDEX (currently {CAMERA_INDEX}) to 0 or 2."
        )
        return

    # Request 120 FPS from the camera. The camera will set the closest supported
    # rate. Many Pupil Labs cameras support 120 Hz at lower resolutions.
    cap.set(cv2.CAP_PROP_FPS, 120)

    # Read back the actual settings the camera driver applied.
    # These may differ from what we requested (hardware limitations).
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(
        f"✅ Camera Connected! Resolution: {actual_width:.0f}x{actual_height:.0f} "
        f"@ {actual_fps:.0f} FPS (requested 120)"
    )

    # ------------------------------------------------------------------
    # INITIALIZE TRACKING VARIABLES
    # ------------------------------------------------------------------
    frame_count = 0          # Total frames processed this session.
    start_time = time.time() # Session start time (for rolling FPS calculation).

    logger.info("Press 'q' in the video window to quit.")
    logger.info("-" * 70)

    # ------------------------------------------------------------------
    # MAIN PROCESSING LOOP — runs until 'q' is pressed or camera fails
    # ------------------------------------------------------------------
    while True:
        # ==============================================================
        # A. GRAB A FRAME FROM THE USB CAMERA (hardware I/O)
        # ==============================================================
        # time.perf_counter() gives nanosecond-precision timestamps,
        # much better than time.time() for benchmarking short operations.
        t_grab_start = time.perf_counter()

        # cap.read() pulls one frame from the USB bus. This is the
        # hardware bottleneck — it blocks until the camera delivers
        # a frame (typically 1/fps seconds).
        ret, frame = cap.read()

        if not ret:
            logger.error("Failed to grab frame — camera may have disconnected.")
            break

        t_grab_end = time.perf_counter()

        # ==============================================================
        # B. PUPIL DETECTION — find the center of the dark pupil
        # ==============================================================
        t_proc_start = time.perf_counter()

        # Step B1: Convert the BGR color frame to grayscale.
        # Pupil detection only needs brightness values, not color.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step B2: Inverted binary threshold.
        # The pupil is DARK (low pixel values). THRESH_BINARY_INV turns
        # dark pixels → WHITE (255) and bright pixels → BLACK (0).
        # This creates a white mask of the pupil region.
        _, thresh = cv2.threshold(
            gray, PUPIL_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )

        # Step B3: Compute image moments to find the centroid of the white mask.
        # M["m00"] = total area of white pixels.
        # If m00 > 0, the centroid is at (m10/m00, m01/m00).
        M = cv2.moments(thresh)
        pupil_x, pupil_y = -1, -1

        if M["m00"] != 0:
            # Compute the center of mass (centroid) of the detected pupil.
            pupil_x = int(M["m10"] / M["m00"])
            pupil_y = int(M["m01"] / M["m00"])

            # Draw a red crosshair on the original frame for visual feedback.
            # This appears in the cv2.imshow() window so the user can verify
            # the detection is tracking correctly.
            cv2.drawMarker(
                frame, (pupil_x, pupil_y),
                (0, 0, 255),           # Color: red in BGR
                cv2.MARKER_CROSS,      # Crosshair style
                20,                    # Marker size in pixels
                2                      # Line thickness
            )

        t_proc_end = time.perf_counter()

        # ==============================================================
        # C. CALCULATE AND LOG PERFORMANCE METRICS
        # ==============================================================
        # Hardware grab time: how long the USB read blocked (camera → RAM).
        grab_time_ms = (t_grab_end - t_grab_start) * 1000

        # Algorithm processing time: how long our pupil detection took.
        proc_time_ms = (t_proc_end - t_proc_start) * 1000

        # Total end-to-end latency for this frame (grab + processing).
        total_latency_ms = grab_time_ms + proc_time_ms

        # Update the rolling FPS counter.
        frame_count += 1
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Log detailed metrics every 10 frames to the log FILE (DEBUG level).
        # This avoids flooding but gives enough data for analysis.
        if frame_count % 10 == 0:
            logger.debug(
                f"Frame {frame_count:>6d} | "
                f"FPS: {current_fps:6.1f} | "
                f"Grab: {grab_time_ms:6.1f}ms | "
                f"Algo: {proc_time_ms:5.2f}ms | "
                f"Latency: {total_latency_ms:6.1f}ms | "
                f"Pupil: ({pupil_x:>4d}, {pupil_y:>4d})"
            )

        # Log summary to CONSOLE (INFO level) every 100 frames.
        if frame_count % 100 == 0:
            logger.info(
                f"📊 Frame {frame_count} | "
                f"FPS: {current_fps:.1f} | "
                f"Grab: {grab_time_ms:.1f}ms | "
                f"Algo: {proc_time_ms:.2f}ms | "
                f"Total: {total_latency_ms:.1f}ms"
            )

        # ==============================================================
        # D. DISPLAY VIDEO FEED WITH CROSSHAIR OVERLAY
        # ==============================================================
        cv2.imshow("Direct Pupil Tracker", frame)
        # Uncomment the line below to also see the binary threshold mask
        # (white = detected pupil region, black = everything else):
        # cv2.imshow("Threshold Mask (What the computer sees)", thresh)

        # Check if 'q' was pressed to exit. waitKey(1) = 1ms delay
        # (fast enough not to slow down the loop).
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("🛑 'q' pressed — shutting down.")
            break

    # ------------------------------------------------------------------
    # CLEANUP — release hardware resources and close windows
    # ------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

    # Log session summary.
    total_elapsed = time.time() - start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

    logger.info("=" * 70)
    logger.info("DIRECT HARDWARE BENCHMARK — SESSION END")
    logger.info(f"Total Frames     : {frame_count}")
    logger.info(f"Total Time       : {total_elapsed:.2f}s")
    logger.info(f"Average FPS      : {avg_fps:.1f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()