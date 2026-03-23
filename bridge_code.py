# -*- coding: utf-8 -*-
"""
Dual-Mode Bridge: Pupil Detection & Laser Spot Detection (Master)
=================================================================

This script acts as a BRIDGE between the Pupil Labs eye-tracking software
(Pupil Capture / Pupil Service) and a LabVIEW data acquisition system.

HOW IT WORKS:
  1. It connects to Pupil Service via ZMQ (a messaging protocol) to receive
     real-time pupil position data and raw eye camera frames.
  2. It listens for UDP commands from LabVIEW to switch between operating modes.
  3. Depending on the active mode, it either:
       - Forwards pupil (x,y) coordinates to LabVIEW (Mode 1)
       - Detects a laser reflection in the camera frame and sends its position (Mode 0)
       - Runs a speed/accuracy comparison of two laser detection algorithms (Mode 2)
       - Collects calibration samples (Mode 3) and computes offsets (Mode 4/5)
  4. All coordinates are sent to LabVIEW as UDP packets containing comma-separated
     normalized values (0.0 to 1.0 range).

MODES SUMMARY:
  Mode 0 = Laser detection using CENTROID algorithm (moment-based)
  Mode 1 = Pupil tracking via Pupil Labs (default mode)
  Mode 2 = Algorithm speed test: runs BOTH centroid and blob detection, logs timing
  Mode 3 = Calibration data collection (stores pupil positions)
  Mode 4 = Finalize calibration (compute mean offset from collected samples)
  Mode 5 = Reset calibration (clear offsets back to zero)

REQUIREMENTS:
  - Pupil Capture or Pupil Service must be running on localhost
  - Python packages: zmq, msgpack, numpy, opencv-python
  - LabVIEW must send/receive UDP on the configured ports

Updates: Added Algorithm Execution Timers for Speed Analysis in Mode 2.
         Added comprehensive logging with timestamped log files.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import zmq          # ZeroMQ — messaging library to talk to Pupil Service
import msgpack      # MessagePack — efficient binary serialization (Pupil's format)
import socket       # Python's built-in UDP/TCP socket library (for LabVIEW comms)
import numpy as np  # NumPy — numerical operations (array math, mean calculations)
import cv2          # OpenCV — computer vision (image processing, blob detection)
import threading    # For running the LabVIEW command listener in a background thread
import sys          # System utilities (Python version info for logging)
import time         # High-precision timing for algorithm benchmarks
import logging      # Python's logging framework (replaces print statements)
import os           # File system operations (create log directory)
from datetime import datetime  # Human-readable timestamps for log filenames

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Create a 'logs/' directory next to this script if it doesn't already exist.
# All log files will be written here with timestamps in their filenames.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a unique log filename using the current date and time.
# Example: logs/bridge_20260323_142500.log
log_filename = os.path.join(
    LOG_DIR,
    f"bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Configure the root logger:
#   - Level DEBUG means ALL messages (DEBUG, INFO, WARNING, ERROR) are captured
#   - Format includes timestamp, severity level, and message text
#   - Two handlers: one writes to the log FILE, the other to the CONSOLE
logger = logging.getLogger("bridge")
logger.setLevel(logging.DEBUG)

# File handler — writes everything (DEBUG and above) to the log file
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

# Console handler — writes INFO and above to the terminal (less noisy)
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
# Network settings for communicating with LabVIEW via UDP.
LABVIEW_IP = "127.0.0.1"    # IP address of LabVIEW (localhost = same machine)
UDP_SEND_PORT = 50020        # Port we SEND data TO (LabVIEW listens here)
UDP_RECV_PORT = 50021        # Port we RECEIVE commands FROM (LabVIEW sends here)

# Which eye camera to subscribe to. Pupil Labs headsets have:
#   Eye 0 = right eye, Eye 1 = left eye (from the wearer's perspective)
EYE_ID = 1

# Laser detection parameters:
#   LASER_THRESHOLD — brightness cutoff. Pixels brighter than this value (0-255)
#   are considered potential laser reflections. 200 is very bright = only laser spots.
LASER_THRESHOLD = 200

#   BLUR_KERNEL — size of the Gaussian blur kernel applied before thresholding.
#   Blurring reduces noise so we don't detect random bright pixels as laser spots.
BLUR_KERNEL = (9, 9)

# ============================================================================
# GLOBAL STATE VARIABLES
# ============================================================================
# current_mode: The active operating mode, changed by LabVIEW commands.
#   Starts in Mode 1 (pupil tracking) by default.
current_mode = 1

# lock: A threading lock to protect current_mode from race conditions.
#   The command listener thread and main loop thread both access current_mode,
#   so we need this to prevent simultaneous read/write conflicts.
lock = threading.Lock()

# Calibration offset values. These are subtracted from pupil coordinates to
# correct for any systematic bias in the tracking data.
offset_x = 0.0
offset_y = 0.0

# Buffer that collects pupil positions during calibration (Mode 3).
# When calibration ends (Mode 4), we average these to compute the offset.
calibration_buffer = []


# ============================================================================
# BLOB DETECTOR SETUP
# ============================================================================
def setup_blob_detector():
    """
    Create and configure an OpenCV SimpleBlobDetector for laser spot detection.

    The blob detector is an alternative to our centroid-based approach. It looks
    for circular bright spots in a thresholded image. The parameters below filter
    detections by color, size, shape (circularity), convexity, and elongation
    (inertia ratio) to ensure we only detect round, bright spots — i.e., lasers.

    Returns:
        cv2.SimpleBlobDetector: A configured blob detector instance.
    """
    params = cv2.SimpleBlobDetector_Params()

    # --- Color Filter ---
    # blobColor = 255 means we're looking for WHITE blobs (bright laser spots)
    # in the thresholded image.
    params.filterByColor = True
    params.blobColor = 255

    # --- Area Filter ---
    # Only detect blobs between 50 and 5000 pixels in area.
    # This filters out tiny noise specks (< 50 px) and huge bright regions (> 5000 px).
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 5000

    # --- Circularity Filter ---
    # Circularity = 4*pi*Area / Perimeter^2. A perfect circle = 1.0.
    # We require at least 0.5, meaning somewhat circular shapes are accepted.
    # This helps exclude irregular bright patches that aren't laser spots.
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # --- Convexity Filter ---
    # Convexity = Area / Area_of_convex_hull. A convex shape = 1.0.
    # We require 0.8+, filtering out concave/irregular shapes.
    params.filterByConvexity = True
    params.minConvexity = 0.8

    # --- Inertia Ratio Filter ---
    # Inertia ratio measures how elongated a shape is. A circle = 1.0, a line = 0.0.
    # We require 0.4+, so moderately elongated shapes are still accepted.
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    # OpenCV API changed between v2 and v3 — use the correct constructor.
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


# Initialize the blob detector once at module load time so it's ready to use.
blob_detector = setup_blob_detector()


# ============================================================================
# LABVIEW COMMAND LISTENER (runs in a background thread)
# ============================================================================
def listen_for_commands():
    """
    Listen for mode-switch commands from LabVIEW over UDP.

    This function runs in a daemon thread (background) and blocks waiting for
    incoming UDP packets on UDP_RECV_PORT. Each packet contains a single ASCII
    digit that tells us which mode to switch to:

        "0" → Mode 0: Laser detection (centroid algorithm)
        "1" → Mode 1: Pupil tracking (via Pupil Labs data)
        "2" → Mode 2: Algorithm speed test (both algorithms, with timing)
        "3" → Mode 3: Start calibration (begin collecting pupil positions)
        "4" → Mode 4: End calibration (compute offset from collected data)
        "5" → Mode 5: Reset calibration (clear offsets to zero)

    The global `current_mode` variable is updated under a thread lock to
    prevent race conditions with the main processing loop.
    """
    global current_mode, offset_x, offset_y, calibration_buffer

    # Create a UDP socket and bind it to listen on the configured port.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LABVIEW_IP, UDP_RECV_PORT))
    logger.info(f"👂 Command Listener Active on {LABVIEW_IP}:{UDP_RECV_PORT}")

    while True:
        try:
            # Block until a UDP packet arrives (up to 1024 bytes).
            data, addr = sock.recvfrom(1024)
            command = data.decode('utf-8').strip()

            # Acquire the lock before modifying shared state.
            with lock:
                if command == "1":
                    # Switch to pupil tracking mode.
                    current_mode = 1
                    logger.info("🔄 Mode → 1 (Pupil Tracking)")

                elif command == "0":
                    # Switch to laser detection mode (centroid algorithm).
                    current_mode = 0
                    logger.info("🔄 Mode → 0 (Laser Detection — Centroid)")

                elif command == "2":
                    # Switch to algorithm speed test mode.
                    current_mode = 2
                    logger.info("🔄 Mode → 2 (Algorithm Speed Test)")

                elif command == "3":
                    # Begin calibration: clear old samples and start collecting.
                    current_mode = 3
                    calibration_buffer = []
                    logger.info("🔄 Mode → 3 (Calibration — Collecting Samples)")

                elif command == "4":
                    # Finalize calibration: compute the average pupil position
                    # from the collected samples and store it as the offset.
                    if len(calibration_buffer) > 10:
                        offset_x = np.mean([p[0] for p in calibration_buffer])
                        offset_y = np.mean([p[1] for p in calibration_buffer])
                        logger.info(
                            f"✅ Calibration complete — offset=({offset_x:.5f}, {offset_y:.5f}) "
                            f"from {len(calibration_buffer)} samples"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Calibration had only {len(calibration_buffer)} samples "
                            f"(need >10). Offset NOT updated."
                        )
                    current_mode = 1  # Return to pupil tracking after calibration.

                elif command == "5":
                    # Reset calibration: clear offsets back to zero.
                    offset_x, offset_y = 0.0, 0.0
                    current_mode = 1
                    logger.info("🔄 Calibration RESET — offsets cleared to (0, 0)")

                else:
                    logger.warning(f"⚠️ Unknown command received: '{command}' from {addr}")

        except Exception as e:
            logger.error(f"❌ Command listener error: {e}", exc_info=True)


# ============================================================================
# LASER DETECTION — ALGORITHM A: CENTROID (Moment-Based)
# ============================================================================
def detect_laser_centroid(frame, width, height):
    """
    Detect a laser spot using image moments (centroid calculation).

    Algorithm steps:
      1. Apply Gaussian blur to reduce noise.
      2. Apply a brightness threshold — only very bright pixels (> LASER_THRESHOLD)
         survive. This isolates the laser spot from the darker background.
      3. Compute the image moments of the thresholded image. The moment M["m00"]
         is the total white area. If it's non-zero, the centroid (center of mass)
         is at (M["m10"]/M["m00"], M["m01"]/M["m00"]).
      4. Normalize the centroid to [0, 1] range by dividing by frame dimensions.

    Args:
        frame (np.ndarray): Grayscale eye camera frame.
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.

    Returns:
        tuple: (norm_x, norm_y) — normalized laser position, or (-1, -1) if not found.
    """
    try:
        # Step 1: Blur the image to smooth out high-frequency noise.
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)

        # Step 2: Threshold — keep only pixels brighter than LASER_THRESHOLD.
        _, thresh = cv2.threshold(blurred, LASER_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Step 3: Compute image moments to find the center of the bright region.
        M = cv2.moments(thresh)

        if M["m00"] != 0:
            # Centroid coordinates, normalized to [0, 1].
            return (M["m10"] / M["m00"]) / width, (M["m01"] / M["m00"]) / height

        # No bright region found — laser not detected in this frame.
        return -1, -1

    except Exception as e:
        logger.error(f"Centroid detection error: {e}")
        return -1, -1


# ============================================================================
# LASER DETECTION — ALGORITHM B: OPENCV BLOB DETECTOR
# ============================================================================
def detect_laser_blob(frame, width, height):
    """
    Detect a laser spot using OpenCV's SimpleBlobDetector.

    Algorithm steps:
      1. Apply Gaussian blur to reduce noise (same as centroid approach).
      2. Apply a brightness threshold to isolate the laser spot.
      3. Feed the thresholded image into the pre-configured blob detector.
         The detector finds contiguous bright regions that match our shape/size
         criteria (circularity, convexity, area, inertia — set in setup_blob_detector).
      4. If multiple blobs are found, pick the LARGEST one (most likely the laser).
      5. Normalize the blob center to [0, 1] range.

    Args:
        frame (np.ndarray): Grayscale eye camera frame.
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.

    Returns:
        tuple: (norm_x, norm_y) — normalized laser position, or (-1, -1) if not found.
    """
    try:
        # Step 1: Blur to reduce noise.
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)

        # Step 2: Threshold — isolate bright pixels.
        _, thresh = cv2.threshold(blurred, LASER_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Step 3: Run the blob detector on the thresholded image.
        keypoints = blob_detector.detect(thresh)

        if keypoints:
            # Step 4: Select the largest blob (by diameter) as the laser spot.
            kp = max(keypoints, key=lambda k: k.size)

            # Step 5: Normalize the keypoint center to [0, 1] range.
            return kp.pt[0] / width, kp.pt[1] / height

        # No blobs matched our criteria — laser not detected.
        return -1, -1

    except Exception as e:
        logger.error(f"Blob detection error: {e}")
        return -1, -1


# ============================================================================
# CALIBRATION OFFSET APPLICATION
# ============================================================================
def apply_offset(x, y):
    """
    Subtract calibration offsets from raw pupil coordinates.

    During calibration (Mode 3/4), we compute the average pupil position
    while the user looks at a known reference point. This average becomes
    the "home" offset. By subtracting it, subsequent readings are relative
    to that reference, effectively defining a new (0,0) origin.

    Args:
        x (float): Raw normalized x-coordinate from Pupil Labs.
        y (float): Raw normalized y-coordinate from Pupil Labs.

    Returns:
        tuple: (adjusted_x, adjusted_y) with offsets subtracted.
    """
    return x - offset_x, y - offset_y


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
def main():
    """
    Main entry point. Sets up connections to Pupil Service and LabVIEW,
    then enters an infinite loop that processes frames and pupil data
    based on the current operating mode.
    """
    global current_mode, calibration_buffer

    # ------------------------------------------------------------------
    # LOG SESSION HEADER — record system info at the start of every run
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("BRIDGE CODE — SESSION START")
    logger.info("=" * 70)
    logger.info(f"Timestamp       : {datetime.now().isoformat()}")
    logger.info(f"Python Version  : {sys.version}")
    logger.info(f"OpenCV Version  : {cv2.__version__}")
    logger.info(f"NumPy Version   : {np.__version__}")
    logger.info(f"LabVIEW IP      : {LABVIEW_IP}")
    logger.info(f"UDP Send Port   : {UDP_SEND_PORT}")
    logger.info(f"UDP Recv Port   : {UDP_RECV_PORT}")
    logger.info(f"Eye Camera ID   : {EYE_ID}")
    logger.info(f"Laser Threshold : {LASER_THRESHOLD}")
    logger.info(f"Blur Kernel     : {BLUR_KERNEL}")
    logger.info(f"Log File        : {log_filename}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # START BACKGROUND THREAD — listens for LabVIEW commands
    # ------------------------------------------------------------------
    threading.Thread(target=listen_for_commands, daemon=True).start()

    # ------------------------------------------------------------------
    # CONNECT TO PUPIL SERVICE via ZMQ
    # ------------------------------------------------------------------
    # Pupil Service exposes a REQ/REP socket on port 50020. We send
    # "SUB_PORT" to ask which port to subscribe to for real-time data.
    context = zmq.Context()
    req = context.socket(zmq.REQ)
    req.connect("tcp://127.0.0.1:50020")
    req.send_string("SUB_PORT")
    sub_port = req.recv_string()
    logger.info(f"📡 Connected to Pupil Service — SUB port: {sub_port}")

    # Subscribe to two ZMQ topics:
    #   "pupil.{EYE_ID}.2d" — 2D pupil detection results (norm_pos, confidence)
    #   "frame.eye.{EYE_ID}" — raw camera frames (for laser detection)
    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{sub_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, f"pupil.{EYE_ID}.2d")
    sub.setsockopt_string(zmq.SUBSCRIBE, f"frame.eye.{EYE_ID}")
    logger.info(f"📬 Subscribed to topics: pupil.{EYE_ID}.2d, frame.eye.{EYE_ID}")

    # ------------------------------------------------------------------
    # CREATE UDP SOCKET — for sending data to LabVIEW
    # ------------------------------------------------------------------
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_addr = (LABVIEW_IP, UDP_SEND_PORT)
    logger.info(f"📤 UDP output → {LABVIEW_IP}:{UDP_SEND_PORT}")

    # Track the latest pupil position across frames (used in Mode 2 reporting).
    latest_pupil = (-1, -1)

    # Frame counter for periodic logging (avoid flooding the log).
    frame_count = 0

    logger.info("🚀 Entering main processing loop (Mode 1 — Pupil Tracking)")

    # ------------------------------------------------------------------
    # INFINITE PROCESSING LOOP
    # ------------------------------------------------------------------
    while True:
        try:
            # Receive a ZMQ message: [topic, payload]
            # topic = which data stream this message is from
            # payload = msgpack-encoded dictionary with the actual data
            topic_bytes, payload = sub.recv_multipart()
            topic = topic_bytes.decode('utf-8')
            msg = msgpack.loads(payload, raw=False)

            # Read the current mode under the lock to avoid race conditions.
            with lock:
                active_mode = current_mode

            # ==============================================================
            # ALWAYS: Update latest pupil position if this is a pupil message
            # with sufficient confidence (> 0.6 = reasonably certain detection).
            # ==============================================================
            if "pupil" in topic and msg['confidence'] > 0.6:
                latest_pupil = msg['norm_pos']

            # ==============================================================
            # MODE 3: CALIBRATION — Collect pupil position samples
            # ==============================================================
            if active_mode == 3 and "pupil" in topic and msg['confidence'] > 0.6:
                calibration_buffer.append(msg['norm_pos'])
                # Log every 50th sample to track progress without flooding.
                if len(calibration_buffer) % 50 == 0:
                    logger.debug(
                        f"Calibration: {len(calibration_buffer)} samples collected"
                    )

            # ==============================================================
            # MODE 1: PUPIL TRACKING — Forward pupil position to LabVIEW
            # ==============================================================
            elif active_mode == 1 and "pupil" in topic and msg['confidence'] > 0.6:
                # Apply calibration offset, then send adjusted coordinates.
                fx, fy = apply_offset(*msg['norm_pos'])
                udp_sock.sendto(f"{fx:.5f},{fy:.5f}".encode('utf-8'), udp_addr)
                frame_count += 1

                # Log every 100th pupil sample.
                if frame_count % 100 == 0:
                    logger.debug(
                        f"Mode 1 | Pupil sent: ({fx:.5f}, {fy:.5f}) | "
                        f"frame #{frame_count}"
                    )

            # ==============================================================
            # MODE 0: LASER DETECTION (Centroid) — detect laser & send to LV
            # ==============================================================
            elif active_mode == 0 and "frame" in topic:
                # Extract the image from the ZMQ message.
                fmt = msg.get('format', 'raw')
                img_data = msg.get('img', msg.get('buffer', None))

                if img_data:
                    w, h = msg['width'], msg['height']

                    # Decode the frame based on its format.
                    if 'jpg' in fmt or 'jpeg' in fmt:
                        # JPEG compressed — decode to grayscale.
                        frame = cv2.imdecode(
                            np.frombuffer(img_data, np.uint8),
                            cv2.IMREAD_GRAYSCALE
                        )
                    else:
                        # Raw bytes — reshape directly into a 2D array.
                        frame = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w)

                    # Run centroid-based laser detection.
                    lx, ly = detect_laser_centroid(frame, w, h)

                    if lx != -1:
                        udp_sock.sendto(
                            f"{lx:.5f},{ly:.5f}".encode('utf-8'), udp_addr
                        )
                        frame_count += 1
                        if frame_count % 100 == 0:
                            logger.debug(
                                f"Mode 0 | Laser centroid: ({lx:.5f}, {ly:.5f})"
                            )

            # ==============================================================
            # MODE 2: ALGORITHM SPEED TEST — time both algorithms per frame
            # ==============================================================
            elif active_mode == 2 and "frame" in topic:
                fmt = msg.get('format', 'raw')
                img_data = msg.get('img', msg.get('buffer', None))

                if img_data:
                    w, h = msg['width'], msg['height']

                    # Decode the frame (same logic as Mode 0).
                    if 'jpg' in fmt or 'jpeg' in fmt:
                        frame = cv2.imdecode(
                            np.frombuffer(img_data, np.uint8),
                            cv2.IMREAD_GRAYSCALE
                        )
                    else:
                        frame = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w)

                    # ----- TIME ALGORITHM A: CENTROID -----
                    t0 = time.perf_counter()
                    cx, cy = detect_laser_centroid(frame, w, h)
                    t1 = time.perf_counter()
                    time_centroid = (t1 - t0) * 1000  # Convert seconds → ms

                    # ----- TIME ALGORITHM B: BLOB DETECTOR -----
                    t2 = time.perf_counter()
                    bx, by = detect_laser_blob(frame, w, h)
                    t3 = time.perf_counter()
                    time_blob = (t3 - t2) * 1000  # Convert seconds → ms

                    # Get the most recent pupil position for comparison.
                    px, py = latest_pupil

                    # Log the speed and coordinate comparison.
                    logger.info(
                        f"⏱️ SPEED | Centroid: {time_centroid:.2f}ms | "
                        f"Blob: {time_blob:.2f}ms"
                    )
                    logger.info(
                        f"🎯 COORD | Centroid: ({cx:.3f},{cy:.3f}) | "
                        f"Blob: ({bx:.3f},{by:.3f}) | "
                        f"Pupil: ({px:.3f},{py:.3f})"
                    )

                    # Send all six values to LabVIEW in one packet:
                    # centroid_x, centroid_y, blob_x, blob_y, pupil_x, pupil_y
                    msg_str = (
                        f"{cx:.5f},{cy:.5f},{bx:.5f},{by:.5f},{px:.5f},{py:.5f}"
                    )
                    udp_sock.sendto(msg_str.encode('utf-8'), udp_addr)

        except KeyboardInterrupt:
            # Ctrl+C pressed — exit cleanly.
            logger.info("🛑 KeyboardInterrupt — shutting down.")
            break

        except Exception as e:
            # Log any unexpected errors instead of silently ignoring them.
            logger.error(f"❌ Main loop error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("BRIDGE CODE — SESSION END")
    logger.info(f"Total frames processed: {frame_count}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()