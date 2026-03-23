# -*- coding: utf-8 -*-
"""
Algorithm Analysis: Centroid vs. OpenCV Blob Detection
======================================================

PURPOSE:
  Rigorous, side-by-side comparison of the two laser-spot detection algorithms
  used in the Pupil Tracking project:

    ALGORITHM A — CENTROID (image-moment based)
    ALGORITHM B — OpenCV SimpleBlobDetector

  For every frame that contains a laser spot, both algorithms are run and their
  speed, precision, and accuracy are measured.

METRICS MEASURED:
  ┌───────────┬──────────────────────────────────────────────────────────────┐
  │ Metric    │ How It Is Computed                                          │
  ├───────────┼──────────────────────────────────────────────────────────────┤
  │ SPEED     │ time.perf_counter() around each algorithm call per frame.   │
  │           │ Reported as mean, median, std, min, max in milliseconds.    │
  ├───────────┼──────────────────────────────────────────────────────────────┤
  │ PRECISION │ Standard deviation of detected (x, y) positions over N      │
  │           │ frames on a STATIC laser spot. Lower std = more precise.    │
  ├───────────┼──────────────────────────────────────────────────────────────┤
  │ ACCURACY  │ If a ground-truth center is supplied, Euclidean distance    │
  │           │ (RMSE) from detected positions to ground truth. If not      │
  │           │ supplied, inter-algorithm agreement is reported instead      │
  │           │ (how far apart the two algorithms' detections are).         │
  └───────────┴──────────────────────────────────────────────────────────────┘

DATA SOURCES (choose one via command-line arguments):
  --live     Connect to Pupil Service via ZMQ and analyze frames in real time.
  --video    Load a pre-recorded video file for offline, repeatable analysis.

OUTPUT:
  1. Console + log file: formatted summary table
  2. CSV file in logs/: per-frame raw data for further analysis
  3. (Optional) Matplotlib plots: speed histograms, position scatter plots

USAGE EXAMPLES:
  python algorithm_analysis.py --live --frames 500
  python algorithm_analysis.py --video recording.mp4 --frames 1000
  python algorithm_analysis.py --live --frames 300 --plot

REQUIREMENTS:
  - Python packages: opencv-python, numpy, zmq, msgpack
  - matplotlib (optional, only needed if --plot is used)
  - For --live mode: Pupil Capture/Service must be running on localhost
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse     # Command-line argument parsing
import csv          # Writing per-frame results to CSV files
import logging      # Structured logging to file and console
import os           # File system operations
import sys          # System info for session header
import time         # High-precision timing with perf_counter
from datetime import datetime  # Timestamps for log/CSV filenames

import cv2          # OpenCV — image processing and blob detection
import numpy as np  # NumPy — statistical calculations (mean, std, etc.)

# ============================================================================
# LOGGING SETUP (shared with bridge_code.py and direct_hardware.py)
# ============================================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(LOG_DIR, f"analysis_{timestamp_str}.log")

logger = logging.getLogger("analysis")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================================
# LASER DETECTION PARAMETERS (same as bridge_code.py for fair comparison)
# ============================================================================
LASER_THRESHOLD = 200   # Brightness cutoff — only very bright pixels survive
BLUR_KERNEL = (9, 9)    # Gaussian blur kernel size for noise reduction


# ============================================================================
# BLOB DETECTOR SETUP
# ============================================================================
def setup_blob_detector():
    """
    Create and configure an OpenCV SimpleBlobDetector.
    Settings are identical to those in bridge_code.py to ensure a fair comparison.

    Returns:
        cv2.SimpleBlobDetector: Configured detector instance.
    """
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 255          # Detect white blobs (bright laser spots)

    params.filterByArea = True
    params.minArea = 50             # Ignore tiny noise specks
    params.maxArea = 5000           # Ignore huge bright regions

    params.filterByCircularity = True
    params.minCircularity = 0.5     # Must be somewhat circular

    params.filterByConvexity = True
    params.minConvexity = 0.8       # Must be mostly convex

    params.filterByInertia = True
    params.minInertiaRatio = 0.4    # Must not be too elongated

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


blob_detector = setup_blob_detector()


# ============================================================================
# DETECTION ALGORITHMS (identical to bridge_code.py)
# ============================================================================
def detect_laser_centroid(frame, width, height):
    """
    Algorithm A: Detect laser spot using image moments (centroid).

    Steps: Blur → Threshold → Moments → Centroid.
    Returns normalized (x, y) in [0,1] or (-1, -1) if not found.
    """
    try:
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        _, thresh = cv2.threshold(blurred, LASER_THRESHOLD, 255, cv2.THRESH_BINARY)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            return (M["m10"] / M["m00"]) / width, (M["m01"] / M["m00"]) / height
        return -1, -1
    except Exception as e:
        logger.error(f"Centroid detection error: {e}")
        return -1, -1


def detect_laser_blob(frame, width, height):
    """
    Algorithm B: Detect laser spot using OpenCV SimpleBlobDetector.

    Steps: Blur → Threshold → Blob detection → Select largest blob.
    Returns normalized (x, y) in [0,1] or (-1, -1) if not found.
    """
    try:
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        _, thresh = cv2.threshold(blurred, LASER_THRESHOLD, 255, cv2.THRESH_BINARY)
        keypoints = blob_detector.detect(thresh)
        if keypoints:
            kp = max(keypoints, key=lambda k: k.size)
            return kp.pt[0] / width, kp.pt[1] / height
        return -1, -1
    except Exception as e:
        logger.error(f"Blob detection error: {e}")
        return -1, -1


# ============================================================================
# FRAME SOURCE: LIVE ZMQ (from Pupil Service)
# ============================================================================
def frames_from_zmq(eye_id=1, max_frames=500):
    """
    Generator that yields grayscale frames from Pupil Service over ZMQ.

    Connects to Pupil Service on localhost, subscribes to the eye camera's
    frame topic, and yields one grayscale numpy array per frame.

    Args:
        eye_id (int): Which eye camera to subscribe to (0 or 1).
        max_frames (int): Maximum number of frames to yield before stopping.

    Yields:
        tuple: (frame, width, height) — grayscale frame and its dimensions.
    """
    import zmq
    import msgpack

    logger.info("Connecting to Pupil Service via ZMQ...")

    context = zmq.Context()
    req = context.socket(zmq.REQ)
    req.connect("tcp://127.0.0.1:50020")
    req.send_string("SUB_PORT")
    sub_port = req.recv_string()

    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{sub_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, f"frame.eye.{eye_id}")
    logger.info(f"Subscribed to frame.eye.{eye_id} on port {sub_port}")

    count = 0
    while count < max_frames:
        topic_bytes, payload = sub.recv_multipart()
        msg = msgpack.loads(payload, raw=False)

        fmt = msg.get('format', 'raw')
        img_data = msg.get('img', msg.get('buffer', None))

        if img_data:
            w, h = msg['width'], msg['height']
            if 'jpg' in fmt or 'jpeg' in fmt:
                frame = cv2.imdecode(
                    np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE
                )
            else:
                frame = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w)

            count += 1
            yield frame, w, h

    sub.close()
    req.close()
    context.term()


# ============================================================================
# FRAME SOURCE: VIDEO FILE (offline analysis)
# ============================================================================
def frames_from_video(video_path, max_frames=500):
    """
    Generator that yields grayscale frames from a video file.

    Args:
        video_path (str): Path to the video file (e.g., .mp4, .avi).
        max_frames (int): Maximum number of frames to yield.

    Yields:
        tuple: (frame, width, height) — grayscale frame and its dimensions.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video opened: {video_path} ({w}x{h}, {total} total frames)")

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
        yield gray, w, h

    cap.release()


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================
def run_analysis(frame_source, ground_truth=None):
    """
    Run both detection algorithms on every frame from the source and collect
    timing and position data.

    Args:
        frame_source: A generator yielding (frame, width, height) tuples.
        ground_truth: Optional (x, y) tuple — known laser position for RMSE.

    Returns:
        dict: Results containing arrays of per-frame measurements.
    """
    # Storage arrays for per-frame data.
    centroid_times = []     # ms per frame for Algorithm A
    blob_times = []         # ms per frame for Algorithm B
    centroid_positions = [] # (x, y) per frame for Algorithm A
    blob_positions = []     # (x, y) per frame for Algorithm B
    agreements = []         # Euclidean distance between the two algorithms

    frame_num = 0

    for frame, w, h in frame_source:
        frame_num += 1

        # ----- Time Algorithm A: Centroid -----
        t0 = time.perf_counter()
        cx, cy = detect_laser_centroid(frame, w, h)
        t1 = time.perf_counter()
        centroid_ms = (t1 - t0) * 1000

        # ----- Time Algorithm B: Blob -----
        t2 = time.perf_counter()
        bx, by = detect_laser_blob(frame, w, h)
        t3 = time.perf_counter()
        blob_ms = (t3 - t2) * 1000

        # Store timing data regardless of detection success.
        centroid_times.append(centroid_ms)
        blob_times.append(blob_ms)

        # Only store positions when both algorithms detected something.
        centroid_detected = (cx != -1 and cy != -1)
        blob_detected = (bx != -1 and by != -1)

        if centroid_detected:
            centroid_positions.append((cx, cy))
        if blob_detected:
            blob_positions.append((bx, by))

        # Compute inter-algorithm agreement (Euclidean distance between detections).
        if centroid_detected and blob_detected:
            dist = np.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
            agreements.append(dist)

        # Log progress every 50 frames.
        if frame_num % 50 == 0:
            logger.debug(
                f"Frame {frame_num} | "
                f"Centroid: {centroid_ms:.2f}ms ({cx:.4f},{cy:.4f}) | "
                f"Blob: {blob_ms:.2f}ms ({bx:.4f},{by:.4f})"
            )

    logger.info(f"Analysis complete — {frame_num} frames processed.")

    return {
        "total_frames": frame_num,
        "centroid_times": np.array(centroid_times),
        "blob_times": np.array(blob_times),
        "centroid_positions": np.array(centroid_positions) if centroid_positions else np.array([]),
        "blob_positions": np.array(blob_positions) if blob_positions else np.array([]),
        "agreements": np.array(agreements) if agreements else np.array([]),
        "ground_truth": ground_truth,
    }


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================
def compute_stats(arr, label=""):
    """
    Compute summary statistics for an array of values.

    Args:
        arr (np.ndarray): 1D array of measurements.
        label (str): Human-readable label for logging.

    Returns:
        dict: mean, median, std, min, max — or all zeros if array is empty.
    """
    if len(arr) == 0:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def print_report(results):
    """
    Print a formatted comparison report to both console and log file.

    Args:
        results (dict): Output from run_analysis().
    """
    ct = compute_stats(results["centroid_times"])
    bt = compute_stats(results["blob_times"])

    # ── HEADER ──
    logger.info("")
    logger.info("=" * 72)
    logger.info("  ALGORITHM COMPARISON REPORT — Centroid vs. Blob Detection")
    logger.info("=" * 72)
    logger.info(f"  Total frames analyzed: {results['total_frames']}")
    logger.info("")

    # ── SPEED COMPARISON ──
    logger.info("  ┌─── SPEED (milliseconds per frame) ──────────────────────┐")
    logger.info(f"  │ {'Metric':<10} │ {'Centroid':>10} │ {'Blob':>10} │ {'Winner':>10} │")
    logger.info(f"  │{'─' * 10}──┼{'─' * 10}──┼{'─' * 10}──┼{'─' * 10}──│")

    for metric in ["mean", "median", "std", "min", "max"]:
        c_val = ct[metric]
        b_val = bt[metric]
        if metric in ("mean", "median", "min", "max"):
            winner = "Centroid" if c_val < b_val else "Blob" if b_val < c_val else "Tie"
        else:
            winner = "—"
        logger.info(
            f"  │ {metric:<10} │ {c_val:>9.3f}  │ {b_val:>9.3f}  │ {winner:>10} │"
        )
    logger.info(f"  └{'─' * 55}┘")
    logger.info("")

    # ── DETECTION RATE ──
    c_detected = len(results["centroid_positions"])
    b_detected = len(results["blob_positions"])
    total = results["total_frames"]
    logger.info("  ┌─── DETECTION RATE ──────────────────────────────────────┐")
    logger.info(
        f"  │ Centroid detected laser in {c_detected}/{total} frames "
        f"({100 * c_detected / total:.1f}%)"
    )
    logger.info(
        f"  │ Blob     detected laser in {b_detected}/{total} frames "
        f"({100 * b_detected / total:.1f}%)"
    )
    logger.info(f"  └{'─' * 55}┘")
    logger.info("")

    # ── PRECISION (standard deviation of positions) ──
    logger.info("  ┌─── PRECISION (lower std = more consistent) ─────────────┐")
    if len(results["centroid_positions"]) > 1:
        c_std_x = np.std(results["centroid_positions"][:, 0])
        c_std_y = np.std(results["centroid_positions"][:, 1])
        logger.info(f"  │ Centroid  σ_x = {c_std_x:.6f}   σ_y = {c_std_y:.6f}")
    else:
        logger.info("  │ Centroid  Not enough detections for precision analysis")

    if len(results["blob_positions"]) > 1:
        b_std_x = np.std(results["blob_positions"][:, 0])
        b_std_y = np.std(results["blob_positions"][:, 1])
        logger.info(f"  │ Blob      σ_x = {b_std_x:.6f}   σ_y = {b_std_y:.6f}")
    else:
        logger.info("  │ Blob      Not enough detections for precision analysis")
    logger.info(f"  └{'─' * 55}┘")
    logger.info("")

    # ── ACCURACY (RMSE to ground truth or inter-algorithm agreement) ──
    logger.info("  ┌─── ACCURACY ────────────────────────────────────────────┐")
    gt = results["ground_truth"]
    if gt is not None:
        # Compute RMSE to known ground truth for each algorithm.
        if len(results["centroid_positions"]) > 0:
            c_errors = np.sqrt(
                (results["centroid_positions"][:, 0] - gt[0]) ** 2 +
                (results["centroid_positions"][:, 1] - gt[1]) ** 2
            )
            logger.info(f"  │ Centroid RMSE to ground truth: {np.sqrt(np.mean(c_errors**2)):.6f}")
        if len(results["blob_positions"]) > 0:
            b_errors = np.sqrt(
                (results["blob_positions"][:, 0] - gt[0]) ** 2 +
                (results["blob_positions"][:, 1] - gt[1]) ** 2
            )
            logger.info(f"  │ Blob     RMSE to ground truth: {np.sqrt(np.mean(b_errors**2)):.6f}")
    else:
        logger.info("  │ No ground truth provided — showing inter-algorithm agreement:")
        if len(results["agreements"]) > 0:
            ag = compute_stats(results["agreements"])
            logger.info(
                f"  │ Mean distance between algorithms: {ag['mean']:.6f} "
                f"(std: {ag['std']:.6f})"
            )
        else:
            logger.info("  │ No frames where both algorithms detected — cannot compute.")
    logger.info(f"  └{'─' * 55}┘")
    logger.info("=" * 72)


def save_csv(results):
    """
    Save per-frame raw data to a CSV file in the logs/ directory.

    The CSV contains one row per frame with columns:
      frame, centroid_ms, blob_ms, centroid_x, centroid_y, blob_x, blob_y

    Args:
        results (dict): Output from run_analysis().

    Returns:
        str: Path to the created CSV file.
    """
    csv_path = os.path.join(LOG_DIR, f"analysis_results_{timestamp_str}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "centroid_ms", "blob_ms",
            "centroid_x", "centroid_y", "blob_x", "blob_y"
        ])

        c_pos_idx = 0  # Index into centroid_positions
        b_pos_idx = 0  # Index into blob_positions

        for i in range(results["total_frames"]):
            c_ms = results["centroid_times"][i]
            b_ms = results["blob_times"][i]

            # Match positions by index (only stored when detected).
            cx, cy = -1, -1
            bx, by = -1, -1

            if c_pos_idx < len(results["centroid_positions"]):
                cx = results["centroid_positions"][c_pos_idx][0]
                cy = results["centroid_positions"][c_pos_idx][1]
                c_pos_idx += 1

            if b_pos_idx < len(results["blob_positions"]):
                bx = results["blob_positions"][b_pos_idx][0]
                by = results["blob_positions"][b_pos_idx][1]
                b_pos_idx += 1

            writer.writerow([
                i + 1,
                f"{c_ms:.4f}", f"{b_ms:.4f}",
                f"{cx:.6f}", f"{cy:.6f}",
                f"{bx:.6f}", f"{by:.6f}"
            ])

    logger.info(f"📄 CSV saved: {csv_path}")
    return csv_path


def generate_plots(results):
    """
    Generate matplotlib visualizations of the analysis results.

    Creates two plots saved as PNG files in logs/:
      1. Speed histogram — distribution of algorithm execution times
      2. Position scatter — detected laser positions for each algorithm

    Args:
        results (dict): Output from run_analysis().
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation.")
        logger.warning("Install with: pip install matplotlib")
        return

    # ── PLOT 1: Speed Histogram ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(results["centroid_times"], bins=50, alpha=0.7,
                 label="Centroid", color="#2196F3")
    axes[0].hist(results["blob_times"], bins=50, alpha=0.7,
                 label="Blob", color="#FF5722")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Speed Distribution: Centroid vs Blob")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── PLOT 2: Position Scatter ──
    if len(results["centroid_positions"]) > 0:
        axes[1].scatter(
            results["centroid_positions"][:, 0],
            results["centroid_positions"][:, 1],
            alpha=0.4, s=10, label="Centroid", color="#2196F3"
        )
    if len(results["blob_positions"]) > 0:
        axes[1].scatter(
            results["blob_positions"][:, 0],
            results["blob_positions"][:, 1],
            alpha=0.4, s=10, label="Blob", color="#FF5722"
        )
    axes[1].set_xlabel("Normalized X")
    axes[1].set_ylabel("Normalized Y")
    axes[1].set_title("Detected Laser Positions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, f"analysis_plots_{timestamp_str}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"📊 Plots saved: {plot_path}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================
def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze Centroid vs. Blob laser detection algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python algorithm_analysis.py --live --frames 500
  python algorithm_analysis.py --video recording.mp4 --frames 1000
  python algorithm_analysis.py --live --frames 300 --plot
  python algorithm_analysis.py --video test.avi --gt 0.5 0.5 --plot
        """
    )

    # Source selection (mutually exclusive).
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--live", action="store_true",
        help="Use live frames from Pupil Service via ZMQ"
    )
    source.add_argument(
        "--video", type=str,
        help="Path to a video file for offline analysis"
    )

    # Analysis parameters.
    parser.add_argument(
        "--frames", type=int, default=500,
        help="Number of frames to analyze (default: 500)"
    )
    parser.add_argument(
        "--eye", type=int, default=1, choices=[0, 1],
        help="Eye camera ID for live mode (default: 1)"
    )
    parser.add_argument(
        "--gt", type=float, nargs=2, metavar=("X", "Y"),
        help="Ground truth laser position (normalized 0-1) for RMSE accuracy"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib plots (requires matplotlib)"
    )

    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """
    Main function: parse args, set up frame source, run analysis, report results.
    """
    args = parse_args()

    # ── SESSION HEADER ──
    logger.info("=" * 72)
    logger.info("  ALGORITHM ANALYSIS — SESSION START")
    logger.info("=" * 72)
    logger.info(f"  Timestamp      : {datetime.now().isoformat()}")
    logger.info(f"  Python Version : {sys.version}")
    logger.info(f"  OpenCV Version : {cv2.__version__}")
    logger.info(f"  NumPy Version  : {np.__version__}")
    logger.info(f"  Source         : {'LIVE (ZMQ)' if args.live else f'VIDEO ({args.video})'}")
    logger.info(f"  Frames         : {args.frames}")
    logger.info(f"  Ground Truth   : {args.gt if args.gt else 'Not provided'}")
    logger.info(f"  Log File       : {log_filename}")
    logger.info("=" * 72)

    # ── SELECT FRAME SOURCE ──
    if args.live:
        frame_source = frames_from_zmq(eye_id=args.eye, max_frames=args.frames)
    else:
        if not os.path.isfile(args.video):
            logger.error(f"Video file not found: {args.video}")
            sys.exit(1)
        frame_source = frames_from_video(args.video, max_frames=args.frames)

    # ── RUN ANALYSIS ──
    ground_truth = tuple(args.gt) if args.gt else None
    results = run_analysis(frame_source, ground_truth=ground_truth)

    if results["total_frames"] == 0:
        logger.error("No frames were processed. Check your source.")
        sys.exit(1)

    # ── REPORT RESULTS ──
    print_report(results)
    csv_path = save_csv(results)

    if args.plot:
        generate_plots(results)

    # ── SESSION FOOTER ──
    logger.info("")
    logger.info("=" * 72)
    logger.info("  ALGORITHM ANALYSIS — SESSION END")
    logger.info(f"  Results saved to: {csv_path}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
