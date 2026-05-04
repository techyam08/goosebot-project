import cv2
import time
import board
import busio
import threading
from dataclasses import dataclass
from flask import Flask, Response, render_template_string
from adafruit_pca9685 import PCA9685
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = './best_goose_rknn_model'

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CENTER_X = CAMERA_WIDTH / 2

HOST_IP = '0.0.0.0'
HOST_PORT = 5000

# ============================================================
# TUNING
# ============================================================
ROI_VERTICAL_CUTOFF = 0.65
Kp = 0.0007
Kd = 0.0009
BASE_SPEED = 0.23
LANE_WIDTH_PIXELS = 450

# Motor physics
MIN_MOTOR_POWER = 0.07
MAX_STEER = 0.75

# Performance tuning
DETECTION_INTERVAL = 0.03
CONTROL_INTERVAL = 0.01
STREAM_INTERVAL = 0.04
CAMERA_BUFFER_SIZE = 1
YOLO_IMGSZ = 640
YOLO_CONF = 0.20

# Safety timeout
DETECTION_TIMEOUT = 0.50
NO_LANE_SPEED = 0.1

# ============================================================
# SHARED STATE
# ============================================================
@dataclass
class DetectionState:
    best_y_x: float | None = None
    best_w_x: float | None = None
    target_x: float = CENTER_X
    error: float = 0.0
    steering: float = 0.0
    stop_requested: bool = False
    last_update: float = 0.0
    frame_id: int = 0
    fps: float = 0.0


detection_lock = threading.Lock()
stream_lock = threading.Lock()
shutdown_event = threading.Event()

latest_detection = DetectionState(last_update=time.time())
latest_stream_frame = None

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

# ============================================================
# MOTOR CLASS
# ============================================================
class Motor:
    def __init__(self, pca, in1, in2):
        self.pca = pca
        self.in1 = pca.channels[in1]
        self.in2 = pca.channels[in2]

    def set_speed(self, speed):
        if abs(speed) < 0.01:
            pwm = 0
        else:
            abs_s = abs(speed)
            mapped_speed = MIN_MOTOR_POWER + (abs_s * (1.0 - MIN_MOTOR_POWER))
            pwm = int(min(mapped_speed, 1.0) * 65535)

        if speed > 0:
            self.in1.duty_cycle = pwm
            self.in2.duty_cycle = 0
        elif speed < 0:
            self.in1.duty_cycle = 0
            self.in2.duty_cycle = pwm
        else:
            self.stop()

    def stop(self):
        self.in1.duty_cycle = 0
        self.in2.duty_cycle = 0


# ============================================================
# VISION + WEB STREAM FRAME THREAD
# This one thread handles:
# 1. Camera capture
# 2. YOLO detection
# 3. Steering calculation
# 4. Updating the frame used by the web stream
# ============================================================
def vision_stream_loop():
    global latest_detection, latest_stream_frame

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

    if not cap.isOpened():
        print("Camera Error: Cannot open camera")
        shutdown_event.set()
        return

    print("Vision/Web stream thread started")

    prev_error = 0.0
    frame_id = 0
    last_inference_time = 0.0
    fps_smooth = 0.0

    try:
        while not shutdown_event.is_set():
            now = time.time()

            # Limit YOLO rate so the Radxa does not get overloaded
            if now - last_inference_time < DETECTION_INTERVAL:
                time.sleep(0.002)
                continue

            ok, frame = cap.read()

            if not ok or frame is None:
                time.sleep(0.005)
                continue

            frame_id += 1

            # Flip once so everything uses the corrected camera view
            frame = cv2.flip(frame, 1)

            start_time = time.time()

            # Run YOLO
            results = model.predict(
                source=frame,
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            best_y_x = None
            best_w_x = None
            max_y_area = 0
            max_w_area = 0
            stop_requested = False

            cutoff_pixel = CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF

            for box in boxes:
                cls = model.names[int(box.cls[0])]
                x, y, w, h = box.xywh[0].tolist()

                # Redline detection removed/disabled
                # if cls == 'redline':
                #     stop_requested = True

                # Ignore detections too high in the frame
                if y < cutoff_pixel:
                    continue

                area = w * h

                if cls == 'yellow line' and area > max_y_area:
                    max_y_area = area
                    best_y_x = x

                elif cls == 'white line' and area > max_w_area:
                    max_w_area = area
                    best_w_x = x

            # Target selection
            if best_y_x is not None and best_w_x is not None:
                target_x = (best_y_x + best_w_x) / 2

            elif best_y_x is not None:
                target_x = best_y_x + (LANE_WIDTH_PIXELS / 2)

            elif best_w_x is not None:
                target_x = best_w_x - (LANE_WIDTH_PIXELS / 2)

            else:
                target_x = CENTER_X

            # PD steering
            error = target_x - CENTER_X
            derivative = error - prev_error
            prev_error = error

            steering = (error * Kp) + (derivative * Kd)
            steering = max(min(steering, MAX_STEER), -MAX_STEER)

            inference_dt = max(time.time() - start_time, 1e-6)
            instant_fps = 1.0 / inference_dt

            if fps_smooth == 0.0:
                fps_smooth = instant_fps
            else:
                fps_smooth = 0.85 * fps_smooth + 0.15 * instant_fps

            new_detection = DetectionState(
                best_y_x=best_y_x,
                best_w_x=best_w_x,
                target_x=target_x,
                error=error,
                steering=steering,
                stop_requested=stop_requested,
                last_update=time.time(),
                frame_id=frame_id,
                fps=fps_smooth
            )

            with detection_lock:
                latest_detection = new_detection

            # Create annotated frame for web stream
            annotated_frame = result.plot()
            draw_debug_overlay(annotated_frame, new_detection)

            with stream_lock:
                latest_stream_frame = annotated_frame

            last_inference_time = time.time()

    except Exception as e:
        print(f"Vision/Web Stream Loop Error: {e}")
        shutdown_event.set()

    finally:
        cap.release()
        print("Vision/Web stream thread ended")


# ============================================================
# DRIVE / MOTOR CONTROL THREAD
# This thread only controls motors using the latest DetectionState.
# ============================================================
def drive_loop():
    print("Initializing motor hardware...")

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 100

        left_motors = [Motor(pca, 7, 6), Motor(pca, 4, 5)]
        right_motors = [Motor(pca, 2, 3), Motor(pca, 1, 0)]

    except Exception as e:
        print(f"Hardware Init Error: {e}")
        shutdown_event.set()
        return

    def set_drive(fwd, steer):
        steer = max(min(steer, MAX_STEER), -MAX_STEER)

        left = fwd + steer
        right = fwd - steer

        max_val = max(abs(left), abs(right))

        if max_val > 1.0:
            left /= max_val
            right /= max_val

        for m in left_motors:
            m.set_speed(left)

        for m in right_motors:
            m.set_speed(right)

    def stop_all():
        for m in left_motors + right_motors:
            m.stop()

    print("Drive thread started")

    try:
        while not shutdown_event.is_set():
            with detection_lock:
                det = latest_detection

            now = time.time()
            detection_age = now - det.last_update

            # If detections are stale, stop the robot
            if detection_age > DETECTION_TIMEOUT:
                set_drive(0.0, 0.0)
                time.sleep(CONTROL_INTERVAL)
                continue

            # If no lane was detected, crawl forward or stop
            lane_seen = det.best_y_x is not None or det.best_w_x is not None

            if lane_seen:
                drive_speed = BASE_SPEED
            else:
                drive_speed = NO_LANE_SPEED

            set_drive(drive_speed, det.steering)

            time.sleep(CONTROL_INTERVAL)

    except Exception as e:
        print(f"Drive Loop Error: {e}")
        shutdown_event.set()

    finally:
        print("Stopping all motors...")
        stop_all()

        try:
            pca.deinit()
        except Exception:
            pass

        print("Drive thread ended")


# ============================================================
# DEBUG DRAWING
# ============================================================
def draw_debug_overlay(frame, det: DetectionState):
    debug_lines = [
        f"best_w_x: {det.best_w_x}",
        f"best_y_x: {det.best_y_x}",
        f"target_x: {det.target_x:.1f}",
        f"CENTER_X: {CENTER_X:.1f}",
        f"error: {det.error:.1f}",
        f"steering: {det.steering:.4f}",
        f"BASE_SPEED: {BASE_SPEED:.2f}",
        f"YOLO FPS: {det.fps:.1f}",
        f"frame_id: {det.frame_id}",
    ]

    x0, y0 = 10, 25

    for i, line in enumerate(debug_lines):
        y = y0 + i * 22

        cv2.putText(
            frame,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    debug_y = int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF) + 20

    cv2.circle(
        frame,
        (int(det.target_x), debug_y),
        10,
        (0, 255, 0),
        -1
    )

    cv2.line(
        frame,
        (int(CENTER_X), 0),
        (int(CENTER_X), CAMERA_HEIGHT),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (0, int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF)),
        (CAMERA_WIDTH, int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF)),
        (255, 255, 0),
        1
    )


# ============================================================
# FLASK STREAMING
# This only serves the latest processed frame.
# Detection is NOT done here.
# ============================================================
def generate_frames():
    while not shutdown_event.is_set():
        with stream_lock:
            if latest_stream_frame is None:
                frame = None
            else:
                frame = latest_stream_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        flag, encoded_image = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )

        if not flag:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded_image) +
            b'\r\n'
        )

        time.sleep(STREAM_INTERVAL)


@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Robot Vision</title>
        <style>
            body {
                background: #111;
                color: #eee;
                text-align: center;
                font-family: monospace;
            }

            img {
                border: 2px solid #555;
                margin-top: 20px;
            }
        </style>
    </head>

    <body>
        <h1>RADXA ROBOT - VISION + STREAM THREAD / MOTOR THREAD</h1>
        <p>One thread handles camera, detection, and stream frame updates.</p>
        <p>One separate thread handles motor control.</p>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </body>
    </html>
    """)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("Starting robot system...")

    vision_thread = threading.Thread(
        target=vision_stream_loop,
        daemon=True,
        name="vision_stream_thread"
    )

    motor_thread = threading.Thread(
        target=drive_loop,
        daemon=True,
        name="motor_control_thread"
    )

    vision_thread.start()
    motor_thread.start()

    print(f"Starting Web Server at http://{HOST_IP}:{HOST_PORT}")

    try:
        app.run(
            host=HOST_IP,
            port=HOST_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )

    except KeyboardInterrupt:
        print("Ctrl + C detected. Shutting down...")

    finally:
        shutdown_event.set()

        vision_thread.join(timeout=2.0)
        motor_thread.join(timeout=2.0)

        print("Program ended")