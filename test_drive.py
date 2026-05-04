import cv2
import time
import board
import busio
import threading
from dataclasses import dataclass, field
from flask import Flask, Response, render_template_string
from adafruit_pca9685 import PCA9685
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = './best_rknn_model'

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CENTER_X = CAMERA_WIDTH / 2

# Flask config
HOST_IP = '0.0.0.0'
HOST_PORT = 5000

# ============================================================
# TUNING
# ============================================================
ROI_VERTICAL_CUTOFF = 0.65
Kp = 0.0007
Kd = 0.0009
BASE_SPEED = 0.18
LANE_WIDTH_PIXELS = 450

# Stop sign / redline logic
STOP_DURATION = 2.0
STOP_COOLDOWN = 5.0
STOP_THRESHOLD_Y = CAMERA_HEIGHT * 0.8

# Motor physics
MIN_MOTOR_POWER = 0.07
MAX_STEER = 0.7

# Performance tuning for Radxa Rock 5C 2 GB RAM
DETECTION_INTERVAL = 0.03       # minimum time between YOLO predictions
CONTROL_INTERVAL = 0.01         # motor update interval
STREAM_INTERVAL = 0.04          # about 25 FPS max streaming
CAMERA_BUFFER_SIZE = 1          # keep freshest frame only
YOLO_IMGSZ = 640                # lower than 640 for speed/RAM; use 320 for more speed, 640 for more accuracy
YOLO_CONF = 0.5

# Safety timeout: if detections are old, slow/stop instead of driving blind
DETECTION_TIMEOUT = 0.50
NO_LANE_SPEED = 0.06            # crawl speed when lane is temporarily missing; set to 0.0 to stop

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


frame_lock = threading.Lock()
detection_lock = threading.Lock()
stream_lock = threading.Lock()
shutdown_event = threading.Event()

latest_raw_frame = None
latest_frame_id = 0
latest_stream_frame = None
latest_detection = DetectionState(last_update=time.time())

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
# CAMERA THREAD
# Only grabs frames. Does not run YOLO. Does not control motors.
# ============================================================
def camera_loop():
    global latest_raw_frame, latest_frame_id

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

    if not cap.isOpened():
        print("Camera Error: Cannot open camera")
        shutdown_event.set()
        return

    print("Camera thread started")

    try:
        while not shutdown_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            # Flip once here so every other thread receives corrected camera view
            frame = cv2.flip(frame, 1)

            with frame_lock:
                latest_raw_frame = frame
                latest_frame_id += 1

    except Exception as e:
        print(f"Camera Loop Error: {e}")
        shutdown_event.set()
    finally:
        cap.release()
        print("Camera thread ended")


# ============================================================
# VISION / DETECTION THREAD
# Only runs YOLO and calculates lane target/stop request.
# Does not directly touch motors.
# ============================================================
def detection_loop():
    global latest_detection, latest_stream_frame

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Detection thread started")

    last_processed_frame_id = -1
    prev_error = 0.0
    last_inference_time = time.time()
    fps_smooth = 0.0

    try:
        while not shutdown_event.is_set():
            # Limit inference rate so the 2 GB board does not get overloaded
            now = time.time()
            if now - last_inference_time < DETECTION_INTERVAL:
                time.sleep(0.002)
                continue

            with frame_lock:
                if latest_raw_frame is None:
                    time.sleep(0.005)
                    continue
                frame = latest_raw_frame.copy()
                frame_id = latest_frame_id

            # Skip if no new frame has arrived
            if frame_id == last_processed_frame_id:
                time.sleep(0.002)
                continue

            last_processed_frame_id = frame_id
            start_time = time.time()

            # Run YOLO on the freshest frame
            results = model.predict(
                source=frame,
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                verbose=False
            )
            result = results[0]
            boxes = result.boxes

            # ------------------------------------------------------------
            # Vision processing
            # ------------------------------------------------------------
            best_y_x = None
            best_w_x = None
            max_y_area = 0
            max_w_area = 0
            stop_requested = False
            current_time = time.time()
            cutoff_pixel = CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF

            for box in boxes:
                cls = model.names[int(box.cls[0])]
                x, y, w, h = box.xywh[0].tolist()

                # Stop line detection
                #if cls == 'redline' and y > STOP_THRESHOLD_Y:
                    #stop_requested = True

                # Ignore lane detections too high in the frame
                if y < cutoff_pixel:
                    continue

                area = w * h
                if cls == 'yellowline' and area > max_y_area:
                    max_y_area = area
                    best_y_x = x
                elif cls == 'whiteline' and area > max_w_area:
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

            # PID steering calculation
            error = target_x - CENTER_X
            derivative = error - prev_error
            prev_error = error
            steering = (error * Kp) + (derivative * Kd)
            steering = max(min(steering, MAX_STEER), -MAX_STEER)

            # FPS smoothing
            inference_dt = max(time.time() - start_time, 1e-6)
            instant_fps = 1.0 / inference_dt
            fps_smooth = instant_fps if fps_smooth == 0.0 else (0.85 * fps_smooth + 0.15 * instant_fps)

            new_detection = DetectionState(
                best_y_x=best_y_x,
                best_w_x=best_w_x,
                target_x=target_x,
                error=error,
                steering=steering,
                stop_requested=stop_requested,
                last_update=current_time,
                frame_id=frame_id,
                fps=fps_smooth
            )

            with detection_lock:
                latest_detection = new_detection

            # ------------------------------------------------------------
            # Video/debug frame for Flask only
            # ------------------------------------------------------------
            annotated_frame = result.plot()
            draw_debug_overlay(annotated_frame, new_detection)

            with stream_lock:
                latest_stream_frame = annotated_frame

            last_inference_time = time.time()

    except Exception as e:
        print(f"Detection Loop Error: {e}")
        shutdown_event.set()
    finally:
        print("Detection thread ended")


# ============================================================
# DRIVE / MOTOR CONTROL THREAD
# Only controls motors using the latest DetectionState.
# This keeps the robot responsive even if YOLO has a slower frame rate.
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

    last_stop_time = 0.0
    print("Drive thread started")

    try:
        while not shutdown_event.is_set():
            with detection_lock:
                det = latest_detection

            now = time.time()
            detection_age = now - det.last_update

            # Safety: if detection is too old, stop or crawl slowly
            if detection_age > DETECTION_TIMEOUT:
                set_drive(0.0, 0.0)
                time.sleep(CONTROL_INTERVAL)
                continue

            # Stop line logic handled here, separate from detection logic
            if det.stop_requested and (now - last_stop_time) > STOP_COOLDOWN:
                print("!!! STOPPING FOR RED LINE !!!")
                stop_all()
                time.sleep(STOP_DURATION)
                last_stop_time = time.time()
                continue

            # If no lane was found, crawl straight or stop depending on NO_LANE_SPEED
            lane_seen = det.best_y_x is not None or det.best_w_x is not None
            drive_speed = BASE_SPEED if lane_seen else NO_LANE_SPEED

            set_drive(drive_speed, det.steering)
            time.sleep(CONTROL_INTERVAL)

    except Exception as e:
        print(f"Drive Loop Error: {e}")
        shutdown_event.set()
    finally:
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
        cv2.putText(frame, line, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    debug_y = int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF) + 20
    cv2.circle(frame, (int(det.target_x), debug_y), 10, (0, 255, 0), -1)
    cv2.line(frame, (int(CENTER_X), 0), (int(CENTER_X), CAMERA_HEIGHT), (255, 255, 255), 1)
    cv2.line(frame, (0, int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF)),
             (CAMERA_WIDTH, int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF)), (255, 255, 0), 1)

    if det.stop_requested:
        cv2.putText(frame, "RED LINE DETECTED", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


# ============================================================
# FLASK STREAMING
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

        flag, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
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
            body { background: #111; color: #eee; text-align: center; font-family: monospace; }
            img { border: 2px solid #555; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>RADXA ROBOT - SEPARATED CONTROL / DETECTION</h1>
        <p>Camera thread + detection thread + drive thread + Flask stream</p>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </body>
    </html>
    """)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("Starting separated robot system...")

    threads = [
        threading.Thread(target=camera_loop, daemon=True, name="camera_thread"),
        threading.Thread(target=detection_loop, daemon=True, name="detection_thread"),
        threading.Thread(target=drive_loop, daemon=True, name="drive_thread"),
    ]

    for t in threads:
        t.start()

    print(f"Starting Web Server at http://{HOST_IP}:{HOST_PORT}")

    try:
        app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        shutdown_event.set()
        time.sleep(0.5)
        print("Program ended")
