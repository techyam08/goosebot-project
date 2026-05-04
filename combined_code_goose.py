import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import time
import sys
import board
import busio
from adafruit_pca9685 import PCA9685
import tty
import termios
import select

# --- Configuration ---
# IMPORTANT: Use the exact model path you were using in your command.
# If your model file is named 'yolo11n_dts_rknn_model.rknn', use that.
# The YOLO class is smart and will load .pt, .rknn, etc.
MODEL_PATH = 'best_rknn_model' 

# Set the host IP to '0.0.0.0' to make it accessible on your network
HOST_IP = '0.0.0.0'
HOST_PORT = 5000
# ---------------------

app = Flask(__name__)

# Load your YOLOv11 RKNN model
try:
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the MODEL_PATH is correct and the model file exists.")
    exit()

def generate_frames():
    """
    Generator function to stream video frames with YOLO detection.
    """
    print("Starting prediction stream from source 0 (webcam)...")
    
    # Use stream=True for continuous video processing
    # show=False prevents Ultralytics from opening its own cv2 window
    try:
        results_generator = model(source=0, stream=True, show=False)
    except Exception as e:
        print(f"Error starting video stream (source=0): {e}")
        print("Is the camera connected and accessible?")
        return

    for r in results_generator:
        try:
            # .plot() is the easiest way to get the frame with boxes drawn
            annotated_frame = r.plot() 

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Failed to encode frame")
                continue

            # Convert to bytes and yield in multipart format
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error during processing/streaming: {e}")
            break

@app.route('/')
def index():
    """Video streaming home page."""
    # A simple HTML page to display the video feed
    html_page = """
    <html>
    <head>
        <title>YOLO RKNN Stream</title>
        <style>
            body { font-family: sans-serif; text-align: center; background-color: #222; color: white; }
            img { background-color: #000; border: 1px solid #555; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>YOLO RKNN Live Stream</h1>
        <h3>(Running on Radxa Rock 5C lite)</h3>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </body>
    </html>
    """
    return render_template_string(html_page)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    # Returns the generator function as a multipart response
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"Starting Flask server...")
    print(f"Access the stream in your browser at: http://<YOUR_ROCK_5C_IP>:{HOST_PORT}/")
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True)
    
# --- Motor and Robot Configuration ---
SPEED = 0.25
# Grace period in seconds to wait for another keypress before stopping
GRACE_PERIOD = 0.3

class Motor:
    """A class to control one motor via a PCA9685 PWM driver."""
    def __init__(self, pca, in1_channel, in2_channel):
        self.pca = pca
        self.in1 = pca.channels[in1_channel]
        self.in2 = pca.channels[in2_channel]

    def set_speed(self, speed):
        """Sets the motor speed and direction from -1.0 to 1.0."""
        pwm_value = int(abs(speed) * 65535)
        if pwm_value > 65535: pwm_value = 65535
        
        if speed > 0:
            self.in1.duty_cycle = pwm_value
            self.in2.duty_cycle = 0
        elif speed < 0:
            self.in1.duty_cycle = 0
            self.in2.duty_cycle = pwm_value
        else:
            self.stop()
            
    def stop(self):
        self.in1.duty_cycle = 0
        self.in2.duty_cycle = 0

# --- Robot Movement Functions ---
def move_forward():
    print("Forward ", end="\r")
    for motor in right_motors: motor.set_speed(SPEED)
    for motor in left_motors: motor.set_speed(SPEED)

def move_backward():
    print("Backward", end="\r")
    for motor in right_motors: motor.set_speed(-SPEED)
    for motor in left_motors: motor.set_speed(-SPEED)

def turn_left():
    print("Left    ", end="\r")
    for motor in right_motors: motor.set_speed(SPEED)
    for motor in left_motors: motor.set_speed(-SPEED)

def turn_right():
    print("Right   ", end="\r")
    for motor in right_motors: motor.set_speed(-SPEED)
    for motor in left_motors: motor.set_speed(SPEED)

def stop_all():
    print("Stopped ", end="\r")
    for motor in all_motors: motor.stop()

# --- Main Program ---
if __name__ == "__main__":
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 100
        
        motor_fl = Motor(pca, in1_channel=7, in2_channel=6)
        motor_fr = Motor(pca, in1_channel=2, in2_channel=3)
        motor_rl = Motor(pca, in1_channel=4, in2_channel=5)
        motor_rr = Motor(pca, in1_channel=1, in2_channel=0)
        
        all_motors = [motor_fl, motor_rl, motor_fr, motor_rr]
        right_motors = [motor_fr, motor_rr]
        left_motors = [motor_fl, motor_rl]
        
        print("All motors initialized.")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())
        
        print("Ready for input. Press WASD to move. Press x to exit.")
        
        # Start in a known state
        current_action = stop_all
        current_action()
        last_key_time = 0

        # Main control loop
        while True:
            # Default to the current action if nothing changes
            action = current_action

            # Check if a key is pressed
            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1)
                last_key_time = time.time()
                
                if key == 'w':
                    action = move_forward
                elif key == 's':
                    action = move_backward
                elif key == 'a':
                    action = turn_left
                elif key == 'd':
                    action = turn_right
                elif key == 'x':
                    print("\nExiting...")
                    break
                else:
                    action = stop_all
            
            # If no key has been pressed for the duration of the grace period, stop.
            elif time.time() - last_key_time > GRACE_PERIOD:
                action = stop_all
            
            # Only send a new command to the motors if the state has changed.
            if action is not current_action:
                action()
                current_action = action
        

    except KeyboardInterrupt:
        print("\nExiting...")
        
    finally:
        stop_all()
        termios.tcgetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nProgram finished.")

    


