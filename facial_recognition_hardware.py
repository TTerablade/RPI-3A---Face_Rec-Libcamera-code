import face_recognition
import cv2
import numpy as np
import time
import pickle
from gpiozero import LED

# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize GPIO
output = LED(14)

# Initialize our variables
cv_scaler = 4  # Scaling factor to improve performance
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# List of authorized names (case-sensitive!)
authorized_names = ["john", "alice", "bob"]

def process_frame(frame):
    global face_locations, face_encodings, face_names

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))

    # Convert to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces & encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model='large')

    face_names = []
    authorized_face_detected = False

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if face_distances.size > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name in authorized_names:
                    authorized_face_detected = True
        face_names.append(name)

    # GPIO control
    if authorized_face_detected:
        output.on()
    else:
        output.off()

    return frame

def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

def main():
    # Use OpenCV's VideoCapture to interface with libcamera (via V4L2)
    # On Raspberry Pi, libcamera provides /dev/video0 or /dev/videoX
    print("[INFO] Starting video stream...")
    video_capture = cv2.VideoCapture(0)  # Adjust if necessary
    
    # Set camera resolution (libcamera fallback to default if unsupported)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        # Process frame
        processed_frame = process_frame(frame)
        display_frame = draw_results(processed_frame)

        # FPS
        current_fps = calculate_fps()
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    output.off()
    print("[INFO] Video stream ended.")

if __name__ == "__main__":
    main()
