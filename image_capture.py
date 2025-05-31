import os
import cv2
import subprocess
from datetime import datetime

# Change this to the name of the person you're photographing
PERSON_NAME = "jaryd"

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos_libcamera(name):
    folder = create_folder(name)
    
    photo_count = 0
    
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")
    
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
    
    # Use libcamera-vid to create a live preview stream
    preview_process = subprocess.Popen([
        "libcamera-vid", 
        "--inline", 
        "--nopreview", 
        "-t", "0", 
        "-o", "-", 
        "--width", "640", 
        "--height", "480", 
        "--codec", "yuv420"
    ], stdout=subprocess.PIPE)

    while True:
        # Read raw YUV frames (this is more of a conceptual placeholder; libcamera-vid doesn't stream directly as images)
        # Instead, let’s leverage a simpler approach: use libcamera-still to capture one image at a time
        # Here’s a lightweight approach with libcamera-still to take snapshots
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            
            # Capture the image using libcamera-still
            subprocess.run([
                "libcamera-still", 
                "-o", filepath, 
                "--width", "640", 
                "--height", "480",
                "--nopreview"
            ])
            print(f"Photo {photo_count} saved: {filepath}")

            # Display the captured photo
            image = cv2.imread(filepath)
            cv2.imshow('Preview', image)

        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

if __name__ == "__main__":
    capture_photos_libcamera(PERSON_NAME)
