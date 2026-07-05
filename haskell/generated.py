import cv2
import mediapipe as mp
import subprocess
import sys
import time
import os

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Find paths
haskell_dir = os.path.dirname(os.path.abspath(__file__))
exe_name = "angle_calc.exe" if sys.platform == "win32" else "angle_calc"
exe_path = os.path.join(haskell_dir, exe_name)
hs_path = os.path.join(haskell_dir, "angle_calc.hs")

# Compile angle_calc.hs if the executable doesn't exist
if not os.path.exists(exe_path) and os.path.exists(hs_path):
    print("Compiling Haskell angle_calc utility...")
    try:
        subprocess.run(["ghc", "-O2", hs_path], cwd=haskell_dir, check=True)
        print("Haskell utility compiled successfully.")
    except Exception as e:
        print(f"Warning: Could not compile Haskell program: {e}")
        print("Will attempt to run via runghc dynamically.")

# Start Haskell process
try:
    if os.path.exists(exe_path):
        haskell_proc = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            cwd=haskell_dir
        )
    else:
        haskell_proc = subprocess.Popen(
            ["runghc", "angle_calc.hs"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=haskell_dir
        )
except Exception as e:
    print(f"Error starting Haskell subprocess: {e}")
    print("Please install GHC (Haskell Platform) or ensure 'runghc' is available.")
    sys.exit(1)

def compute_angle_from_landmarks(landmarks, idx1, idx2, idx3):
    # Extract normalized x,y (MediaPipe gives [0,1])
    p1 = (landmarks[idx1].x, landmarks[idx1].y)
    p2 = (landmarks[idx2].x, landmarks[idx2].y)
    p3 = (landmarks[idx3].x, landmarks[idx3].y)
    
    # Send to Haskell via stdin
    line = f"{p1[0]} {p1[1]} {p2[0]} {p2[1]} {p3[0]} {p3[1]}"
    try:
        haskell_proc.stdin.write(line + '\n')
        haskell_proc.stdin.flush()
        
        # Read result from stdout
        angle_str = haskell_proc.stdout.readline().strip()
        if angle_str:
            return float(angle_str)
    except Exception as e:
        print(f"Subprocess communication error: {e}")
    return None  # Error case

# Camera loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Example: Angle at elbow (indices from MediaPipe: 11=left shoulder, 13=left elbow, 15=left wrist)
        angle = compute_angle_from_landmarks(landmarks, 11, 13, 15)
        if angle is not None:
            print(f"Inner angle: {angle} radians ({angle * 180 / 3.14159:.1f} degrees)")
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
try:
    haskell_proc.stdin.close()
    haskell_proc.wait()  # Ensure Haskell exits cleanly
except Exception:
    pass