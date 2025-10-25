

#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2

import mediapipe as mp
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
  if len(sys.argv) < 2:
    print("Usage: python poseture.py <camera_index>")
    return 
  main2()

def main2():
      # STEP 2: Create a PoseLandmarker object.
  base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
  options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)

  # Open the camera stream.
  cap = cv2.VideoCapture(int(sys.argv[1]))
  if not cap.isOpened():
    print("Error: Could not open camera.")
    return

  while True:
    ret, frame = cap.read()
    if not ret:
      print("Error: Could not read frame.")
      break

    # Convert the frame to RGB as Mediapipe expects RGB images.
    rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


    # Detect pose landmarks from the camera frame.
    detection_result = detector.detect(image)

    # Visualize the detection result.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Exit the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release the camera and close all OpenCV windows.
  cap.release()
  cv2.destroyAllWindows()

def tutorial_onlystillpicture():
    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='lib/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file("lib/pic/girl.jpg")

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imshow("hi", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2.imshow("hi", visualized_mask)
    cv2.waitKey(0)
    # img = cv2.imread("girl.jpg")
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

if __name__ == "__main__":
    main()
