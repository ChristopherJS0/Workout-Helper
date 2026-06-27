import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python import vision
from cv2.gapi.streaming import timestamp

def calcAngles(landmarker, landmarks, frame):
    # Example function to calculate angles between specific landmarks
    # This is a placeholder; actual implementation will depend on the landmarks of interest
    # Check if EITHER L_Elbow + L_Shoulder + L_Wrist OR R_Elbow + R_Shoulder + R_Wrist are present

    h, w, _ = frame.shape
    print(f"Max values of height and width{h, w}")
    # left arm landmarks section - Angle Detection
    if landmarks[vision.PoseLandmark.LEFT_ELBOW] and landmarks[vision.PoseLandmark.LEFT_SHOULDER]:
        left_elbow = landmarks[vision.PoseLandmark.LEFT_ELBOW]
        left_shoulder = landmarks[vision.PoseLandmark.LEFT_SHOULDER]
        # Calculate angle between left shoulder, to left elbow, to horizontal line
        # Arm is ideally a straight line down
        # so an imaginary x axis line will be used to see if it's close to 90*

        left_elbow_pos = np.array([left_elbow.x * w,
                                   left_elbow.y * h,
                                   left_elbow.z * w
                                   ], dtype=np.float32)

        left_shoulder_pos = np.array([left_shoulder.x * w,
                                  left_shoulder.y * h,
                                  left_shoulder.z * w
                                  ], dtype=np.float32)
        
        lsx, lsy = int(left_shoulder_pos[0]), int(left_shoulder_pos[1])
        lex, ley = int(left_elbow_pos[0]), int(left_elbow_pos[1])
        print(f"Left Shoulder = {lsx, lsy}")
        print(f"Left Elbow = {lex, ley}")

    if landmarks[vision.PoseLandmark.RIGHT_ELBOW] and landmarks[vision.PoseLandmark.RIGHT_SHOULDER]:

        right_elbow = landmarks[vision.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = landmarks[vision.PoseLandmark.RIGHT_SHOULDER]

        right_elbow_pos = np.array([right_elbow.x * w,
                                   right_elbow.y * h,
                                   right_elbow.z * w
                                   ], dtype=np.float32)

        right_shoulder_pos = np.array([right_shoulder.x * w,
                                  left_shoulder.y * h,
                                  left_shoulder.z * w
                                  ], dtype=np.float32)
        
        rsx, rsy = int(right_shoulder_pos[0]), int(right_shoulder_pos[1])
        rex, rey = int(right_elbow_pos[0]), int(right_elbow_pos[1])
        print(f"Right Shoulder = {rsx, rsy}")
        print(f"Right Elbow = {rex, rey}")

def main():
    # Configuration
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options1 = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.VIDEO
    )

    # Video Feed
    # Cv2 is a python library that uses webcams or any camera linked to
    #   the computer and will be necessary for this project.
    cap = cv2.VideoCapture(0) # 0 is the first camera source cv2 can detect.
    landmarker = vision.PoseLandmarker.create_from_options(options1)

    # With: keyword for python that handles file management and closing files once 
    #   code block ends.
    # mp.Pose params: lower confidence means more times program will acknowledge 
    #   a body part even if it isn't highly confident. 50% is a good enough value.
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            # Calculate timestamp (ms) based on frame count or actual time
            # Assuming ~30 FPS for simplicity, or use time.time() * 1000
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            # Detect Pose (Use detect_for_video for VIDEO mode)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Convert back to BGR for OpenCV display
            image.flags.writeable = True
            display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # IMAGE ANNOTATION SEGMENT
            if results.pose_landmarks:
                # Create a copy to draw on (drawing_utils expects a writable numpy array)
                annotated_image = np.copy(display_image)
                
                for pose_landmarks in results.pose_landmarks:
                    drawing_utils.draw_landmarks(
                        image=annotated_image,
                        landmark_list=pose_landmarks,
                        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS, # Draws the skeleton lines
                        # Each dot will have a number
                        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(), # Default dots
                        connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2) # Custom line color/thickness
                    )
                
                display_image = annotated_image
                calcAngles(landmarker, pose_landmarks, frame)

            # Display the annotated image
            cv2.imshow('Pose Landmarker', display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        pass
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
