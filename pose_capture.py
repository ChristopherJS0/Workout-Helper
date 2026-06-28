import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python import vision
from cv2.gapi.streaming import timestamp

# Getting angle of left bicep to a horizontal line.
def calcLeftBicepAngle(landmarks, frame):
    h, w, _ = frame.shape

    # left arm landmarks section
    if landmarks[vision.PoseLandmark.LEFT_ELBOW] and landmarks[vision.PoseLandmark.LEFT_SHOULDER]:
        left_elbow = landmarks[vision.PoseLandmark.LEFT_ELBOW]
        left_shoulder = landmarks[vision.PoseLandmark.LEFT_SHOULDER]
        
        # Get actual positions of points 
        lsx, lsy = int(left_shoulder.x * w), int(left_shoulder.y * h)
        lex, ley = int(left_elbow.x * w), int(left_elbow.y * h)
        
        dx = lex - lsx
        dy = ley - lsy

        angle_rad = np.arctan2(dy,dx)
        angle_deg = np.degrees(angle_rad)
        ref_point_x = lsx + 100 
        ref_point_y = lsy
        
        # HORIZONTAL LINE ON SHOULDER
        cv2.line(frame, (lsx, lsy), (ref_point_x, ref_point_y), (0, 255, 0), 2)
                
        # Draw the Arm Vector (Shoulder to Elbow) (Blue)
        cv2.line(frame, (lsx, lsy), (lex, ley), (255, 0, 0), 2)
        
        # Draw the Angle Text near the Shoulder
        # Format: "Angle: -45.2 deg"
        label = f"{angle_deg:.1f} deg"
        cv2.putText(frame, label, (lsx + 10, lsy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Optional: Draw circles at joints for clarity
        cv2.circle(frame, (lsx, lsy), 5, (0, 255, 0), -1) # Shoulder
        cv2.circle(frame, (lex, ley), 5, (255, 0, 0), -1) # Elbow
        print(f"Left Bicep Angle: {angle_deg:.2f} degrees.")

# Getting angle of right bicep to a horizontal line.
def calcRightBicepAngle(landmarks, frame):
       h, w, _ = frame.shape

       if landmarks[vision.PoseLandmark.RIGHT_ELBOW] and landmarks[vision.PoseLandmark.RIGHT_SHOULDER]:
        right_elbow = landmarks[vision.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = landmarks[vision.PoseLandmark.RIGHT_SHOULDER]

        # Get actual positions of points
        rsx, rsy = int(right_shoulder.x * w), int(right_shoulder.y * h)
        rex, rey = int(right_elbow.x * w), int(right_elbow.y * h)
        
        dx = rex - rsx
        dy = rey - rsy

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        ref_point_x = rsx + 100 
        ref_point_y = rsy

        # HORIZONTAL LINE ON SHOULDER
        cv2.line(frame, (rsx, rsy), (ref_point_x, ref_point_y), (0, 255, 0), 2)
        
        # Arm Vector (Shoulder to Elbow) (Blue)
        cv2.line(frame, (rsx, rsy), (rex, rey), (255, 0, 0), 2)
        
        # Angle Text (Yellow)
        label = f"{angle_deg:.1f} deg"
        cv2.putText(frame, label, (rsx + 10, rsy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Joint Circles
        cv2.circle(frame, (rsx, rsy), 5, (0, 255, 0), -1) # Shoulder
        cv2.circle(frame, (rex, rey), 5, (255, 0, 0), -1) # Elbow
        print(f"Right Bicep Angle: {angle_deg:.2f} degrees.")

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

    # Controller For Pose Tracking
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
                calcLeftBicepAngle(pose_landmarks, frame)
                calcRightBicepAngle(pose_landmarks, frame)

            # Display the annotated image
            cv2.imshow('Pose Landmarker', frame)

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
