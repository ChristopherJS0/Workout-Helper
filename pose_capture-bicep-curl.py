import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python import vision

MIN_VISIBILITY  = 0.6

# ----------- CALIBRATION SETTINGS ----------- #
ARM_STRAIGHT_TARGET = 90.0      # degrees; arm hanging straight down reads ~90 deg from horizontal
ARM_STRAIGHT_TOLERANCE = 12.0   # +/- degrees still counted as "straight down"
FOREARM_STRAIGHT_TOLERANCE = 25.0  # degrees; forearm hanging straight down reads ~90 deg from horizontal
STILLNESS_THRESHOLD_PX = 20     # max shoulder drift (px) between frames to count as "still"
CALIBRATION_HOLD_TIME = 4.0     # seconds the pose must be held (within the requested 3-5s range)
AWAY_TOLERANCE_TIME = 4         # time allowed from being away from the view of camera.
# --------------------------------------------- #

# ----------- CURL SETTINGS ----------- #
CURL_ANGLE_TARGET = 80.0      # degrees; forearm angle at the top of a curl (from horizontal)
CURL_ANGLE_TOLERANCE = 15.0    # +/- degrees still counted
CURL_HOLD_TIME = 1.0          # seconds the curl must be held to count as a rep
RESTING_HOLD_TIME = 2.0        # seconds the arm must be held in resting position before next curl can count
# -------------------------------------- #

# ----------- ANGLE CALCULATION FUNCTIONS ----------- #
# Each function draws only its own arm line/label (if draw=True) and returns
# the angle in degrees, or None if the relevant landmarks aren't visible.

def calcLeftBicepAngle(landmarks, frame, draw=True):
    h, w, _ = frame.shape

    if landmarks[vision.PoseLandmark.LEFT_ELBOW].visibility > MIN_VISIBILITY and landmarks[vision.PoseLandmark.LEFT_SHOULDER].visibility > MIN_VISIBILITY:
        left_elbow = landmarks[vision.PoseLandmark.LEFT_ELBOW]
        left_shoulder = landmarks[vision.PoseLandmark.LEFT_SHOULDER]

        lsx, lsy = int(left_shoulder.x * w), int(left_shoulder.y * h)
        lex, ley = int(left_elbow.x * w), int(left_elbow.y * h)

        dx = lex - lsx
        dy = ley - lsy

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        if draw:
            label = f"{angle_deg:.1f} deg"
            cv2.line(frame, (lsx, lsy), (lex, ley), (255, 0, 0), 2)
            cv2.putText(frame, label, (lsx + 10, lsy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return angle_deg
    return None

def calcRightBicepAngle(landmarks, frame, draw=True):
    h, w, _ = frame.shape

    if landmarks[vision.PoseLandmark.RIGHT_ELBOW].visibility > MIN_VISIBILITY and landmarks[vision.PoseLandmark.RIGHT_SHOULDER].visibility > MIN_VISIBILITY:
        right_elbow = landmarks[vision.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = landmarks[vision.PoseLandmark.RIGHT_SHOULDER]

        rsx, rsy = int(right_shoulder.x * w), int(right_shoulder.y * h)
        rex, rey = int(right_elbow.x * w), int(right_elbow.y * h)

        dx = rex - rsx
        dy = rey - rsy

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        if draw:
            label = f"{angle_deg:.1f} deg"
            cv2.line(frame, (rsx, rsy), (rex, rey), (255, 0, 0), 2)
            cv2.putText(frame, label, (rsx + 10, rsy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return angle_deg
    return None

def calcLeftForearmAngle(landmarks, frame, draw=True):
    h, w, _ = frame.shape

    if landmarks[vision.PoseLandmark.LEFT_ELBOW].visibility > MIN_VISIBILITY and landmarks[vision.PoseLandmark.LEFT_WRIST].visibility > MIN_VISIBILITY:
        left_elbow = landmarks[vision.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[vision.PoseLandmark.LEFT_WRIST]

        lex, ley = int(left_elbow.x * w), int(left_elbow.y * h)
        lwx, lwy = int(left_wrist.x * w), int(left_wrist.y * h)

        dx = lwx - lex
        dy = lwy - ley

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        if draw:
            label = f"{angle_deg:.1f} deg"
            cv2.line(frame, (lex, ley), (lwx, lwy), (255, 0, 0), 2)
            cv2.putText(frame, label, (lex + 10, ley - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return angle_deg
    return None

def calcRightForearmAngle(landmarks, frame, draw=True):
    h, w, _ = frame.shape

    if landmarks[vision.PoseLandmark.RIGHT_ELBOW].visibility > MIN_VISIBILITY and landmarks[vision.PoseLandmark.RIGHT_WRIST].visibility > MIN_VISIBILITY:
        right_elbow = landmarks[vision.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[vision.PoseLandmark.RIGHT_WRIST]

        rex, rey = int(right_elbow.x * w), int(right_elbow.y * h)
        rwx, rwy = int(right_wrist.x * w), int(right_wrist.y * h)

        dx = rwx - rex
        dy = rwy - rey

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        if draw:
            label = f"{angle_deg:.1f} deg"
            cv2.line(frame, (rex, rey), (rwx, rwy), (255, 0, 0), 2)
            cv2.putText(frame, label, (rex + 10, rey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return angle_deg
    return None
# ----------- end: ANGLE CALCULATION FUNCTIONS ----------- #


# ----------- CALIBRATION FUNCTIONS ----------- #
def isArmStraightDown(angle, part):
    """True if the angle is within tolerance of hanging straight down (~90 deg)."""
    # Returns Bool: True if the arm is considered straight down, False otherwise.
    if angle is None:
        return False
    if part == 'bicep':
        return abs(angle - ARM_STRAIGHT_TARGET) <= ARM_STRAIGHT_TOLERANCE
    elif part == 'forearm':
        return abs(angle - ARM_STRAIGHT_TARGET) <= FOREARM_STRAIGHT_TOLERANCE
    return False

def getSingleShoulderPos(landmarks, frame, side):
    """Pixel position of one shoulder ('left' or 'right'), used to check stillness."""
    h, w, _ = frame.shape
    lm = landmarks[vision.PoseLandmark.LEFT_SHOULDER if side == 'left' else vision.PoseLandmark.RIGHT_SHOULDER]
    if lm.visibility > MIN_VISIBILITY:
        return (lm.x * w, lm.y * h)
    return None

def checkArmReady(side, landmarks, frame, prev_pos):
    """
    Checks whether ONE arm (bicep + forearm) is hanging straight down and
    that shoulder hasn't drifted since the last frame (standing still).

    Returns (ready: bool, shoulder_pos: tuple|None)
    """
    if side == 'left':
        bicep = calcLeftBicepAngle(landmarks, frame, draw=True)
        forearm = calcLeftForearmAngle(landmarks, frame, draw=True)
    else:
        bicep = calcRightBicepAngle(landmarks, frame, draw=True)
        forearm = calcRightForearmAngle(landmarks, frame, draw=True)

    arm_straight = isArmStraightDown(bicep, 'bicep') and isArmStraightDown(forearm, 'forearm')
    shoulder_pos = getSingleShoulderPos(landmarks, frame, side)

    is_still = False
    if shoulder_pos is not None and prev_pos is not None:
        dist = np.hypot(shoulder_pos[0] - prev_pos[0], shoulder_pos[1] - prev_pos[1])
        is_still = dist <= STILLNESS_THRESHOLD_PX
    # First frame a shoulder becomes visible: no prior position to compare yet.

    return (arm_straight and is_still), shoulder_pos

def calibrateArm(calibration_start_time, calibrated_sides, side):
    if calibration_start_time[side] is None:
        calibration_start_time[side] = time.time()
    elapsed = time.time() - calibration_start_time[side] # Calc. time from when the arm was first detected as ready
    if elapsed >= CALIBRATION_HOLD_TIME:
        calibrated_sides[side] = True

def decalibrationTimer(away_start_time, side):
    ''' If the arm is away from the camera for too long, decalibrate it. Returns True if decalibration should occur. '''
    if away_start_time[side] == None:
        away_start_time[side] = time.time()
    timeAway = time.time() - away_start_time[side]
    if timeAway >= AWAY_TOLERANCE_TIME:
        return True
    return False
        

def resetCalibration(calibration_start_time, calibrated_sides, side):
    # That arm's pose broke or it moved: reset its hold timer.
    calibration_start_time[side] = None
    calibrated_sides[side] = False

# ----------- end: CALIBRATION FUNCTIONS ----------- #

# -------- CURL FUNCTIONS ----------- #
def isArmCurled(angle):
    """True if the forearm angle is within tolerance of curled (~80 deg)."""
    if angle is None:
        return False
    return abs(angle - CURL_ANGLE_TARGET) <= CURL_ANGLE_TOLERANCE

def isArmResting(side, landmarks, frame):
    """True if the forearm and bicep are hanging straight down (resting)."""
    if side == 'left':
        forearm_angle = calcLeftForearmAngle(landmarks, frame, draw=True)
        bicep_angle = calcLeftBicepAngle(landmarks, frame, draw=True)
    else:
        forearm_angle = calcRightForearmAngle(landmarks, frame, draw=True)
        bicep_angle = calcRightBicepAngle(landmarks, frame, draw=True)
    return isArmStraightDown(forearm_angle, 'forearm') and isArmStraightDown(bicep_angle, 'bicep')

def isProperForm(side, landmarks, frame):
    """True if the bicep is straight down and the forearm is curled."""
    if side == 'left':
        bicep_angle = calcLeftBicepAngle(landmarks, frame, draw=True)
        forearm_angle = calcLeftForearmAngle(landmarks, frame, draw=True)
    else:
        bicep_angle = calcRightBicepAngle(landmarks, frame, draw=True)
        forearm_angle = calcRightForearmAngle(landmarks, frame, draw=True)
    return isArmStraightDown(bicep_angle, 'bicep') and isArmCurled(forearm_angle)

def curlTimer(curling_start_time, curled_sides, curlCount, side):
    ''' Almost identical to calibrate arm, only this is meant to reinforce
    proper curl timing. Prevents quick curls.'''
    if curling_start_time[side] is None:
        curling_start_time[side] = time.time()
    elapsed = time.time() - curling_start_time[side] # Calc. time from when the arm was first detected as ready
    if elapsed >= CURL_HOLD_TIME:
        print(f"Curl completed for {side} arm! Total curls: {curlCount[side] + 1}")
        curled_sides[side] = True
        curlCount[side] += 1

def restingTimer(resting_start_time, curled_sides, side):
    ''' Almost identical to calibrate arm, only this is meant to reinforce
    proper resting timing. Prevents quick curls.'''
    if resting_start_time[side] is None:
        resting_start_time[side] = time.time()
    elapsed = time.time() - resting_start_time[side] # Calc. time from when the arm was first detected as ready
    if elapsed >= RESTING_HOLD_TIME:
        print(f"Resting period completed for {side} arm!")
        curled_sides[side] = False

# ----------- end: CURL FUNCTIONS ----------- #

# ----------- STATUS OVERLAY ----------- #
def drawStatus(frame, mode, calibrated_sides, calibration_progress):
    h, w, _ = frame.shape
    is_calibrated = calibrated_sides['left'] or calibrated_sides['right']

    if mode == 'calibration':
        if is_calibrated:
            if calibrated_sides['left'] and calibrated_sides['right']:
                which = "Both Arms"
            elif calibrated_sides['left']:
                which = "Left Arm"
            else:
                which = "Right Arm"
            text, color = f"Calibrated ({which})", (0, 255, 0)
        else:
            text, color = "Calibrating...", (0, 255, 255)

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if not is_calibrated:
            # Making the progress bar a bit more visible by drawing a white border around it
            bar_x, bar_y, bar_w, bar_h = 20, 55, 250, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
            fill_w = int(bar_w * min(max(calibration_progress, 0.0), 1.0))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 255), -1)

        cv2.putText(frame, "Mode: Calibration  |  'c' switch mode  |  'r' reset",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cal_text = "Calibrated" if is_calibrated else "Not Calibrated"
        cal_color = (0, 255, 0) if is_calibrated else (0, 0, 255)
        cv2.putText(frame, f"Mode: Regular  ({cal_text})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, cal_color, 2)
        cv2.putText(frame, "'c' switch to Calibration mode  |  'r' reset",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
# ----------- end: STATUS OVERLAY ----------- #

def main():
    # Configuration
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options1 = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.VIDEO
    )

    # Video Feed
    cap = cv2.VideoCapture(0)
    landmarker = vision.PoseLandmarker.create_from_options(options1)

    # ---- Calibration / mode state (tracked per-arm so one arm is enough) ---- #
    mode = 'calibration'            # 'calibration' or 'regular'
    prev_shoulder_pos = {'left': None, 'right': None}
    calibration_start_time = {'left': None, 'right': None}
    calibrated_sides = {'left': False, 'right': False}

    curling_start_time = {'left': None, 'right': None}
    resting_start_time = {'left': None, 'right': None}
    curled_sides = {'left': False, 'right': False}
    away_start_time = {'left': None, 'right': None}
    curlCount = {'left': 0, 'right': 0}
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            image.flags.writeable = True
            display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks[-1]

                if mode == 'calibration':
                    progresses = []
                    # Check each arm for straight-down pose and stillness, and update calibration state. 
                    # Should determine left and/or right arm calibration independently, so that if one arm is calibrated, the other can still be calibrated.
                    for side in ('left', 'right'):
                        ready, pos = checkArmReady(side, pose_landmarks, display_image, prev_shoulder_pos[side])
                        prev_shoulder_pos[side] = pos

                        if ready:
                            calibrateArm(calibration_start_time, calibrated_sides, side)
                        else:
                            # That arm's pose broke or it moved: reset its hold timer.
                            resetCalibration(calibration_start_time, calibrated_sides, side)

                        if calibration_start_time[side] is not None: # Add progress for this arm if it's currently being held still and straight down
                            progresses.append((time.time() - calibration_start_time[side]) / CALIBRATION_HOLD_TIME)
                        else: # Add 0 progress for this arm if it's not being held still and straight down
                            progresses.append(0.0)

                    progress = max(progresses)
                    drawStatus(display_image, mode, calibrated_sides, progress)

                    # Change here to switch to curling mode,
                    # note: make sure the callibrated sides are saved to send to
                    # curling mode. Curling mode will adjust to either 
                    # front facing or side view. 
                    if (calibrated_sides['left'] or calibrated_sides['right']):
                        mode = 'curling'
                        true_sides = [side for side, is_calibrated in calibrated_sides.items() if is_calibrated]

                elif mode == 'curling':
                    # Initiation: Set for either one sided or front sided.
                    # must break and return person isn't there OR switched sides.
                    if not (calibrated_sides['left'] or calibrated_sides['right']):
                        mode = 'calibration'
                        calibration_start_time = {'left': None, 'right': None}
                        prev_shoulder_pos = {'left': None, 'right': None}
                        continue

                    # check if undready-sides are ready using checkArmReady.
                    # only sides in true_sides should be validated.
                    # if other sides are ready:true, then decalibration timer should start bc sides are switching.
                    for side in true_sides:
                        ready, pos = checkArmReady(side, pose_landmarks, display_image, prev_shoulder_pos[side])
                        prev_shoulder_pos[side] = pos

                        # if User out of Camera View, Decalibrate Timer
                        if pos is None:
                            if decalibrationTimer(away_start_time, side):
                                resetCalibration(calibration_start_time, calibrated_sides, side)
                                mode = 'calibration'
                                calibration_start_time = {'left': None, 'right': None}
                                prev_shoulder_pos = {'left': None, 'right': None}
                                break
                        else: 
                            away_start_time[side] = None # reset the away timer if the user is back in view
                        
                        # Cconditions: Must be in proper curl form and must have not curled already (to prevent multiple counts for one curl)
                        if isProperForm(side, pose_landmarks, display_image) and curled_sides[side] == False:
                            curlTimer(curling_start_time, curled_sides, curlCount, side)
                        else:
                            curling_start_time[side] = None

                        # Conditions: Must have curled already and resting.
                        if curled_sides[side] and isArmResting(side, pose_landmarks, display_image): 
                            restingTimer(resting_start_time, curled_sides, side)
                        else:
                            resting_start_time[side] = None

                    drawStatus(display_image, mode, calibrated_sides, progress)
            
                else:  # regular mode: just read and display the angles
                    calcLeftBicepAngle(pose_landmarks, display_image)
                    calcRightBicepAngle(pose_landmarks, display_image)
                    calcLeftForearmAngle(pose_landmarks, display_image)
                    calcRightForearmAngle(pose_landmarks, display_image)
                    drawStatus(display_image, mode, calibrated_sides, 1.0)

            # Display the annotated image (arm lines + status only, no skeleton)
            cv2.imshow('Pose Landmarker', display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                mode = 'regular' if mode == 'calibration' else 'calibration'
                if mode == 'calibration':
                    calibration_start_time = {'left': None, 'right': None}
                    prev_shoulder_pos = {'left': None, 'right': None}
            elif key == ord('r'):
                calibrated_sides = {'left': False, 'right': False}
                calibration_start_time = {'left': None, 'right': None}
                prev_shoulder_pos = {'left': None, 'right': None}
                mode = 'calibration'

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        pass
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()