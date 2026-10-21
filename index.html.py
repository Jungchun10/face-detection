import cv2
import numpy as np
import math

# Face detector included with OpenCV
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("Error: Could not load OpenCV face detector.")
    raise SystemExit(1)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open webcam. Check camera permission or try index 1.")
    raise SystemExit(1)

# Approximate 3D face points (nose, chin, eye corners, mouth corners)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye outer corner
    (225.0, 170.0, -135.0),   # right eye outer corner
    (-150.0, -150.0, -125.0), # left mouth corner
    (150.0, -150.0, -125.0)   # right mouth corner
], dtype=np.float64)

def estimate_pitch_roll(face_gray, face_box, frame_shape):
    """Estimate rough pitch and roll from facial landmarks."""
    x, y, w, h = face_box

    # Detect eyes and mouth within the selected face crop.
    # These are approximate landmarks; Haar detections can vary by lighting.
    face_crop = face_gray[y:y+h, x:x+w]
    if face_crop.size == 0:
        return None, None

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(
        face_crop, scaleFactor=1.1, minNeighbors=5,
        minSize=(max(10, w // 12), max(10, h // 12))
    )

    # Pick two largest eye detections in the upper half of the face.
    eyes = [(ex, ey, ew, eh) for ex, ey, ew, eh in eyes if ey < h * 0.65]
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:4]
    if len(eyes) < 2:
        return None, None

    eyes = sorted(eyes, key=lambda e: e[0])[:]
    # Use centers of the two most horizontally separated detected eyes.
    pair = None
    best_distance = 0
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            c1 = (eyes[i][0] + eyes[i][2] / 2, eyes[i][1] + eyes[i][3] / 2)
            c2 = (eyes[j][0] + eyes[j][2] / 2, eyes[j][1] + eyes[j][3] / 2)
            distance = abs(c2[0] - c1[0])
            if distance > best_distance:
                best_distance = distance
                pair = (c1, c2)
    if pair is None or best_distance < w * 0.12:
        return None, None

    left_eye = pair[0]
    right_eye = pair[1]
    # Find a nose-like point using the lower central area of the face.
    noses = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_mcs_nose.xml"
    ) if False else None
    # Approximate nose position from face geometry.
    nose = (x + w * 0.5, y + h * 0.58)
    # Approximate mouth corners from face geometry.
    mouth_left = (x + w * 0.32, y + h * 0.76)
    mouth_right = (x + w * 0.68, y + h * 0.76)
    chin = (x + w * 0.5, y + h * 0.95)

    image_points = np.array([
        nose,
        chin,
        (x + left_eye[0], y + left_eye[1]),
        (x + right_eye[0], y + right_eye[1]),
        mouth_left,
        mouth_right
    ], dtype=np.float64)

    frame_h, frame_w = frame_shape[:2]
    focal_length = float(frame_w)
    camera_matrix = np.array([
        [focal_length, 0, frame_w / 2],
        [0, focal_length, frame_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch = float(angles[0])
    roll = float(angles[2])
    return pitch, roll

while True:
    ok, frame = camera.read()
    if not ok:
        print("Error: Could not read from webcam.")
        break

    # Mirror image so movement feels natural.
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) > 0:
        # Select only the largest visible face (usually the closest face).
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        center_x = x + w // 2
        center_y = y + h // 2

        pitch, roll = estimate_pitch_roll(gray, (x, y, w, h), frame.shape)

        # Change this threshold to adjust when the close-face warning appears.
        too_close = w >= int(frame.shape[1] * 0.35) or h >= int(frame.shape[0] * 0.55)
        box_color = (0, 0, 255) if too_close else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        cv2.putText(frame, f"X: {center_x}  Y: {center_y}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        if pitch is None or roll is None:
            cv2.putText(frame, "Pitch: --  Roll: -- (face landmarks unavailable)",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        else:
            cv2.putText(frame, f"Pitch: {pitch:.1f} deg  Roll: {roll:.1f} deg",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        if too_close:
            cv2.putText(frame, "WARNING: FACE TOO CLOSE!",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No face detected",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face Coordinates and Head Angles", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
