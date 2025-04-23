import cv2
import mediapipe as mp
import math

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_eye_aspect_ratio(eye_landmarks):
    # Calculate Euclidean distances between the vertical eye landmarks
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])  # Vertical distance
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])  # Vertical distance
    
    # Calculate the horizontal distance between the eye corners
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])  # Horizontal distance
    
    # Eye Aspect Ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize Mediapipe FaceMesh
cap = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh()
marks = mp.solutions.drawing_utils

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    height, width, _ = img.shape
    results = face_mesh.process(img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get pixel coordinates of left and right eye landmarks
            left_eye_landmarks = [
                (face_landmarks.landmark[33].x * width, face_landmarks.landmark[33].y * height),  # outer corner
                (face_landmarks.landmark[160].x * width, face_landmarks.landmark[160].y * height),  # top
                (face_landmarks.landmark[158].x * width, face_landmarks.landmark[158].y * height),  # bottom
                (face_landmarks.landmark[133].x * width, face_landmarks.landmark[133].y * height),  # inner corner
                (face_landmarks.landmark[153].x * width, face_landmarks.landmark[153].y * height),  # top-middle
                (face_landmarks.landmark[144].x * width, face_landmarks.landmark[144].y * height)   # bottom-middle
            ]
            
            right_eye_landmarks = [
                (face_landmarks.landmark[362].x * width, face_landmarks.landmark[263].y * height),  # outer corner
                (face_landmarks.landmark[385].x * width, face_landmarks.landmark[386].y * height),  # top
                (face_landmarks.landmark[387].x * width, face_landmarks.landmark[374].y * height),  # bottom
                (face_landmarks.landmark[263].x * width, face_landmarks.landmark[362].y * height),  # inner corner
                (face_landmarks.landmark[373].x * width, face_landmarks.landmark[385].y * height),  # top-middle
                (face_landmarks.landmark[380].x * width, face_landmarks.landmark[380].y * height)   # bottom-middle
            ]

            # Calculate EAR for both eyes
            left_ear = calculate_eye_aspect_ratio(left_eye_landmarks)
            right_ear = calculate_eye_aspect_ratio(right_eye_landmarks)

            # Calculate average EAR
            ear = (left_ear + right_ear) / 2.0
            print(f"EAR: {ear:.2f}")

            # Define a blink threshold (experiment to find the best value)
            blink_threshold = 0.
            if ear < blink_threshold:
                cv2.putText(img, "Blinking", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Not Blinking", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw landmarks for debugging (optional)
            for (x, y) in left_eye_landmarks:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
            for (x, y) in right_eye_landmarks:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow('Blink Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
