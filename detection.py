import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=10, refine_landmarks=True
)

UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 78]
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
UPPER_LIP_CENTER = 0
LOWER_LIP_CENTER = 17


alpha = 0.2
threshold = 7

averages = {}


def draw_keypoints(frame, landmarks, indices, color):
    """Draw keypoints for specified indices."""
    for idx in indices:
        x = int(landmarks[idx].x * frame.shape[1])
        y = int(landmarks[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, color, -1)


def calculate_distance(p1, p2, frame_shape):
    """Calculate Euclidean distance between two points."""
    x1, y1 = int(p1.x * frame_shape[1]), int(p1.y * frame.shape[0])
    x2, y2 = int(p2.x * frame_shape[1]), int(p2.y * frame.shape[0])
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), (x1, y1), (x2, y2)


def get_bounding_box(landmarks, frame_shape):
    """Calculate the bounding box for a face based on landmarks."""
    x_coords = [int(landmark.x * frame_shape[1]) for landmark in landmarks]
    y_coords = [int(landmark.y * frame_shape[0]) for landmark in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
            if face_index not in averages:
                averages[face_index] = {
                    "horizontal_avg": None, "vertical_avg": None}

            draw_keypoints(frame, face_landmarks.landmark,
                           UPPER_LIP_INDICES, (0, 0, 255))
            draw_keypoints(frame, face_landmarks.landmark,
                           LOWER_LIP_INDICES, (255, 0, 0))

            horizontal_distance, point1, point2 = calculate_distance(
                face_landmarks.landmark[LEFT_LIP_CORNER],
                face_landmarks.landmark[RIGHT_LIP_CORNER],
                frame.shape
            )
            vertical_distance, upper_point, lower_point = calculate_distance(
                face_landmarks.landmark[UPPER_LIP_CENTER],
                face_landmarks.landmark[LOWER_LIP_CENTER],
                frame.shape
            )

            horizontal_avg = averages[face_index]["horizontal_avg"]
            vertical_avg = averages[face_index]["vertical_avg"]

            if horizontal_avg is None:
                horizontal_avg = horizontal_distance
                vertical_avg = vertical_distance
            else:
                horizontal_avg = alpha * horizontal_distance + \
                    (1 - alpha) * horizontal_avg
                vertical_avg = alpha * vertical_distance + \
                    (1 - alpha) * vertical_avg

            averages[face_index]["horizontal_avg"] = horizontal_avg
            averages[face_index]["vertical_avg"] = vertical_avg

            bounding_box = get_bounding_box(
                face_landmarks.landmark, frame.shape)
            cv2.line(frame, point1, point2, (255, 255, 0), 2)
            cv2.line(frame, upper_point, lower_point,
                     (255, 255, 0), 2)

            if (abs(horizontal_distance - horizontal_avg) > threshold or
                    abs(vertical_distance - vertical_avg) > threshold):
                cv2.putText(frame, "Speaking", (bounding_box[0], bounding_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                              (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Speaking", (bounding_box[0], bounding_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                              (bounding_box[2], bounding_box[3]), (0, 0, 255), 2)

    cv2.imshow('Lip Keypoints Detection with Lines and Bounding Box', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
