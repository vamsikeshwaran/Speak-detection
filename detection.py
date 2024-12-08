import cv2
import mediapipe as mp
import math
import time
import pyaudio
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 1000


def is_audio_detected(data):
    amplitude = np.frombuffer(data, dtype=np.int16)
    return np.max(np.abs(amplitude)) > THRESHOLD


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
face_ids = {}
available_ids = []
last_speaking_times = {}

last_audio_detection_time = None


def assign_face_id(face_index):
    """Assign a consistent ID to each face."""
    if face_index not in face_ids:
        if available_ids:
            face_ids[face_index] = available_ids.pop(0)
        else:
            face_ids[face_index] = len(face_ids) + 1
    return face_ids[face_index]


def release_unused_ids(active_faces):
    """Release IDs of faces no longer detected."""
    global face_ids, available_ids
    unused_faces = set(face_ids.keys()) - active_faces
    for face in unused_faces:
        available_ids.append(face_ids[face])
        available_ids.sort()
        del face_ids[face]


def draw_keypoints(frame, landmarks, indices, color):
    for idx in indices:
        x = int(landmarks[idx].x * frame.shape[1])
        y = int(landmarks[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, color, -1)


def calculate_distance(p1, p2, frame_shape):
    x1, y1 = int(p1.x * frame_shape[1]), int(p1.y * frame_shape[0])
    x2, y2 = int(p2.x * frame_shape[1]), int(p2.y * frame_shape[0])
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), (x1, y1), (x2, y2)


def get_bounding_box(landmarks, frame_shape):
    x_coords = [int(landmark.x * frame_shape[1]) for landmark in landmarks]
    y_coords = [int(landmark.y * frame_shape[0]) for landmark in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max


audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    active_faces = set()

    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_detected = is_audio_detected(data)

    if audio_detected:
        if last_audio_detection_time is None:
            last_audio_detection_time = time.time()

    if last_audio_detection_time is not None and (time.time() - last_audio_detection_time) <= 1:
        audio_detected = True
    else:
        audio_detected = False
        last_audio_detection_time = None

    if results.multi_face_landmarks:
        for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
            face_id = assign_face_id(face_index)
            active_faces.add(face_index)

            if face_id not in averages:
                averages[face_id] = {
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

            horizontal_avg = averages[face_id]["horizontal_avg"]
            vertical_avg = averages[face_id]["vertical_avg"]

            if horizontal_avg is None:
                horizontal_avg = horizontal_distance
                vertical_avg = vertical_distance
            else:
                horizontal_avg = alpha * horizontal_distance + \
                    (1 - alpha) * horizontal_avg
                vertical_avg = alpha * vertical_distance + \
                    (1 - alpha) * vertical_avg

            averages[face_id]["horizontal_avg"] = horizontal_avg
            averages[face_id]["vertical_avg"] = vertical_avg

            bounding_box = get_bounding_box(
                face_landmarks.landmark, frame.shape)
            cv2.line(frame, point1, point2, (255, 255, 0), 2)
            cv2.line(frame, upper_point, lower_point, (255, 255, 0), 2)

            is_speaking = (
                abs(horizontal_distance - horizontal_avg) > threshold or
                abs(vertical_distance - vertical_avg) > threshold
            )

            if is_speaking:
                last_speaking_times[face_id] = time.time()

            if face_id in last_speaking_times and \
                    (time.time() - last_speaking_times[face_id]) <= 1 and audio_detected:
                status = "Speaking"
            else:
                status = "Not Speaking"

            print(status)

            cv2.putText(frame, f"ID: {face_id} {status}", (bounding_box[0], bounding_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status == "Speaking" else (0, 0, 255), 2)
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                          (bounding_box[2], bounding_box[3]), (0, 255, 0) if status == "Speaking" else (0, 0, 255), 2)

    release_unused_ids(active_faces)

    cv2.imshow('Lip Keypoints Detection with Audio & Speaking Status', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

stream.stop_stream()
stream.close()
audio.terminate()
