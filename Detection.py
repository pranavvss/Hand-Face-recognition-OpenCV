import cv2
import mediapipe as mp
import face_recognition
import tensorflow as tf

# Check GPU availability for TensorFlow
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Initialize MediaPipe Face Mesh for facial landmarks detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load image from the absolute path and extract encoding
try:
    print("Loading image...")
    image = face_recognition.load_image_file('D:/new/faces/Pranav.jpg')
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    name = "Pranav"
    age = "18"
    known_face_names.append((name, age))
    print("Image loaded and encoding extracted.")
except Exception as e:
    print(f"Error loading image Pranav.jpg: {e}")

# Capture video from DroidCam
cap = cv2.VideoCapture(0)

# Set resolution to 640x480 (adjust as necessary for speed and quality)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Set FPS to 60 (or the maximum supported by your camera)
cap.set(cv2.CAP_PROP_FPS, 60)

# Set up the named window for resizing
cv2.namedWindow('Face and Hand Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face recognition
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Face not detected in database"
        age = ""

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name, age = known_face_names[best_match_index]

        # Draw a box around the face and add name and age or the "Face not detected in database" message
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, f"Name: {name}, Age: {age}" if name != "Face not detected in database" else name,
                    (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            )

    if not face_locations:
        cv2.putText(frame, "No Face Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


    cv2.imshow('Face and Hand Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
