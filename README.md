# Hand & Face detection model

>[!NOTE]
>Hand/Face detection model using python (For beginner to Intermediate Python Programmers).

---------------------------------------------------------------------------

**ABOUT THIS PROJECT-**

This project is a real-time face and hand detection system that leverages machine learning and computer vision techniques. The system captures video from a camera, identifies and tracks faces, and recognizes specific individuals based on pre-loaded images and data about them. The project integrates multiple technologies, including TensorFlow for GPU acceleration, MediaPipe for face landmarks detection, and Face Recognition for identifying known faces. The detected faces are annotated with their name and age, or an indication if the face is not found in the database. Additionally, facial landmarks are drawn on the detected faces.

---------------------------------------------------------------------------

**EXAMPLES OF THE OUTCOME-**

https://github.com/user-attachments/assets/59ba1655-adee-4ad8-86f6-70ac39931561


![image](https://github.com/user-attachments/assets/826a7797-06ef-43da-bf1d-c89fd0325ab8)


![image](https://github.com/user-attachments/assets/2912f7e2-40b7-4539-b48d-47142ca40397)


---------------------------------------------------------------------------

**REQUIRNMENTS-**

Language, Libraries, Software and Hardware: 

- Python 3.10 (recommended version for compatibility with TensorFlow GPU)
- [opencv-python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html): For video capture and display.
- [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide): For facial landmarks detection.
- [face-recognition](https://face-recognition.readthedocs.io/en/latest/readme.html): For face detection and recognition.
- [tensorflow-gpu](https://www.tensorflow.org/guide/gpu): To leverage GPU for faster processing.
- DroidCam: Used to capture video from a mobile device as a webcam.
- A system with an NVIDIA GPU and CUDA support for TensorFlow GPU acceleration.

---------------------------------------------------------------------------

**STEP BY STEP GUIDE BELOW**

---------------------------------------------------------------------------

1. Importing Libraries and Checking GPU Availability
   
```python
import cv2
import mediapipe as mp
import face_recognition
import tensorflow as tf

# Check GPU availability for TensorFlow
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

```

Explanation- We start by importing libraries (Make sure you have installed python), we use a command (pip install .library name. for any kind of python library we need to install.) We have imported cv2 which is used for capturing the video, mediapipe is use to make the facial landmark, Face_recognition is used to match the face in the webcam, with a data of face images that we'll add in this project, if any of the picture matches with the face on webcam it shows certain data that we added for that face. At last we import tensorflow which make all these process smoother by making our GPU do all these stuff (You must have a Nvdia Compatible GPU).

---------------------------------------------------------------------------

2. Initializing MediaPipe for Face Landmarks Detection
   
```python
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
```

Explanation- mp_face_mesh will initializes the MediaPipe Face Mesh solution for detecting facial landmarks. face_mesh will creates a FaceMesh object that can detect up to 2 faces in the frame with specified confidence levels for detection and tracking and mp_drawing is a utility for drawing the detected facial landmarks on the frames.

---------------------------------------------------------------------------

3. Loading Known Face Encodings
   
```python
known_face_encodings = []
known_face_names = []

# Load image from the absolute path and extract encoding
try:
    print("Loading image...")
    image = face_recognition.load_image_file('D:/new/faces/Pranav.jpg') #Place the path of image you want to be detected
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    name = "Pranav"
    age = "18"
    known_face_names.append((name, age))
    print("Image loaded and encoding extracted.")
except Exception as e:
    print(f"Error loading image Pranav.jpg: {e}")
```

You can use this same command for as many pictures you want just make sure the folders are in the right places where it should be 


![image](https://github.com/user-attachments/assets/bd61f5e8-7244-4d85-ae9b-e436270cb45f)


like this
- Make a main directory or folder
- under which make aother folder named "faces"
- Inside that put as many pictures you want, make sure to rename them as Person1.jpg Person2.jpg for ease.

Explanation- known_face_encodings and known_face_names lists are initialized to store face encodings and corresponding names. The script attempts to load an image (in my case Pranav.jpg) and extracts the face encoding using the face_recognition library. This encoding is added to the list of known faces. If the image is loaded and processed successfully, the encoding and corresponding name/age (You can store any data you want, Such as Nmae and the employee level) are stored. Any issues during this process are caught and displayed.

---------------------------------------------------------------------------

4. Capturing Video from [DroidCam](https://www.dev47apps.com/)

Droid Cam is a free software (You dont need a webcam) download droid cam in your phone as well as your pc/laptop connect them with either usb or wifi. Once done Modify the old code by adding these new lines 

```python
cap = cv2.VideoCapture(0)

# Set resolution to 640x480 (adjust as necessary for speed and quality)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Set FPS to 60 (or the maximum supported by your camera)
cap.set(cv2.CAP_PROP_FPS, 60)
```

Explanation: cap = cv2.VideoCapture(0) is used for Capturing video from the first available camera device, in this case, DroidCam. cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320): Sets the video frame width to 320 pixels. cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240): Sets the video frame height to 240 pixels(if you want you can change this according to your liking).  cap.set(cv2.CAP_PROP_FPS, 60): Attempts to set the frames per second (FPS) to 60, aiming for smooth video output. (In your case according to your Pc/Laptop specification and monitor refresh rate you can change this value "60".

---------------------------------------------------------------------------

5. Setting Up the Display Window

```python
cv2.namedWindow('Face and Hand Detection', cv2.WINDOW_NORMAL)
```
---------------------------------------------------------------------------

6. Processing Each Frame

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

```

Explanation: Our script enters an infinite loop to continuously capture and process each video frame. If the frame is captured successfully, it is converted from BGR (OpenCV's default color format) to RGB (required for face recognition). [Read Documentation on BGR TO RGB OpenCV'S if you are not aware of this process](https://www.geeksforgeeks.org/convert-bgr-and-rgb-with-python-opencv/)

---------------------------------------------------------------------------

7. Face Recognition
   
```python
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

```
Explanation: The face_recognition library is used to detect face locations and encodings in the current frame. For each detected face, the script compares the encoding with the known face encodings (all the faces we have saved in faces folder) to find a match. If a match is found, the corresponding data (name and age, in my case) are retrieved and displayed on the frame; otherwise, the message "Face not detected in database" is displayed.

---------------------------------------------------------------------------

8. Facial Landmarks Detection
   
```python
results = face_mesh.process(frame_rgb)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
        )
```

Explanation: The script processes the frame with face_mesh to detect facial landmarks. If landmarks are detected, they are drawn on the face using the mp_drawing utility, with green lines and red dots representing the landmarks.

---------------------------------------------------------------------------

9. Displaying the Results
    
```python
if not face_locations:
    cv2.putText(frame, "No Face Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

cv2.imshow('Face and Hand Detection', frame)

if cv2.waitKey(10) & 0xFF == ord('q'):
    break
```

Explanation: If no faces are detected in the frame, the message "No Face Detected" is displayed. The processed frame is displayed in the window. The loop continues until the user presses the 'q' or Ctrl+ C in the terminal key.

---------------------------------------------------------------------------

10. Releasing Resources
    
```python
cap.release()
cv2.destroyAllWindows()
```

Explanation: Once the loop exits, the video capture is released, and all OpenCV windows are closed to free up resources.

---------------------------------------------------------------------------


- In these 10 simple steps we are create an Face recognition and also learn about movement detectors, without even buying any hardware like arduino. Hardware like arduino, Sensors etc can help us make advance models but these are Expensive, So if you still want to make this type of projects there are libraies available.

---------------------------------------------------------------------------

>[!NOTE]
>If you are intrested in these type of projects do check out my other computer vision and python projects-

>[Multi Stream Vision-Real Time Object Detection, 2023](https://github.com/pranavvss/Multi-Stream-Vision-Real-Time-Object-Detection-)- Under this project I have included object detection and used a pre trained data for more accuracy to test my code, This project already contains all the features that were mentioned above in Hand-Face-Detection-Model.

>[Gestura: Multi Control Interface, 2023](https://github.com/pranavvss/Gestura-Multi-Control-Interface)- Under this project I have implemented the concept of how we do certain task on our computer using Gestures and Webcam, which our python program detects and implement on the computer in real time.

>ThankYou.

