import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path='simplified_emotion_model_from_csv.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Load OpenCV's Haar Cascade for face detection
cascade_path = '/home/ozanbaba/opencv_data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Cannot load haarcascade_frontalface_default.xml file")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define emotions
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Initialize variables for emotion detection timing
frame_count = 0
emotion = None

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    # Debugging statement to print number of faces detected
    print(f"Faces detected: {len(faces)}")

    # Perform emotion detection every 150 frames (approximately every 5 seconds at 30 FPS)
    if frame_count % 150 == 0 and len(faces) > 0:
        # Assume only one face for simplicity, take the first detected face
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=-1)  # Add the channel dimension
        face = np.expand_dims(face, axis=0)   # Add the batch dimension
        face = face.astype(np.float32) / 255.0  # Normalize the image

        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], face)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        emotion = np.argmax(output_data)

    # Draw a frame around the detected face
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display the detected emotion label on the frame
        if emotion is not None:
            emotion_label = emotions[emotion]
            cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Detected Emotion', frame)

    # Increment frame count
    frame_count += 1

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
