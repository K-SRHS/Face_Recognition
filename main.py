from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the sticker image
sticker = cv2.imread('C:/Users/yunho/Desktop/PythonWorkspace/MPJ/sticker.png', cv2.IMREAD_UNCHANGED)

# Convert the sticker to RGBA format
sticker = cv2.cvtColor(sticker, cv2.COLOR_BGR2BGRA)

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Initialize flags for selected effects
apply_mosaic = False
apply_sticker = False

def apply_effects(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply the selected effects to each detected face
    for (x, y, w, h) in faces:
        # Apply mosaic effect
        if apply_mosaic:
            # Extract the region of interest (face) from the frame
            face = frame[y:y+h, x:x+w]

            # Apply mosaic blur to the face
            face = cv2.resize(face, (w // 10, h // 10))
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)

            # Replace the original face with the blurred face
            frame[y:y+h, x:x+w] = face

        # Apply sticker effect
        if apply_sticker:
            # Resize the sticker to fit the face
            sticker_resized = cv2.resize(sticker, (w, h))

            # Calculate the coordinates for pasting the sticker
            x_offset = x
            y_offset = y

            # Get the region of interest (ROI) in the frame
            roi = frame[y_offset:y_offset + sticker_resized.shape[0], x_offset:x_offset + sticker_resized.shape[1]]

            # Create a mask to exclude the transparent part of the sticker
            mask = sticker_resized[:, :, 3] / 255.0

            # Apply the sticker to the ROI
            for c in range(0, 3):
                roi[:, :, c] = sticker_resized[:, :, c] * mask + roi[:, :, c] * (1 - mask)

    return frame

def generate_frames():
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        if not ret:
            break

        # Apply the selected effects to the frame
        output_frame = apply_effects(frame)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', output_frame)

        # Get the bytes of the image
        frame_bytes = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
