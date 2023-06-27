import cv2
import imutils
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Load the Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the face mask detector model from disk
mask_string = r"mask_detection_model_optim.tflite"
maskNet = tf.lite.Interpreter(model_path=mask_string)
maskNet.allocate_tensors()

# Start the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame
    faces = detect_faces(frame, face_cascade)

    # loop over the detected faces
    for (x, y, w, h) in faces:
        # extract the face ROI, convert it from BGR to RGB channel ordering, and preprocess it
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        # convert the face to a numpy array
        face = np.array(face, dtype="float32")
        face = np.expand_dims(face, axis=0)

        # run inference on the face mask detector model
        maskNet.set_tensor(maskNet.get_input_details()[0]['index'], face)
        maskNet.invoke()

        # get the output tensor and convert it to a numpy array
        output_tensor = maskNet.get_tensor(maskNet.get_output_details()[0]['index'])
        preds = np.squeeze(output_tensor)

        # determine the class label and color
        (mask, withoutMask) = preds
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # draw bounding box and label on the output frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

