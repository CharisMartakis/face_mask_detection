import tensorflow as tf
import cv2
import imutils
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import sys

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mask_string = r"mask_detection_model_optim.tflite"
maskNet = tf.lite.Interpreter(model_path=mask_string)
maskNet.allocate_tensors()

print("[ΕΝΗΜΕΡΩΣΗ] Το βίντεο ξεκίνησε...")
vs = VideoStream(src=0).start()
fl = 0

while True:
    if fl == 1:
        if cv2.getWindowProperty("Mask Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
    fl = 1

    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    faces = detect_faces(frame, face_cascade)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        face = np.array(face, dtype="float32")
        face = np.expand_dims(face, axis=0)

        maskNet.set_tensor(maskNet.get_input_details()[0]['index'], face)
        maskNet.invoke()

        output_tensor = maskNet.get_tensor(maskNet.get_output_details()[0]['index'])
        preds = np.squeeze(output_tensor)

        (mask, withoutMask) = preds

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Mask Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

vs.stream.release()
cv2.destroyAllWindows()