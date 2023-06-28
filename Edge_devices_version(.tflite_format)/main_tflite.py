import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

def detect_and_predict_mask(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		preds = []

		for face in faces:
			face = np.expand_dims(face, axis=0)
			maskNet.set_tensor(input_details[0]['index'], face)

			maskNet.invoke()

			pred = maskNet.get_tensor(output_details[0]['index'])
			preds.append(pred)

	return (locs, preds)

prototxtPath = r"face_detector_model/deploy.prototxt"
weightsPath = r"face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

mask_string = r"mask_detection_model_optim.tflite"
maskNet = tf.lite.Interpreter(model_path=mask_string)
maskNet.allocate_tensors()
input_details = maskNet.get_input_details()
output_details = maskNet.get_output_details()

print("[ΕΝΗΜΕΡΩΣΗ] Το βίντεο ξεκίνησε...")
vs = VideoStream(src=0).start()
fl = 0

while True:

	if cv2.getWindowProperty("Mask_Detection", cv2.WND_PROP_VISIBLE) < 1 & fl == 1:
		break
	else:
		fl = 1

	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred[0]

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, label, (startX, startY - 10),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Mask Detection", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == 27:
		break

vs.stream.release()
cv2.destroyAllWindows()

