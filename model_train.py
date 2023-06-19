import os
import sys
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#ORIGINAL BEST VALUES
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
IMAGE_SIZE = 224

# #TEST MODE
# INIT_LR = 0.001
# EPOCHS = 40
# BS = 2
# IMAGE_SIZE = 224


#									"""Μέρος 1ο - Preprocessing"""

folder_of_model = f"models/{INIT_LR}_{EPOCHS}_{BS}"

if os.path.exists(folder_of_model):
	print(f"Αυτή η έκδοση του μοντέλου που προσπαθήσατε να εκπαιδεύσετε ({folder_of_model[7:]}) έχει ήδη δημιουργηθεί."
	f" Το πρόγραμμα θα τερματιστεί τώρα...")
	sys.exit()

print(f"[ΕΝΗΜΕΡΩΣΗ] Η εκπαίδευση του μοντέλου με ονομασία έκδοσης '{folder_of_model[7:]}' ξεκίνησε...")
os.makedirs(folder_of_model)

dataset_location = r"D:\projects\face_mask_detection\dataset"
dataset_classes = ["with_mask", "without_mask"]
data = []
labels = []

print("[ΕΝΗΜΕΡΩΣΗ] Η φόρτωση των εικόνων ξεκίνησε...")

for dataset_class in dataset_classes:
	path = os.path.join(dataset_location, dataset_class)

	for img_name in os.listdir(path):

		img_path = os.path.join(path, img_name)
		image = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
		image = img_to_array(image)
		image = preprocess_input(image)
		data.append(image)
		labels.append(dataset_class)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

#									"""Μέρος 2ο - Data augmentation-Αύξηση δεδομένων"""

aug = ImageDataGenerator(
rotation_range=20,
zoom_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.15,
horizontal_flip=True,
fill_mode="nearest")

#						"""Μέρος 3ο - Κατασκευή του μοντέλου με τη μέθοδο TRANSFER LEARNING """

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

opt = Adam(learning_rate=INIT_LR)

print("[ΕΝΗΜΕΡΩΣΗ] Γίνεται μεταγλώττιση του μοντέλου...")

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#						"""Μέρος 4ο - Εκπαίδευση και αξιολόγηση του μοντέλου """

print("[ΕΝΗΜΕΡΩΣΗ] Η εκπαίδευση του μοντέλου(head μέρος) ξεκίνησε...")

HISTORY = model.fit( aug.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS, validation_data=(testX, testY), validation_steps=len(testX) // BS, epochs=EPOCHS)
model.save(f"{folder_of_model}/mask_detector.model", save_format="h5")

print("[ΕΝΗΜΕΡΩΣΗ] Αξιολόγηση του νευρωνικού δικτύου...")

predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
report = classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_)
print(report)

with open(f'{folder_of_model}/classification_report.txt', 'w') as file:
	file.write(report)
	file.close()

#						"""Μέρος 5ο - Δημιουργία διαγραμμάτων για το training loss & accuracy """

#Εκτύπωση ενημερωτικού μηνύματος στην οθόνη
print("[ΕΝΗΜΕΡΩΣΗ] Η γραφική απεικόνιση των μετρήσεων ξεκίνησε...")

final_train_loss = HISTORY.history["loss"][-1]
final_train_acc = HISTORY.history["accuracy"][-1]
final_val_loss = HISTORY.history["val_loss"][-1]
final_val_acc = HISTORY.history["val_accuracy"][-1]

plt.style.use("bmh")
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, EPOCHS+1), HISTORY.history["loss"], label="training loss")
plt.plot(np.arange(1, EPOCHS+1), HISTORY.history["val_loss"], label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.annotate(f"Final Train Loss: {final_train_loss:.4f}", (EPOCHS, final_train_loss), textcoords="offset points", xytext=(-10, 20), ha='right', color='blue')
plt.annotate(f"Final Val Loss: {final_val_loss:.4f}", (EPOCHS, final_val_loss), textcoords="offset points", xytext=(-10, 20), ha='right', color='red')
plt.margins(x=0, y=0)
tick_locations = np.arange(0, EPOCHS+1, 5)
tick_locations[0] = 1
plt.xticks(tick_locations, tick_locations)
plt.savefig(f"{folder_of_model}/loss_plot.png", dpi=300, bbox_inches="tight")

plt.style.use("bmh")
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, EPOCHS+1), HISTORY.history["accuracy"], label="training accuracy")
plt.plot(np.arange(1, EPOCHS+1), HISTORY.history["val_accuracy"], label="validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.annotate(f"Final Train Acc: {final_train_acc:.4f}", (EPOCHS, final_train_acc), textcoords="offset points", xytext=(-10, -20), ha='right', color='blue')
plt.annotate(f"Final Val Acc: {final_val_acc:.4f}", (EPOCHS, final_val_acc), textcoords="offset points", xytext=(-10, -20), ha='right', color='red')
plt.margins(x=0, y=0)
tick_locations = np.arange(0, EPOCHS+1, 5)
tick_locations[0] = 1
plt.xticks(tick_locations, tick_locations)
plt.savefig(f"{folder_of_model}/accuracy_plot.png", dpi=300, bbox_inches="tight")

plt.style.use("dark_background")
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, EPOCHS+1), HISTORY.history["loss"], label="training loss", color='blue')
plt.plot(np.arange(1, EPOCHS+1), HISTORY.history["val_loss"], label="validation loss", color='yellow')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.annotate(f"Final Train Loss: {final_train_loss:.4f}", (EPOCHS, final_train_loss), textcoords="offset points", xytext=(-10, 20), ha='right', color='blue')
plt.annotate(f"Final Val Loss: {final_val_loss:.4f}", (EPOCHS, final_val_loss), textcoords="offset points", xytext=(-10, 20), ha='right', color='yellow')
plt.margins(x=0, y=0)
tick_locations = np.arange(0, EPOCHS+1, 5)
tick_locations[0] = 1
plt.xticks(tick_locations, tick_locations)
plt.savefig(f"{folder_of_model}/loss_plot_dark_background.png", dpi=300, bbox_inches="tight")

plt.style.use("dark_background")
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, EPOCHS+1), HISTORY.history["accuracy"], label="training accuracy", color='blue')
plt.plot(np.arange(1, EPOCHS+1), HISTORY.history["val_accuracy"], label="validation accuracy", color='yellow')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.annotate(f"Final Train Acc: {final_train_acc:.4f}", (EPOCHS, final_train_acc), textcoords="offset points", xytext=(-10, -20), ha='right', color='blue')
plt.annotate(f"Final Val Acc: {final_val_acc:.4f}", (EPOCHS, final_val_acc), textcoords="offset points", xytext=(-10, -20), ha='right', color='yellow')
plt.margins(x=0, y=0)
tick_locations = np.arange(0, EPOCHS+1, 5)
tick_locations[0] = 1
plt.xticks(tick_locations, tick_locations)
plt.savefig(f"{folder_of_model}/accuracy_plot_dark_background.png", dpi=300, bbox_inches="tight")

print("[ΕΝΗΜΕΡΩΣΗ] Τέλος εκπαίδευσης του μοντέλου. Τερματισμός προγράμματος...")
