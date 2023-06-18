import os
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
import tensorflow as tf
INIT_LR = 0.001
EPOCHS = 20
BS = 2
IMAGE_SIZE = 224

#									"""Μέρος 1ο - Preprocessing"""

dataset_location = r"C:\Users\NitroJin X\Desktop\Διπλωματική 2022- Τσακιρίδης Οδυσσέας\3)CODING\DATASETS\my_custom_only_for_TESTS\dataset"
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

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

#
# #									"""Μέρος 2ο - Data augmentation-Αύξηση δεδομένων"""
#
# aug = ImageDataGenerator(
# rotation_range=20,
# zoom_range=0.15,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.15,
# horizontal_flip=True,
# fill_mode="nearest")
#
# #						"""Μέρος 3ο - Κατασκευή του μοντέλου με τη μέθοδο TRANSFER LEARNING """
#
# baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
#
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)
#
# model = Model(inputs=baseModel.input, outputs=headModel)
#
# for layer in baseModel.layers:
# 	layer.trainable = False
#
# opt = Adam(learning_rate=INIT_LR)
#
# print("[ΕΝΗΜΕΡΩΣΗ] Γίνεται μεταγλώττιση του μοντέλου...")
# model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
#
#
# #						"""Μέρος 4ο - Εκπαίδευση και αξιολόγηση του μοντέλου """
#
#
# print("[ΕΝΗΜΕΡΩΣΗ] Η εκπαίδευση του μοντέλου(head μέρος) ξεκίνησε...")
# HISTORY = model.fit( aug.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS, validation_data=(testX, testY), validation_steps=len(testX) // BS, epochs=EPOCHS)
#
# print("[ΕΝΗΜΕΡΩΣΗ] Αποθήκευση του μοντέλου ανίχνευσης μάσκας στον φάκελο...")
# model.save("models/mask_detector.model", save_format="h5")




######################################################################################################
model = tf.keras.models.load_model(r'D:\projects\face_mask_detection\models\mask_detector.model')
# loss, acc = model.evaluate(testX, testY, batch_size=2, verbose=0)
# print(f"test accuracy {acc*100}")
# print(f"test accuracy {loss*100}")

print(testX)
predIdxs = model.predict(testX, batch_size=BS)
print(predIdxs)
predIdxs = np.argmax(predIdxs, axis=1)
# predIdxss = np.argmax(predIdxs, axis=0)
print(predIdxs)
# print(predIdxss)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
print(testY)
print(type(testY))

#Εμφάνιση στην οθόνη του τερματικού μιας αναφοράς ταξινόμησης(classification report) που παρέχει μετρήσεις όπως η
#ανάκληση(recall), η βαθμολογία F1(F1-score), η ακρίβεια(precision), η ακρίβεια(accuracy) και υποστήριξη για κάθε
#label/class, καθώς και έναν #μέσο όρο σε όλες τις κλάσεις. Για τη διαδικασία παραγωγής όλων των παραπάνω πήροφοριών είναι
#υπεύθυνη η εντολή #classification_report() της βιβλιοθήκης sklearn.metrics. Αυτή η εντολή δέχεται αρχικά ως δεδομένα τις
#πραγματικές τιμές των labels που αντιστοιχούν στο testX σετ. Τα δεδομένα αυτά βρίσκονται θεωρητικά στην μεταβλητή testY
#αλλά επειδή αυτά τα δεδομένα είναι της μορφής one-hot encoding, θα πρέπει πρώτα να μετατραπούν στην μορφή μονοδιάστατου
#διανύσματος. Για αυτή τη διαδικασία χρησιμοποιείται η μέθοδος .argmax η οποία συμπεριφέρεται με τον ίδιο τρόπο που
#εκτελέστηκε και για το predIdxs προηγουμένως. Πιο συγκεκριμένα,η .argmax θα ελέγξει κάθε γραμμή του testY array αφού το
#axis έχει ορισθεί ως 1 και θα επιστρέψει τον δείκτη με τη μεγαλύτερη τιμή. Παίρνοντας ως παράδειγμα μια σειρά του testY
#που έχει τη μορφή [0. 1.] αυτή θα ελεγχθεί και θα επιστρέψει την τιμή 1 στο νέο μονοδιάστατο διάνυσμα του testY αφού
#μεταξύ του 0 και 1, μεγαλύτερο είναι το 1 που βρίσκεται στη στήλη 1. Ακόμα το classification_report δέχεται ως δεδομένα
#τις τιμές predIdxs που προβλέφθηκαν και προορίζονται για σύγκριση με τις πραγματικές τιμές του testY. Τέλος ενεργοποιείται
#η ιδιότητα target_names στην οποία δίνεται η τιμή lb.classes_ που επιστρέφει το array με τις αλφαριθμητικές ονομασίες τω
#2 labels (with_mask και without_mask). Αυτές δίνονται στο classification_report έτσι ώστε αυτό να προβάλλει τις πληροφορίες
#με περισσότερη αναγνωσιμότητα. Παρακάτω αναλύονται τα metrics που θα εκτυπωθούν:
#1)ανάκληση(recall):  Υπολογίζεται ως η αναλογία μεταξύ του αριθμού των Θετικών δειγμάτων(π.χ. with_mask) που ταξινομήθηκαν
#σωστά ως Θετικά προς τον συνολικό αριθμό των Θετικών δειγμάτων. Η ανάκληση μετράει την ικανότητα του μοντέλου να ανιχνεύει
#Θετικά δείγματα. Όταν η ανάκληση είναι υψηλή τότε το μοντέλο μπορεί να ταξινομήσει σωστά όλα τα θετικά δείγματα ως Θετικά.
#και θεωρείται αξιόπιστο ως προς την ικανότητά του να ανιχνεύει θετικά δείγματα.

#2)βαθμολογία F1(F1-score):

#3)ακρίβεια(precision): Υπολογίζεται ως η αναλογία μεταξύ του αριθμού των Θετικών δειγμάτων(π.χ. with_mask) που ταξινομήθηκαν
#σωστά προς τον συνολικό αριθμό των δειγμάτων που ταξινομήθηκαν ως Θετικά(είτε σωστά είτε λανθασμένα) και μετρά την ακρίβεια
#του μοντέλου στην ταξινόμηση ενός δείγματος ως θετικού. Έτσι, προβάλλει πόσο αξιόπιστο είναι το μοντέλο στην ταξινόμηση
#των δειγμάτων ως Θετικών.

#4)ακρίβεια(accuracy): Υπολογίζεται ως η αναλογία μεταξύ του αριθμού των σωστών προβλέψεων προς τον συνολικό αριθμό των
#προβλέψεων και περιγράφει την απόδοση του μοντέλου σε όλες τις κλάσεις.

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))