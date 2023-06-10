import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Αρχικοποίηση των μεταβλητών initial learning rate, epochs και batch size. Κάνοντας αλλαγές στις παραμέτρους αυτές,
#μπορούμε να εκπαιδεύσουμε διαφορετικά μοντέλα και να τα συγκρίνουμε ώστε στο τέλος να κρατήσουμε εκείνο με τα καλύ-
#τερα δυνατά αποτελέσματα και το μεγαλύτερο ποσοστό επιτυχίας.
#1)Initial learning Rate(INIT_LR): Είναι μια υπερπαράμετρος που καθορίζει πόσο γρήγορα ή αργά η συνάρτηση βελτιστοποίησης
#του σφάλματος που έχουμε επιλέξει(Adam) κατεβαίνει την καμπύλη σφάλματος. Συνήθως η τιμή της βρίσκεται ανάμεσα στο
#0.0001 and 0.01.
#2)Epochs(EPOCHS): Είναι μια υπερπαράμετρος η οποία ορίζει τον αριθμό των επαναλήψεων που θα χρειαστεί να εκτελεστούν
#για την εκπαίδευση του μοντέλου
#3)Batch size(BS): Είναι μια υπερπαράμετρος η οποία θέτει τον αριθμό των δεδομένων εκπαίδευσης(trainX, trainY) που
#χρησιμοποιούμε σε μια εποχή(epoch) για να εκπαιδεύσουμε το νευρωνικό δίκτυο. Συνήθως για τα CNN επιλέγεται το 32.
INIT_LR = 0.001
EPOCHS = 20
BS = 32
									"""Μέρος 1ο - Preprocessing"""
#Δημιουργία μιας μεταβλητής (string type) με περιεχόμενο τη τοποθεσία των εικόνων
#που θα χρησιμοποιηθούν για την εκπαίδευση του μοντέλου
#dataset_location = r"D:\projects\face_mask_detection\dataset"

dataset_location = r"C:\Users\NitroJin X\Desktop\Διπλωματική 2022- Τσακιρίδης Οδυσσέας\3)CODING\DATASETS\my_custom_only_for_TESTS\dataset"

#Δημιουργία μιας λίστας με 2 περιεχόμενα, το "with_mask" και το "without_mask".
dataset_classes = ["with_mask", "without_mask"]

#Δημιουργία μιας λίστας που αργότερα θα περιέχει όλες τις εικόνες ως αριθμούς και
#πιο συγκεκριμένα ως arrays
data = []

#Δημιουργία μιας λίστας που αργότερα θα περιέχει για κάθε μια απο τις εικόνες της λίστας data
#αντίστοιχα έναν αριθμό ο οποίος θα σημαίνει ή "with_mask" ή "without_mask"
labels = []

#Εκτύπωση μηνύματος στην οθόνη
print("[ΕΝΗΜΕΡΩΣΗ] Η φόρτωση των εικόνων ξεκίνησε...")

#Δημιουργία βρόγχου επανάληψης που την πρώτη φορά το dataset_class = "with_mask"
#και τη δεύτερη/τελευταία φορά θα είναι dataset_class = "without_mask"
for dataset_class in dataset_classes:

	#Δημιουργία μεταβλητής τύπου string η οποία περιέχει το αποτέλεσμα
	#της ένωσης 2 επιμέρους αλφαριθμητικών, του dataset_location και του dataset_class
	#παραδείγματος χάριν:
	#dataset_location = r"D:\projects\face_mask_detection\dataset\"
	#dataset_class = "with_mask"
	#Άρα path = "D:\projects\face_mask_detection\dataset\with_mask"
	path = os.path.join(dataset_location, dataset_class)

	#Δημιουργία βρόγχου επανάληψης που σε κάθε επανάληψή του το περιεχόμενο της μεταβλητής img_name
	#θα είναι η ονομασία μιας απο τις εικόνες του φακέλου που δείχνει το path.
	#Πιο συγκεκριμένα, το os.listdir(path) δημιουργεί μια λίστα με όλες τις ονομασίες
	#των εικόνων που περιέχει ο φάκελος που δείχνει το path
	for img_name in os.listdir(path):
		#Δημιουργία μεταβλητής τύπου string η οποία περιέχει το αποτέλεσμα
		#της ένωσης 2 επιμέρους αλφαριθμητικών, του path και του img_name
		#παραδείγματος χάριν:
		#path = r"D:\projects\face_mask_detection\dataset\with_mask\"
		#img_name = "0_0_21.jpg"
		#Άρα img_path = "D:\projects\face_mask_detection\dataset\with_mask\0_0_21.jpg"
		img_path = os.path.join(path, img_name)

		#Εφόσον το img_path δείχνει στην τοποθεσία του αρχείου μιας συγκεκριμένης εικόνας
		#το load_img φορτώνει στην μεταβλητή image την εικόνα αυτή με τις συγκεκριμένες
		#διαστάσεις που ορίζει το target_size. Άρα με αυτήν την εντολή προσαρμόζουμε όλες
		#τις εικόνες έτσι ώστε να έχουν την ίδια διάσταση με το ίδιο aspect ratio που είχαν
		image = load_img(img_path, target_size=(224, 224))

		#Μετατροπή της εικόνας που βρίσκεται στο image σε μορφή πίνακα(array) για να μπορούμε
		#να την επεξεργαστούμε αργότερα πιο εύκολα. Ένα κομμάτι του array αυτού θα έχει για
		#παράδειγμα την παρακάτω μορφή:
		#[84. 58. 45.]
		#[84. 58. 45.]
		#[84. 58. 45.]
		image = img_to_array(image)

		#Επειδή για το CNN model μας θα χρησιμοποιήσουμε την αρχιτεκτονική του μοντέλου
		#mobilenet_v2, χρειάζεται να χρησιμοποιήσουμε την εντολή preprocess_input πάνω
		#στο array της εικόνας μας. Η εντολή αυτή κάνει κάποιες μετατροπές και προσαρμογές
		#στο array έτσι ώστε αυτό να είναι συμβατό κατά την εκπαίδευση του μοντέλου μας.
		#Λαμβάνοντας υπόψιν το παράδειγμα απεικόνισης ενός μέρους του array απο την προηγούμενη
		#εντολή μπορούμε να δούμε την διαφορά του παρακάτω , αφού δηλαδή υποστεί την εντολή του preprocess_input():
		#[-0.34117645 -0.54509807 -0.64705884]
		#[-0.34117645 -0.54509807 -0.64705884]
		#[-0.34117645 -0.54509807 -0.64705884]
		image = preprocess_input(image)

		#Προσθήκη με τη σειρά, όλων των arrays των εικόνων στη λίστα data ώστε να είναι
		#αποθηκευμένες με αυτή τη μορφή σε ένα μέρος που θα μας διευκολύνει στην μετέπειτα
		#επεξεργασία τους
		data.append(image)

		#Για κάθε μια απο τις εικόνες αποθηκεύεται αντίστοιχα και ένα label με την κατάσταση της
		#εικόνας. Είτε δηλαδη "with mask" είτε "without_mask". Όλα τα labels αποθηκεύονται σε μία λίστα
		#όπως και όλες οι εικόνες στην προηγούμενη εντολή για να μπορούν να επεξεργαστούν αργότερα τα δεδομένα
		#με μεγαλύτερη ευκολία
		labels.append(dataset_class)

#Επειδή τα deep learning μοντέλα λειτουργούν σωστά μόνο με arrays, τα labels παρακάτω
#απο λίστα μετατρέπονται και αυτά σε arrays και ειδικότερα τα δεδομένα του που ήταν
#προηγουμένως αλφαριθμητικά, πλέον θα είναι αριθμοί.
#Αρχικά καλείται η κλάση LabelBinarizer() η οποία δημιουργεί το αντικείμενο (object) lb.
#Αυτό γίνεται για να μπορούμε να χρησιμοποιήσουμε τις μεθόδους(class methods) της προαναφερόμενης
#κλάσης πιο εύκολα.
lb = LabelBinarizer()

#Το object lb καλεί την μέθοδο fit_transform πάνω στη λίστα labels και μετατρέπει όλα τα δεδομένα
#της σε 0 και 1. Δηλαδή το αλφαριθμητικό "with_mask" αντικαταστάθηκε με τον αριθμό 0 και το "without_mask"
#με τον αριθμό 1. Επίσης το labels απο λίστα μετατρέπεται σε array (class numpy.ndarray int32) με δεδομένα της μορφής
#[0] ή [1].
labels = lb.fit_transform(labels)

#Η μέθοδος to_categorial χρησιμοποιεί την κωδικοποίηση One-hot encoding η οποία μετατρέπει μια λίστα/array που περιέχει
#κατηγορίες όπως το labels σε μορφή τέτοια που μπορεί να χρησιμοποιηθεί εύκολα από αλγόριθμους μηχανικής
#εκμάθησης (machine learning algorithms).Η βασική ιδέα της κωδικοποίησης αυτής είναι η δημιουργία νέων μεταβλητών που
#λαμβάνουν τις τιμές 0 και 1 για να αντιπροσωπεύουν τις αρχικές κατηγορικές τιμές. Το labels μετά την κωδικοποίηση
#θα περιέχει δεδομένα της μορφής [1. 0.] για "with_mask" ή [0. 1.] για "without_mask" και θα έχει τα εξής χαρακτηριστικά
#(class numpy.ndarray float32)
labels = to_categorical(labels)

#Μετατροπή με τη βοήθεια της βιβλιοθήκης numpy(np) της λίστας data που περιέχει όλα τα array των εικόνων,
#ως ένα array και συγκεκριμένα τύπου float32.
data = np.array(data, dtype="float32")

#Το labels επειδή μετατράπηκε προηγουμένως σε array τύπου float32 δεν χρειάζεται να εκτελέσουμε την εξής ενολή:
#labels = np.array(labels)

#Η μέθοδος train_test_split της βιβλιοθήκης scikit-learn/(sklearn.model_selection) χωρίζει τα δεδομένα των data και
#labels σε 4 arrays 2 κατηγοριών. Ουσιαστικά τα δεδομένα τους θα χωριστούν σύμφωνα με το ποσοστό που ορίζει η ιδιότητα
#test_size όπου στην προκειμένη περίπτωση είναι 0.2 ή αλλιως 20%. Άρα το 20% των δεδομένων θα αποθηκευτεί στα arrays
#testX και testY που αφορούν τη κατηγορία testing και το 80% θα αποθηκευτεί στα trainX και trainY της κατηγορίας
#training. Το X αφορά τα δεδομένα του data και το Y του labels. Τα trainX και trainY θα χρησιμοποιηθούν αργότερα
#για την εκπαίδευση του μοντέλου, το οποίο αφού εκπαιδευτεί θα δοκιμαστεί με τα testX και testY, για την απόδοση, την
#αποτελεσματικότητα και το ποσοστό επιτυχίας του. Η ιδιότητα stratify δείxνει στο array των labels έτσι ώστε ο
#διαχωρισμός των δεδομένων στα trainX, testX, trainY, testY να γίνει με ομοιόμορφη κατανομή και να μην έχουμε
#σφάλματα κατα την εκπαίδευση. Εαν δεν ορίζαμε το stratify ως προς το labels τότε το πρόγραμμα μπορεί να αποθήκευε
#τυχαία στα training arrays μόνο τα δεδομένα που αντιστοιχούν σε labels=[0. 1] και έτσι αργότερα το μοντέλο να έχει
#εκπαιδευτεί μόνο για ανθρώπους που δε φοράνε μάσκες.Η ιδιότητα random_state πέρνει ως όρισμα έναν αριθμό, ο οποίος
#συνήθως είναι το 42. Αυτός ο αριθμός ορίζει την τυχαιότητα που θα χωριστούν τα δεδομένα στα arrays trainX, testX,
#trainY, testY. Αυτή η ιδιότητα βοηθάει έτσι ώστε εαν μελλοντικά θέλουμε να συγκρίνουμε διαφορετικά μοντέλα μεταξύ τους
#να είμαστε σίγουροι ότι τα δεδομένα μας χωρίστηκαν με τον ίδιο τρόπο (λογική τυχαιότητας ν. 42) για την εκπαίδευση
#όλως των υπόλοιπων μοντέλων μας.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

#Data augmentation:
aug = ImageDataGenerator(
rotation_range=20,
zoom_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.15,
horizontal_flip=True,
fill_mode="nearest")

print(type(testX))
print(testX)
print(len(testX))
print(len(trainX))
print(len(testY))
print(len(trainY))
# print(type(labels))
# print(labels)