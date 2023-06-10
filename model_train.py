import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer

#Δημιουργία μιας μεταβλητής (string type) με περιεχόμενο τη τοποθεσία των εικόνων
#που θα χρησιμοποιηθούν για την εκπαίδευση του μοντέλου
dataset_location = r"D:\projects\face_mask_detection\dataset"

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
		lb = LabelBinarizer()
		print(lb)
		#labels = lb.fit_transform(labels)
		#labels = to_categorical(labels)