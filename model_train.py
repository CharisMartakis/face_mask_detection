import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

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

		#Μετατροπή της εικόνας που βρίσκεται στι image σε μορφή πίνακα(array) για να μπορούμε
		#να την επεξεργαστούμε αργότερα πιο εύκολα
		image = img_to_array(image)
		print(image)
		break;
	# 	image = preprocess_input(image)
	#
	# 	data.append(image)
	# 	labels.append(dataset_class)
	break;
