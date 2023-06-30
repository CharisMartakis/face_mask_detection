#Προσθήκη όλων των απαραίτητων πακέτων δηλαδή libraries, modules, frameworks (βιβλιοθήκες, ενότητες, πλαίσια) ή μερος αυτών
#για να μπορέσει να εκτελεστεί ο κώδικας. Κάποια πακέτα εισάγονται ολόκληρα απλά με την εντολή "import όνομα_επιθυμητού_πακέτου".
#Σε άλλες περιπτώσεις εισάγονται μέρη αυτών που θα χρειαστούν ή κάποιες functions, methods τους. Για την εισαγωγή ενός
#μόνο μέρους του πακέτου χρησιμοποιείται η σύνταξη "from όνομα_επιθυμητού_πακέτου import όνομα_επιθυμητού_μέρους_του". Επίσης
#με την προσθήκη της εντολής "as επιθυμητό_νέο_όνομα" δίπλα απο την εντολή εισαγωγής κάποιου  πακέτου ή μερους αυτού, δίνεται η
#δυνατότητα στον κώδικα να απευθύνεται σε αυτό με ένα νέο όνομα, συνήθως πιο σαφές,απλό και σύντομο.

#Το "tensorflow" είναι ένα framework ανοικτού κώδικα (open-source code) όπου η βιβλιοθήκη του περιέχει εντολές με σκοπό τη
#δημιουργία και εκπαίδευση μοντέλων μηχανικής μάθησης (machine learning models). Επίσης όταν το tensorflow εισάγεται, του
#δίνεται το ψευδώνυμο "tf" για να καλείται μέσα στον κώδικα με μεγαλύτερη ευκολία.
import tensorflow as tf

#Εισαγωγή της συνάρτησης "preprocess_input" απο το "tensorflow.keras.applications.mobilenet_v2" module. Η συνάρτηση αυτή
#προεπεξεργάζεται τις εισαγόμενες εικόνες και τις προετοιμάζει με βάση την αρχιτεκτονική του μοντέλου MobileNetV2 που
#αποτελεί το συνελικτικό νευρωνικό δίκτυο του μοντέλου που θα εκπαιδευτεί.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

#Εισαγωγή της συνάρτησης "img_to_array" απο το "tensorflow.keras.preprocessing.image" module. Η συνάρτηση αυτή, μετατρέπει
#μια εικόνα σε διάνυσμα της βιβλιοθήκης NumPy έτσι ώστε αυτή να μπορεί να υποβληθεί σε επεξεργασία από μοντέλαμηχανικής
#εκμάθησης.
from tensorflow.keras.preprocessing.image import img_to_array

#Εισαγωγή της κλάσης "VideoStream" απο το "imutils.video" module. Η κλάση αυτή παρέχει μια απλοποιημένη διεπαφή για πρόσβαση
#σε ροές βίντεο από διάφορες πηγές, όπως web κάμερες ή αρχεία βίντεο.
from imutils.video import VideoStream

#Εισαγωγή της βιβλιοθήκης "numpy" με το ψευδώνυμο "np" που χρησιμεύει για αριθμητικούς υπολογισμούς στην Python. Επίσης
#παρέχει υποστήριξη για μεγάλους πολυδιάστατους πίνακες ή πίνακες διανυσμάτων, αφού περιέχει μια συλλογή μαθηματικών
#συναρτήσεων για την αποτελεσματική επεξεργασία τους.
import numpy as np

#To "imutils" είναι ένα module που περιέχει συναρτήσεις για την διευκόλυνση της χρήσης της βιβλιοθήκης OpenCV (Open Source
#Computer Vision). Κάποιες απο τις λειτουργίες των συναρτήσεων αυτών είναι η αλλαγή μεγέθους των εικόνων και η περιστροφή τους.
import imutils

#Το "cv2" είναι ένα module που ανήκει στην βιβλιοθήκη ανοικτού κώδικα OpenCV η οποία χρησιμοποιείται ευρέως για θέματα
#μηχανικής όρασης (computer vision) και μηχανικής εκμάθησης (machine learning).
import cv2


#									"""Μέρος 1ο - Συνάρτηση για την ανίχνευση μάσκας"""

#Δημιουργία της συνάρτησης που όταν καλείται απο τον κύριο κώδικα, λαμβάνει το στιγμιότυπο/την εικόνα του βίντεο εκείνης
#της δεδομένης στιγμής και το επεξεργάζεται για να επιστρέψει ξανά στον κύριο κώδικα κάποιες πληροφορίες. Αρχικά όταν καλείται
#δέχεται ως ορίσματα την εικόνα απο το βίντεο, το μοντέλο εντοπισμού ανθρώπινου προσώπου και το μοντέλο εντοπισμού μάσκας.
#Έπειτα γίνονται κάποιες επεξεργασίες πάνω στην εικόνα για να είναι συμβατή με τα 2 παραπάνω μοντέλα που αναφέρθηκαν και
#γίνεται πρώτα ο εντοπισμός προσώπων σε αυτή. Το πρώτο μοντέλο ελέγχει τη πιθανότητα ύπαρξης κάποιου προσώπου ή προσώπων
#και αν υπάρχει τότε αποθηκεύει αυτές τις πιθανότητες στη μεταβλητή detections. Έπειτα γίνεται σύγκριση αυτών των πιθανοτήτων
#με τον αριθμό 0.5, δηλαδή το 50%, όπου εάν αυτές είναι μεγαλύτερες απο αυτόν τότε θεωρείται οτι βρέθηκε πρόσωπο αλλιώς
#θεωρείται ότι το πρόγραμμα δεν βρήκε πρόσωπο και απλά ανίχννευσε κάτι παρόμοιο με αυτό. Για τα πρόσωπα που εντοπίστηκαν
#υπολογίζονται οι συντεταγμένες τους πάνω στο στιγμιότυπο και δημιουργείται για κάθε πρόσωπο μια νέα εικόνα μόνο με αυτό
#η οποία δέχεται κάποιες επεξεργασίες. Οι επεξεργασίες αυτές είναι αρχικά η μετατροπή της απο μορφή BGR (Blue Green Red)
#σε RGB (Red Blue Green), αλλαγή των διαστάσεων σε 224x224, μετατροπή σε διάνυσμα και κάποιες άλλες ειδικές επεξεργασίες
#απο τη μέθοδο preprocess_input. Όλες αυτές οι επεξεργασίες γίνονται για να είναι συμβατή η εικόνα κατά την εισαγωγή
#της στο μοντέλο ανίχνευσης μάσκας. Πέρα απο την εικόνα προσώπου υπολογίζονται και οι συντεταγμένες που θα τοποθετηθεί
#αργότερα το παραλληλόγραμμο γύρω απο το πρόσωπο του ανθρώπου που εντοπίστηκε. Στη συνέχεια γίνεται ο έλεγχος ύπαρξης μάσκας
#ή όχι με βάση την εικόνα προσώπου και τα αποτελέσματα που είναι σε μορφή πιθανοτήτων από το 0 έως το 1 αποθηκεύονται στη
#λίστα preds. Για κάθε εικόνα προσώπου αποθηκεύονται δυο πιθανότητες στη λίστα preds, μια για την ύπαρξη μάσκας και μια
#για την μη ύπαρξη μάσκας. Τέλος τα αποτελέσματα αυτά μαζί με τις συντεταγμένες του παραλληλογράμμου, επιστρέφονται στον
#κύριο κώδικα για την περεταίρω επεξεργασία τους.

#Με το "def" γίνεται ορισμός της συνάρτησης όπου ακριβώς δίπλα του ακολουθεί το όνομα που της δόθηκε, δηλαδή το
#"detect_and_predict_mask". Αμέσως μετά το όνομα μέσα σε παρενθέσεις δηλώνονται τα ορίσματα που θα δεχτεί η συνάρτηση
#απο τον κύριο κώδικα. Τα ορίσματα αυτά είναι ένα στιγμιότυπο (frame), το μοντέλο ανίχνευσης προσώπου (faceNet) και το
#μοντέλο ανίχνευσης μάσκας (maskNet).
def detect_and_predict_mask(frame, faceNet, maskNet):

	#Το χαρακτηριστικό (attribute) .shape επιστρέφει πληροφορίες σχετικά με το frame πάνω στο οποίο εκτελείται και με το
	#slicing [:2] που κάνει στη λίστα του δίνει εντολή να επιστραφούν μόνο οι πληροφορίες σχετικά με το ύψος και το πλάτος
	#του frame. Αυτά αποθηκεύονται σε ένα tuple (πλειάδα) που περιέχει τις μεταβλητές h και w τα οποία αντιστοιχούν στο
	#ύψος και το πλάτος
	(h, w) = frame.shape[:2]

	#Δημιουργία ενός blob (Binary Large Object) που θα περιέχει πληροφορίες σχετικά με το frame  χρησιμοποιώντας τη
	#συνάρτηση cv2.dnn.blobFromImage της βιβλιοθήκης OpenCV. Ειδικότερα γίνεται προεπεξεργασία του frame για είσοδο
	#στο μοντέλο ανίχνευσης προσώπου δίνοντας στη συνάρτηση κάποιες προδιαγραφές ως ορίσματα. Το πρώτο όρισμα  είναι το
	#ίδιο το frame, το δεύτερο είναι η τιμή 1 που αντιπροσωπεύει στη συγκεκριμένη περίπτωση τη μη σμίκρινση ή μεγέθυνση
	#του frame, το τρίτο είναι οι νέες διαστάσεις που θα πάρει το frame, δηλαδή 224x224 και το τέταρτο αντιπροσωπεύει
	#τις μέσες τιμές pixel που αφαιρούνται από την εικόνα (). Σε αυτήν την περίπτωση, το (104.0, 177.0, 123.0) αντιστοιχεί
	#στις μέσες τιμές για τα κανάλια κόκκινο, πράσινο και μπλε. Το αντικείμενο blob που θα δημιουργηθεί θα είναι τεσσάρων
	#διαστάσεων όπου η πρώτη είναι η εικόνα που περιέχει άρα ένα αφου του εισάγουμε μόνο ένα frame. Η δεύτερη περιέχει τα
	#τρία κανάλια χρώματος που έχει το frame και τα αναγνωρίζει ως BGR (Blue Green Red). Η τρίτη περιέχει το ύψος δηλαδή
	#224 και η τέταρτη το πλάτος όπου και αυτό είναι 224.
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	#Εισαγωγή του blob στο μοντέλο ανίχνευσης προσώπου με την εντολή ".setInput()".
	faceNet.setInput(blob)

	#Επεξεργασία του blob και ανίχνευση πιθανών προσώπων. Τα αποτελέσματα ανίχνευσης αποθηκεύονται στο διάνυσμα detections
	#ως πιθανότητες απο το 0 εως το 1.
	detections = faceNet.forward()

	#Δημιουργία κενής λίστας στην οποία θα αποθηκευτούν οι εικόνες των προσώπων που μπορεί να βρέθηκαν.
	faces = []

	#Δημιουργία κενής λίστας στην οποία θα αποθηκευτούν οι συντεταγμένες των παραλληλογράμμων που θα τοποθετηθούν γύρω
	#απο τα πρόσωπα που μπορεί να βρέθηκαν.
	locs = []

	#Δημιουργία κενής λίστας στην οποία θα αποθηκευτούν οι προβλέψεις για το αν εντοπίστηκε μάσκα ή όχι στα πρόσωπα που
	#μπορεί να βρέθηκαν.
	preds = []

	#Δημιουργία βρόγχου επανάληψης for που  η μεταβλητή "i" θα παίρνει με τη σειρά τις τιμές από 0 έως τον αριθμό του συνόλου
	#των προσώπων που πιθανά βρέθηκαν. Το range ορίζει τις τιμές που θα πάρει το i δηλαδή απο 0 έως "detections.shape[2]" με
	#βήμα 1. Το "detection.shape[2]" επιστρέφει τον αριθμό των προσώπων που πιθανά βρέθηκαν όπου αυτός ο αριθμός βρίσκεται
	#συγκεκριμένα αποθηκευμένος στη τρίτη διάσταση του "detections". Η πρόσβαση στη τρίτη διάσταση αυτή γίνεται με τη χρήση
	#του ".shape[2]" όπου το όρισμα "2" αντιστοιχεί στην τρίτη διάσταση αφού στη γλώσσα προγραμματισμού Python οι δείκτες μιας
	#λίστας ξεκινούν απο το 0 και όχι το 1.
	for i in range(0, detections.shape[2]):

		#Αποθήκευση της πιθανότητας (confidence score) που περιέχει η νούμερο "i" πρόβλεψη ανίχνευσης προσώπου στη μεταβλητή
		#"confidence". Συγκεκριμένα για την αναζήτηση της αποθηκευμένης αυτής πιθανότητας εκτελείται το "detections[0, 0, i, 2]"
		#όπου το πρώτο "0" σημαίνει ότι θέλουμε την πρόβλεψη απο την πρώτη εικόνα άρα και τη μοναδική έτσι κι αλλιως αφού
		#μόνο πάνω στο frame έγιναν προβλέψεις. Το δεύτερο "0" αντιπροσωπεύει ποιας κλάσης την πιθανότητα θέλουμε αλλά αφού
		#μόνο μια κλάση υπάρχει και αυτή είναι το ανθρώπινο πρόσωπο χρεισιμοποιείται το "0" που αντιστοιχεί στην πρώτη κλάση.
		#Το "i" αντιστοιχεί σε μια απο τις προβλέψεις. Το "2" είναι ο δείκτης που περιέχει την πιθανότητα που χρειάζεται να
		#αποθηκευτεί στο confidence.
		confidence = detections[0, 0, i, 2]

		#Έλεγχος του "confidence" εάν είναι μεγαλύτερο απο 0.5 δηλαδή 50% πιθανότητα. Εάν είναι, τότε σημαίνει ότι η
		#συγκεκριμένη πρόβλεψη είναι αληθής και υπάρχει πρόσωπο στο frame οπότε το πρόγραμμα θα συνεχίσει μέσα στο "if".
		#Ο κώδικας μέσα στο "if" πραγματοποιεί εύρεση των συντεταγμένων του παραλληλογράμου που θα τοποθετηθεί γύρω απο
		#το πρόσωπο και δημιουργεί μια νέα εικόνα με το πρόσωπο που βρέθηκε.
		if confidence > 0.5:

			#Γίνεται αποθήκευση των συντεταγμένων του νοητού παραλληλογράμμου γύρω απο το πρόσωπο που εντοπίστηκε οι οποίες
			#είναι αποθηκευμένες στη τέταρτη διάσταση του "detections" με δείκτες 3, 4, 5, 6 αφού το slicing του διανύσματος
			#ορίζεται ως "3:7". Επίσης αυτές οι συντεταγμένες πολλαπλασιάζονται με ένα διάνυσμα που περιέχει τις διαστάσεις
			#του "frame" για να προσαρμοστούν ως προς το μέγεθος του.
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

			#Οι τέσσερις νέες συντεταγμένες που αποθηκεύτηκαν στο "box" μετατρέπονται αρχικά σε ακέραιες(integers) τιμές
			#μέσω της μεθόδου ".astype("int")" και έπειτα αποθηκεύονται σε ένα tuple και ειδικότερα η κάθε μία σε μια
			#αντίστοιχη μεταβλητή.
			(startX, startY, endX, endY) = box.astype("int")

			#Επειδή υπάρχει η περίπτωση θεωρητικά οι συντεταγμένες να έχουν τιμές εκτός της πραγματικής εικόνας που θα
			#εμφανιστούν, γίνεται προσαρμογή τους να μην ξεπερνάνε τα όρια της εικόνας. Οπότε αρχικά για τις συντεταγμένες
			#της πάνω αριστερά γωνίας του νοητού παραλληλόγραμου υπολογίζεται η νέα τιμή τους, βρίσκοντας την μέγιστη τιμή
			#μεταξύ του 0 και της πραγματικήε τους τιμής. Εάν δηλαδή η πραγματική τους τιμή είναι αρνητική αυτή αναγκαστικά
			#θα μετατραπεί σε 0 για να ξεκινήσει να φαίνεται το παραλληλόγραμμο απο την κάτω αριστερή γωνία του "frame".
			(startX, startY) = (max(0, startX), max(0, startY))

			#Αντίστοιχα εδώ υπολογίζονται οι νέες τιμές της κάτω δεξιά γωνίας όπου βρίσκεται η μέγιστη τιμή μεταξύ των
			#πραγματικών τιμών των συντεταγμένων "endX", "endY" και των ορίων του πλάτους ή του ύψους του "frame" αντίστοιχα
			#-1. Το μείον ένα χρειάζεται επειδή το ανώτατο όριο της εικόνας είτε στο πλάτος είτε στο ύψος δεν είναι το 224
			#αλλά το 223 λόγω του προγραμματισμού σε γλώσσα Python. Αυτό μπορεί να γίνει πιο ξεκάθαρο αν σκεφτεί κανείς
			#πως στην πάνω εντολή τα όρια προσαρμόζονται απο το σημείο 0 και όχι το 1 άρα τα όρια είναι 0-223 και όχι 1-224.
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			#Εδώ γίνεται εξαγωγή ενός κομματιού του "frame" και συγκεκριμένα αυτού που περιέχει το πρόσωπο που εντοπίστηκε.
			#Οι συντεταγμένες που υπολογίστηκαν προηγουμένως συμβάλλουν σε αυτή τη διαδικασία και η νέα εικόνα του προσώπου
			#αποθηκεύεται στη μεταβλητή "face".
			face = frame[startY:endY, startX:endX]

			#Επειδή το "frame" ήταν της μορφής BGR έτσι και το "face" δημιουργήθηκε ως BGR, επειδή όμως αυτή η μορφή εικόνας
			#δεν είναι συμβατή με αυτή που θα πρέπει να ελέγξει το μοντέλο ανίχνευσης μάσκας πρέπει να αλλάξει και να μετατραπεί
			#σε μορφή RGB. Για αυτό αναλαμβάνει η συνάρτηση ".cvtColor()" που δέχεται ως πρώτο όρισμα την εικόνα "face"
			#και σαν δεύτερο όρισμα την μορφή που πρέπει να γίνει η μετατροπή.
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

			#Προσαρμογή της εικόνας "face" στις διαστάσεις 224 x 224, διότι το μοντέλο ανίχνευσης μάσκας έχει εκπαιδευτεί
			#για να δέχεται εικόνες αυτών των συγκεκριμένων διαστάσεων. Αυτό επιτυγχάνεται με την συνάρτηση ".resize()"
			#της βιβλιοθήκης OpenCV η οποία δέχεται ως ορίσματα την εικόνα "face" και τις διαστάσεις προσαρμογής.
			face = cv2.resize(face, (224, 224))

			#Μετατροπή της εικόνας "face" σε μορφή διανύσματος με τη χρήση της συνάρτησης "img_to_array()" από το
			#"tensorflow.keras.preprocessing.image" module. Αυτή η μετατροπή είναι απαράιτητη για την προετοιμασία
			#του "face" για περεταίρω επεξεργασία απο το μοντέλο ανίχνευσης μάσκας.
			face = img_to_array(face)

			#Με τη συνάρτηση "preprocess_input()" απο το "tensorflow.keras.applications.mobilenet_v2" module γίνονται
			#κάποιες τελικές μετατροπές στο "face" για να μπορεί να είναι συμβατό με την αρχιτεκτονική της mobilenet_v2
			#που είναι αυτή που χρησιμοποιήθηκε για την κατασκευή του μοντέλου ανίχνευσης μάσκας.
			face = preprocess_input(face)

			#Προσθήκη του διανύσματος "face" στη λίστα "faces" με την συνάρτηση ".append()" που προσθέτει ένα νέο διάνυσμα
			#κάθε φορά στο τέλος της λίστας.
			faces.append(face)

			#Προσθήκη στη λίστα "locs", με τη βοήθεια πάλι της εντολής ".append()", των συντεταγμένων του παραλληλογράμου
			#που θα σχεδιαστεί αργότερα γύρω απο το πρόσωπο που αντιστοιχεί στο "face" που αποθηκέυτηκε προηγουμένως στην
			#λίστα "faces".
			locs.append((startX, startY, endX, endY))

	#Έλεγχος με τη συνθήκη "if" εάν υπάρχει έστω ένα εντοπισμένο πρόσωπο στο frame. Εάν υπάρχει, τότε ο κώδικας συνεχίζει
	#μέσα στο "if" όπου ξεκινάει η ανίχνευση μάσκας. Αν δεν υπάρχει τότε ο κώδικας επιστρέφει τις κενές λίστες "locs" και
	#"preds" στον κύριο κώδικα. Στη συνθήκη "if" ελέγχεται ουσιαστικά το μήκος της λίστας "faces" εάν είναι μεγαλύτερο
	#του μηδενός, αφού εάν υπήρχε έστω και ένα "face" μέσα του το μήκος θα ήταν τουλάχιστον 1. Ο υπολογισμός του μήκους
	#της λίστας γίνεται με την εντολή "len()".
	if len(faces) > 0:

		#Δημιουργία βρόγχου επανάληψης "for" που θα ελέγξει ξεχωριστά όλα τα πρόσωπα που εντοπίστηκαν και βρίσκονται
		#μέσα στην λίστα "faces".
		for face in faces:

			#Προσθήκη στο "face" μιας επιπλέον διάστασης στην αρχή των άλλων διαστάσεων που θα αντιπροσωπεύει τον αριθμό
			#των εικόνων που περιέχει αυτό το διάνυσμα. Αυτή η μετατροπή γίνεται μόνο για λόγους συμβατότητας με το μοντέλο
			#ανίχνευσης μάσκας επειδή είναι σχεδιασμένο να δέχεται διανύσματα τεσσάρων διαστάσεων ενώ το "face" έχει τρείς.
			#Για παράδειγμα εάν το "face" έχει τη μορφή (224, 224, 3) τότε μετά την μετατροπή θα μοιάζει έτσι (1, 224, 224, 3).
			#Για την προσθήκη της νέας διάστασης χρησιμοποιείται η εντολή "np.expand_dims()" της βιβλιοθήκης "numpy"
			#που δέχεται ως πρώτο όρισμα το "face" και ως δεύτερο την τοποθεσία της νέας διάστασης με δείκτη "0" δηλαδή
			#στην πρώτη θέση.
			face = np.expand_dims(face, axis=0)

			#Με την εκτέλεση της εντολής ".set_tensor()" πάνω στο μοντέλο ανίχνευσης μάσκας ή "maskNet" γίνεται ανάθεση
			#του διανύσματος "face" στον τανυστή εισόδου του μοντέλου. Πιο συγκεκριμένα το μοντέλο θα δεχτεί στην είσοδό του
			#το "face". Το πρώτο όρισμα που δέχεται η εντολή είναι το "input_details[0]['index']" το οποίο είναι μια λίστα
			#λεξικών (Python Dictionaries) που περιέχουν πληροφορίες σχετικά με τους τανυστές εισόδου του μοντέλου. Ειδικότερα
			#με τον δείκτη "0" παρέχονται πληροφορίες απο το λεξικό για τον πρώτο τανυστή εισόδου. Το "'index'" είναι υπεύθυνο
			#για την αναγνώριση και προετοιμασία του συγκεκριμένου τανυστή εισόδου. Το δεύτερο όρισμα είναι το διάνυσμα
			#"face".
			maskNet.set_tensor(input_details[0]['index'], face)

			#Με την εντολή ".invoke()" ξεκινά η διαδικασία πρόβλεψης με βάση το διάνυσμα "face" που περιέχει το πρόσωπο
			#κάποιου ανθρώπου. Τα αποτελέσματα αποθηκεύονται στο μοντέλο.
			maskNet.invoke()

			#Me την εντολή ".get_tensor()" πάνω στο "maskNet" λαμβάνονται τα αποτελέσματα, δηλαδή οι προβλέψεις  απο τον
			#τανυστή εξόδου του μοντέλου. Αυτά τα αποτελέσματα αποθηκεύονται στη μεταβλητή "pred" και περιέχουν την πιθανότητα
			#για την ύπαρξη μάσκας άλλα και την πιθανότητα για την μη ύπαρξη μάσκας. Το ".get_tensor()" δέχεται ένα μόνο
			#όρισμα το οποίο είναι το "output_details[0]['index']" που παρομοίως με τον τανυστή εισόδου παρέχει πληροφορίες
			#και προετοιμάζει τον πρώτο τανυστή εξόδου.
			pred = maskNet.get_tensor(output_details[0]['index'])

			#Η πρόβλεψη "pred" που έγινε προστίθενται στη λίστα "preds" που θα περιέχει όλες τις προβλέψεις για όλα τα πρόσωπα
			#που εντοπίστηκαν.
			preds.append(pred)

	#Τέλος γίνεται επιστροφή στον κύριο κώδικα των "locs" και "preds" ως tuple για την τελική επεξεργασία τους και εμφάνιση
	#των αποτελεσμάτων σε live video stream.
	return (locs, preds)


#		"""Μέρος 2ο - Κύριος κώδικας, ενεργοποίησης video stream και απεικόνηση των αποτελεσμάτων σε παράθυρο"""


#Αποθήκευση της τοποθεσίας του αρχείου "deploy.prototxt" ως raw string  στη μεταβλητή "prototxtPath". Το "deploy.prototxt"
#είναι το πρώτο απο τα δύο αρχεία που αποτελούν το μοντέλο ανίχνευσης προσώπου και περιέχει πληροφορίες σχετικά με την
#αρχιτεκτονική του, τις παραμέτρους του, τα επίπεδα του και τη σύνδεση μεταξύ αυτών. To μοντέλο που χρησιμοποιείται
#ονομάζεται Single Shot MultiBox Detector (SSD) και έχει σχεδιαστεί με βάση την αρχιτεκτονική του "MobileNetV1".
prototxtPath = r"face_detector_model/deploy.prototxt"

#Αποθήκευση της τοποθεσίας του δεύτερου αρχείου του μοντέλου ανίχνευσης προσώπου "res10_300x300_ssd_iter_140000.caffemodel"
#ως raw string  στη μεταβλητή "weightsPath". Αυτό το αρχείο περιέχει πληροφορίες σχετικά με τα βάρη του μοντέλου και τις
#εκπαιδευμένες παραμέτρους του. Τα βάρη αντιπροσωπεύουν τις γνώσεις του μοντέλου για την καλή αναγνώριση διάφορων
#χαρακτηριστικών σε εικόνες. Αυτά έχουν διαμορφωθεί μετά απο εκπαίδευση του μοντέλου πάνω σε ένα τεράστιο σύνολο δεδομένων
#σχετικά με την αναγνώριση προσώπων.
weightsPath = r"face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"

#Με την εντολή "cv2.dnn.readNet()" της βιβλιοθήκης "openCV" γίνεται φόρτωση του μοντέλου αναγνώρισης προσώπου στο αντικείμενο
#"faceNet". Η εντολή δέχεται ως ορίσματα τις δύο τοποθεσίες των αρχείων που προαναφέρθηκαν ώστε αυτά να δημιουργήσουν το
#μοντέλο μετά την ένωση και την κατάλληλη επεξεργασία τους.
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#Αποθήκευση της τοποθεσίας του αρχείου "mask_detection_model_optim.tflite" ως raw string  στη μεταβλητή "mask_string". Το
#αρχείο αυτό αντιπροσωπεύει το μοντέλο που εκπαιδεύτηκε για την ανίχνευση μάσκας και έχει τη μορφή ".tflite".
mask_string = r"mask_detection_model_optim.tflite"

#Φόρτωση του αρχείου "mask_detection_model_optim.tflite" και δημιουργία του μοντέλου ανίχνευσης μάσκας με όνομα αντικειμένου
#"maskNet". Η δημιουργία του μοντέλου οφείλεται στην εντολή tf.lite.Interpreter() της βιβλιοθήκης "tensorflow" και συγκεκριμένα
#του μέρους του που ονομάζεται "tensorflow lite". Η εντολή αυτή δέχεται ως όρισμα την τοποθεσία του αρχείου του μοντέλου
#στον υπολογιστή δηλαδή το "mask_string". Η σύνταξη του ορίσματος είναι η εξής "model_path = mask_string".
maskNet = tf.lite.Interpreter(model_path = mask_string)

#Η εντολή ".allocate_tensors()" είναι απαραίτητη πριν απο οποιαδήποτε χρήση του μοντέλου στον κώδικα διότι κατανέμει στη
#μνήμη τους τανυστές εισόδου και εξόδου. Μετά απο αυτήν την εντολή το μοντέλο "maskNet" είναι έτοιμο να δεχτεί εισόδους
#και να παρέχει εξόδους.
maskNet.allocate_tensors()

#Η εντολή ".get_input_details()" παρέχει πληροφορίες σχετικά με τους τανυστές εισόδου του μοντέλου τις οποίες αποθηκεύει
#στο αντικείμενο "input_details". Πιο συγκεκριμένα θα αποθηκευτεί μια λίστα απο dictionaries όπου κάθε dictionary αντιστοιχεί
#σε ένα τανυστή εισόδου παρέχοντας πληροφορίες όπως όνομα και τύπος δεδομένων.
input_details = maskNet.get_input_details()

#Η εντολή ".get_output_details()" παρέχει πληροφορίες σχετικά με τους τανυστές εξόδου του μοντέλου τις οποίες αποθηκεύει
#στο αντικείμενο "output_details". Όπως και για τους τανυστλες εισόδου και εδώ αντίστοιχα θα αποθηκεύεται μια λίστα απο
#dictionaries όπου κάθε dictionary αντιστοιχεί σε έναν τανυστή εξόδου.
output_details = maskNet.get_output_details()

#Εκτύπωση ενημερωτικού μηνύματος στην οθόνη.
print("[ΕΝΗΜΕΡΩΣΗ] Το βίντεο ξεκίνησε...")






vs = VideoStream(src=0).start()
fl = 0

while True:
	if fl == 1:
		if cv2.getWindowProperty("Mask Detection", cv2.WND_PROP_VISIBLE) < 1:
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

		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Mask Detection", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == 27:
		break

vs.stream.release()
cv2.destroyAllWindows()