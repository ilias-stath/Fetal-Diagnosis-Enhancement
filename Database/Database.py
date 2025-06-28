import mysql.connector as sql
from mysql.connector import Error
import bcrypt
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import uuid
import os 
import shutil
import datetime
import pickle

# Σταθερές για εύκολη διαχείριση
MODELS_DIR = "trained_models" # Αυτός ο φάκελος χρησιμοποιείται πλέον κυρίως για plots
EDA_PLOTS_DIR = "eda_plots" 
# REGISTRY_FILE = os.path.join(MODELS_DIR, "models_registry.json") # Αυτό το αρχείο μητρώου δεν είναι πλέον απαραίτητο με την αποθήκευση στη βάση δεδομένων
DATASET_CSV = 'fetal_health.csv'

# Δημιουργία φακέλων αν δεν υπάρχουν
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(EDA_PLOTS_DIR):
    os.makedirs(EDA_PLOTS_DIR)


def is_json_format(variable):
    """
    Checks if a variable is a string containing valid JSON data.
    """
    if not isinstance(variable, str):
        return False
    try:
        json.loads(variable)
        return True
    except json.JSONDecodeError:
        return False

class FetalHealthModel:
    """
    Μια κλάση για τη διαχείριση του μοντέλου ταξινόμησης της υγείας του εμβρύου.
    Περιλαμβάνει μεθόδους για εκπαίδευση, πρόβλεψη, αποθήκευση και φόρτωση.
    """
    def __init__(self,id=None,name=None,parameters=None,idM=None,model_data=None): 
        """
        Αρχικοποίηση των αντικειμένων της κλάσης.
        """
        self.id = id
        self.model_name = name
        self.parameters = parameters # Λίστα με τα ονόματα των παραμέτρων (features)
        self.idM = idM # ID του χρήστη (maker) που εκπαίδευσε το μοντέλο
        # Δυαδικά δεδομένα του μοντέλου για αποθήκευση/φόρτωση από τη βάση δεδομένων
        self._model_binary_data = model_data 
        
        # Το εκπαιδευμένο μοντέλο και ο scaler
        self.model_object = None 
        self.scaler = None
        

    def storeModel(self):
        """
        Αποθηκεύει το εκπαιδευμένο μοντέλο (και scaler) ως δυαδικά δεδομένα στη βάση δεδομένων.
        """
        if self.model_object is None or self.scaler is None or self._model_binary_data is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί ή τα δυαδικά δεδομένα δεν έχουν δημιουργηθεί για αποθήκευση.")

        # Μετατροπή της λίστας παραμέτρων σε JSON string για αποθήκευση στη βάση δεδομένων
        if not is_json_format(self.parameters):
            parameters_json = json.dumps(self.parameters)

        while True:
            # Κλήση της συνάρτησης create_model_in_db για αποθήκευση του μοντέλου
            self.id = create_model_in_db(self.model_name, parameters_json, self.idM, self._model_binary_data)
            if self.id != -1:
                break

    def train_new_model(self, csv_path: str):
        """
        Μέθοδος 2: Δημιουργία και εκπαίδευση ενός νέου μοντέλου.
        Διαβάζει δεδομένα από ένα CSV, τα επεξεργάζεται, εκπαιδεύει το μοντέλο
        και αποθηκεύει τα εκπαιδευμένα αντικείμενα μέσα στην κλάση.
        
        Args:
            csv_path (str): Η διαδρομή προς το αρχείο fetal_health.csv.
        """
        print(f"--- Ξεκινά η εκπαίδευση του μοντέλου από το αρχείο: {csv_path} ---")
        
        data = pd.read_csv(csv_path)
        data = data.dropna()
        
        X = data.drop('fetal_health', axis=1)
        y = data['fetal_health']
        self.parameters = X.columns.tolist() 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model_object = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Εκπαιδεύεται το Random Forest Classifier...")
        self.model_object.fit(X_train_scaled, y_train)
        
        # Συσκευασία μοντέλου και scaler σε δυαδικά δεδομένα
        model_pack = {
            'model_object': self.model_object,
            'scaler': self.scaler,
            'parameters': self.parameters 
        }
        self._model_binary_data = pickle.dumps(model_pack) # Μετατροπή σε bytes

        # Δημιουργία μοναδικού ονόματος μοντέλου και αξιολόγηση
        self.model_name = "Fetal Health Random Forest Classifier-" + str(uuid.uuid4()) + "-" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        predictions = self.model_object.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        # Δημιουργία φακέλου για τα plots με ένα μοναδικό όνομα βασισμένο στο UUID
        # Το UUID είναι μέρος του self.model_name
        # Η λογική για την ανάκτηση αυτού του μέρους για τη διαγραφή του φακέλου βρίσκεται στη delete_model.
        uuid_part_for_folder = self.model_name[self.model_name.rfind("-") + 3:].split("-")[0]
        model_plots_dir_path = os.path.join(MODELS_DIR, "model_plots_" + uuid_part_for_folder)
        
        if not os.path.exists(model_plots_dir_path):
            os.makedirs(model_plots_dir_path)
            
        print(f"\nΑποθήκευση οπτικοποιήσεων αξιολόγησης στον φάκελο: {model_plots_dir_path}")
        
        # Plot: Πίνακας Σύγχυσης (Confusion Matrix)
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Suspect', 'Pathological'], 
                    yticklabels=['Normal', 'Suspect', 'Pathological'])
        plt.title('Πίνακας Σύγχυσης (Confusion Matrix)')
        plt.ylabel('Πραγματική Κλάση (Actual)')
        plt.xlabel('Προβλεπόμενη Κλάση (Predicted)')
        confusion_matrix_path = os.path.join(model_plots_dir_path, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()

        # Plot: Σημαντικότητα Παραμέτρων (Feature Importance)
        importances = self.model_object.feature_importances_
        feature_names = self.parameters
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('Σημαντικότητα Παραμέτρων (Feature Importance)')
        plt.xlabel('Βαθμός Σημαντικότητας')
        plt.ylabel('Παράμετρος')
        plt.tight_layout()
        feature_importance_path = os.path.join(model_plots_dir_path, 'feature_importance.png')
        plt.savefig(feature_importance_path)
        plt.close()
        
        print("--- Η εκπαίδευση ολοκληρώθηκε! ---")
        print(f"Model Name: {self.model_name}") 
        print(f"Ακρίβεια στο σύνολο δοκιμής: {accuracy:.4f}")

        # Αποθήκευση του μοντέλου στη βάση δεδομένων
        self.storeModel()
        print(f"Model ID (από τη βάση δεδομένων): {self.id}") 

    # def _load_from_db_by_id(self, model_id: int):
    #     """
    #     Φορτώνει τα δεδομένα του μοντέλου (δυαδικά) από τη βάση δεδομένων
    #     και τα αποσυμπιέζει για χρήση.
    #     """
    #     print(f"Φόρτωση μοντέλου με ID: {model_id} από τη βάση δεδομένων...")
        
    #     # Κλήση της καθολικής συνάρτησης για ανάκτηση δεδομένων μοντέλου από τη βάση
    #     model_records = get_models_from_db(model_id=model_id)
        
    #     if not model_records:
    #         raise ValueError(f"Δεν βρέθηκε μοντέλο με ID: {model_id} στη βάση δεδομένων.")
        
    #     # Λαμβάνουμε την πρώτη (και μοναδική) εγγραφή
    #     model_record = model_records[0] 

    #     # Ανάθεση των ανακτηθέντων τιμών στα χαρακτηριστικά της κλάσης
    #     self.id = model_record[0]
    #     self.model_name = model_record[1]
    #     # Μετατροπή του JSON string των παραμέτρων πίσω σε λίστα
    #     self.parameters = json.loads(model_record[2]) 
    #     self.idM = model_record[3]
    #     # Τα δυαδικά δεδομένα του μοντέλου
    #     model_data_blob = model_record[4] 

    #     # Αποσυμπίεση του model_pack από τα δυαδικά δεδομένα
    #     model_pack = joblib.loads(model_data_blob)
    #     self.model_object = model_pack['model_object']
    #     self.scaler = model_pack['scaler']
        
    #     print(f"Το μοντέλο '{self.model_name}' (ID: {self.id}) φορτώθηκε από τη βάση δεδομένων.")

    def predict_health_status(self, csv_path: str, idP):
        """
        Προβλέπει την κατάσταση υγείας ενός ασθενή χρησιμοποιώντας ένα μοντέλο
        που έχει ήδη φορτωθεί στην κλάση (από BLOB δεδομένα).
        Διαβάζει τα δεδομένα του ασθενή από ένα CSV αρχείο.

        Args:
            csv_path (str): Η διαδρομή προς το CSV αρχείο που περιέχει τα δεδομένα
                            ενός και μόνο ασθενή για πρόβλεψη.

        Returns:
            int: Η πρόβλεψη (1: Normal, 2: Suspect, 3: Pathological).
        """
        # Υποθέτουμε ότι το self.id έχει ήδη ρυθμιστεί με το ID του φορτωμένου μοντέλου
        print(f"Φόρτωση μοντέλου (ID: {self.id}) και δεδομένων ασθενή από: {csv_path}")

        # 1. Αποκωδικοποίηση του μοντέλου και του scaler από τα binary δεδομένα
        if self._model_binary_data is None:
            raise ValueError("Το δυαδικό μοντέλο (_model_binary_data) δεν είναι διαθέσιμο. Βεβαιωθείτε ότι έχει φορτωθεί από τη βάση δεδομένων πριν την κλήση αυτής της συνάρτησης.")
        
        try:
            # Χρησιμοποιούμε το pickle.loads για να αποκωδικοποιήσουμε τα binary δεδομένα
            model_pack = pickle.loads(self._model_binary_data)
            self.model_object = model_pack['model_object']
            self.scaler = model_pack['scaler']
            self.parameters = model_pack['parameters']
            print("Μοντέλο, scaler και παράμετροι αποκωδικοποιήθηκαν επιτυχώς.")
        except Exception as e:
            raise ValueError(f"Σφάλμα κατά την αποκωδικοποίηση του δυαδικού μοντέλου: {e}. Βεβαιωθείτε ότι τα δεδομένα είναι έγκυρα pickled δεδομένα.")

        # Έλεγχος ότι τα αντικείμενα φορτώθηκαν επιτυχώς
        if self.model_object is None or self.scaler is None or self.parameters is None:
            raise ValueError("Το μοντέλο ή ο scaler ή οι παράμετροι δεν φορτώθηκαν επιτυχώς από τα δυαδικά δεδομένα.")

        # 2. Διαβάζουμε το CSV αρχείο με την τυπική δομή
        try:
            # Το pd.read_csv() από προεπιλογή θα χρησιμοποιήσει την πρώτη γραμμή ως κεφαλίδες,
            # που είναι ακριβώς αυτό που θέλουμε για το patient_data.csv
            df = pd.read_csv(csv_path)

            # Προαιρετικός έλεγχος: Διασφαλίζουμε ότι το αρχείο έχει μόνο μία γραμμή δεδομένων
            if len(df) != 1:
                raise ValueError(f"Το αρχείο CSV '{csv_path}' πρέπει να περιέχει δεδομένα για έναν μόνο ασθενή (1 γραμμή δεδομένων), αλλά βρέθηκαν {len(df)} γραμμές.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Το CSV αρχείο δεν βρέθηκε στη διαδρομή: {csv_path}")
        except Exception as e:
            raise Exception(f"Σφάλμα κατά την ανάγνωση ή επεξεργασία του CSV αρχείου: {e}")

        # 3. Διασφάλιση ότι οι στήλες του DataFrame είναι στη σωστή σειρά
        if not all(col in df.columns for col in self.parameters):
            missing_cols = [col for col in self.parameters if col not in df.columns]
            raise ValueError(f"Το CSV αρχείο λείπει από απαραίτητες παραμέτρους για το μοντέλο: {missing_cols}")
        
        # Χρησιμοποιούμε το df απευθείας, αλλά το βάζουμε στη σωστή σειρά
        df_ordered = df[self.parameters]
        
        # 4. Εφαρμογή του ΙΔΙΟΥ scaler που χρησιμοποιήθηκε στην εκπαίδευση
        scaled_data = self.scaler.transform(df_ordered)
        
        # 5. Πρόβλεψη
        prediction = self.model_object.predict(scaled_data)
        
        # Το predict επιστρέφει ένα array (π.χ. [1.]), οπότε επιστρέφουμε το πρώτο στοιχείο
        if prediction[0] == 1:
            prediction = "Normal"
        elif prediction[0] == 2:
            prediction = "Suspect"
        else:
            prediction = "Pathological"

        pName = ""
        result = Results(pName,prediction,self.parameters,idP,self.id)
        result.storeResult()

        return prediction
    # def save_model(self, filepath: str):
    #     """
    #     Αποθηκεύει τα εκπαιδευμένα αντικείμενα (μοντέλο και scaler) σε ένα αρχείο.
    #     Αυτή η μέθοδος δεν χρησιμοποιείται πλέον στην κύρια ροή αποθήκευσης μοντέλων στη βάση δεδομένων,
    #     καθώς τα μοντέλα αποθηκεύονται απευθείας στη βάση.
    #     """
    #     if self.id == -1: 
    #         raise ValueError("Δεν υπάρχει εκπαιδευμένο μοντέλο για αποθήκευση.")
            
    #     model_pack = {
    #         'model_id': self.id, 
    #         'model_name': self.model_name,
    #         'parameters': self.parameters,
    #         'model_object': self.model_object,
    #         'scaler': self.scaler
    #     }
    #     joblib.dump(model_pack, filepath)
    #     print(f"Το μοντέλο αποθηκεύτηκε με επιτυχία στο: {filepath}")

    # def load_model(self, filepath: str):
    #     """
    #     Φορτώνει τα εκπαιδευμένα αντικείμενα από ένα αρχείο.
    #     Αυτή η μέθοδος δεν χρησιμοποιείται πλέον στην κύρια ροή φόρτωσης μοντέλων από τη βάση δεδομένων,
    #     καθώς τα μοντέλα φορτώνονται απευθείας από τη βάση.
    #     """
    #     model_pack = joblib.load(filepath)
    #     self.id = model_pack['model_id'] 
    #     self.model_name = model_pack['model_name']
    #     self.parameters = model_pack['parameters']
    #     self.model_object = model_pack['model_object']
    #     self.scaler = model_pack['scaler']
    #     print(f"Το μοντέλο '{self.model_name}' (ID: {self.id}) φορτώθηκε από το: {filepath}")


    def generate_eda_plots(self):
        """Δημιουργεί και αποθηκεύει όλες τις οπτικοποιήσεις EDA."""
        print("\nΔημιουργία και αποθήκευση EDA plots...")
        if not os.path.exists(DATASET_CSV):
            print(f"Σφάλμα: Το αρχείο dataset '{DATASET_CSV}' δεν βρέθηκε.")
            return

        data = pd.read_csv(DATASET_CSV)

        # 1. Κατανομή Κλάσεων
        plt.figure(figsize=(8, 6))
        sns.countplot(x='fetal_health', data=data)
        plt.title('Κατανομή Κλάσεων Υγείας Εμβρύου')
        plt.xlabel('Κατηγορία Υγείας (1: Normal, 2: Suspect, 3: Pathological)')
        plt.ylabel('Αριθμός Περιστατικών')
        plt.savefig(os.path.join(EDA_PLOTS_DIR, 'class_distribution.png'))
        plt.close()

        # 2. Πίνακας Συσχέτισης
        plt.figure(figsize=(20, 16))
        sns.heatmap(data.corr(), cmap='coolwarm')
        plt.title('Πίνακας Συσχέτισης Παραμέτρων')
        plt.savefig(os.path.join(EDA_PLOTS_DIR, 'correlation_heatmap.png'))
        plt.close()

        print(f"Τα EDA plots αποθηκεύτηκαν στον φάκελο: {EDA_PLOTS_DIR}")

    def select_and_predict(self):
        """
        Επιτρέπει στον χρήστη να επιλέξει ένα μοντέλο από τη βάση δεδομένων
        και να κάνει μια πρόβλεψη.
        """
        print("\n--- Πρόβλεψη με Επιλογή Μοντέλου ---")
        
        # Ανάκτηση όλων των μοντέλων από τη βάση δεδομένων
        all_models_data = get_models_from_db(maker_id=self.idM) # Μόνο τα μοντέλα που έφτιαξε ο τρέχων χρήστης

        if not all_models_data:
            print("Σφάλμα: Δεν υπάρχουν εκπαιδευμένα μοντέλα στη βάση δεδομένων.")
            return
        
        print("Διαθέσιμα Μοντέλα:")
        model_map = {}
        for i, model_record in enumerate(all_models_data, 1):
            model_id = model_record[0]
            model_name = model_record[1]
            print(f"{i}. ID: {model_id} - Όνομα: {model_name}")
            model_map[str(i)] = model_id
        
        choice = input(f"Επιλέξτε μοντέλο (1-{len(all_models_data)}): ")
        if choice not in model_map:
            print("Μη έγκυρη επιλογή.")
            return
        
        selected_model_id = model_map[choice]
        
        # Φόρτωση του επιλεγμένου μοντέλου απευθείας από τη βάση δεδομένων
        model_predictor = FetalHealthModel() # Δημιουργία νέας στιγμιότυπου
        try:
            model_predictor._load_from_db_by_id(selected_model_id)
        except ValueError as e:
            print(f"Σφάλμα φόρτωσης μοντέλου: {e}")
            return

        # Παράδειγμα δεδομένων ασθενούς για πρόβλεψη
        new_patient_data = {
            'baseline value': 134.0, 'accelerations': 0.0, 'fetal_movement': 0.0,
            'uterine_contractions': 0.006, 'light_decelerations': 0.003, 'severe_decelerations': 0.0,
            'prolongued_decelerations': 0.0, 'abnormal_short_term_variability': 58.0,
            'mean_value_of_short_term_variability': 1.6, 'percentage_of_time_with_abnormal_long_term_variability': 20.0,
            'mean_value_of_long_term_variability': 6.1, 'histogram_width': 53.0,
            'histogram_min': 107.0, 'histogram_max': 160.0, 'histogram_number_of_peaks': 4.0,
            'histogram_number_of_zeroes': 0.0, 'histogram_mode': 137.0, 'histogram_mean': 136.0,
            'histogram_median': 138.0, 'histogram_variance': 4.0, 'histogram_tendency': 1.0
        }
        predicted_class = model_predictor.predict_health_status(new_patient_data)
        health_map = {1: "Κανονική (Normal)", 2: "Ύποπτη (Suspect)", 3: "Παθολογική (Pathological)"}
        predicted_text = health_map.get(predicted_class, "Άγνωστη Κατηγορία")
        print(f"\nΧρησιμοποιώντας το μοντέλο ID: {selected_model_id}")
        print(f"Η πρόβλεψη του μοντέλου είναι: {predicted_text} (Κλάση: {predicted_class})")


    def delete_model(self):
        """Επιτρέπει στον χρήστη να επιλέξει και να διαγράψει ένα εκπαιδευμένο μοντέλο."""
        print("\n--- Διαγραφή Μοντέλου ---")
        
        # Λίστα μοντέλων από τη βάση δεδομένων
        all_models_data = get_models_from_db(maker_id=self.idM) # Μόνο τα μοντέλα που έφτιαξε ο τρέχων χρήστης
        if not all_models_data:
            print("Δεν υπάρχουν εκπαιδευμένα μοντέλα για διαγραφή.")
            return

        print("Επιλέξτε το μοντέλο που θέλετε να διαγράψετε:")
        model_map = {}
        for i, model_record in enumerate(all_models_data, 1):
            model_id = model_record[0]
            model_name = model_record[1]
            print(f"{i}. ID: {model_id} - Όνομα: {model_name}")
            model_map[str(i)] = model_id

        choice = input(f"Επιλέξτε μοντέλο προς διαγραφή (1-{len(all_models_data)}) ή 'q' για ακύρωση: ")
        
        if choice.lower() == 'q':
            print("Η διαγραφή ακυρώθηκε.")
            return
            
        if choice not in model_map:
            print("Μη έγκυρη επιλογή.")
            return

        selected_model_id = model_map[choice]
        
        confirm = input(f"Είστε σίγουροι ότι θέλετε να διαγράψετε το μοντέλο ID: {selected_model_id}; Αυτή η ενέργεια δεν αναιρείται. (y/n): ")
        if confirm.lower() != 'y':
            print("Η διαγραφή ακυρώθηκε.")
            return

        try:
            # 1. Διαγραφή μοντέλου από τη βάση δεδομένων
            delete_model_from_db(selected_model_id)
            print(f"Διαγράφηκε το μοντέλο ID: {selected_model_id} από τη βάση δεδομένων.")

            # 2. Διαγραφή του φακέλου με τα plots
            # Ανακτάμε το όνομα του μοντέλου από τη βάση δεδομένων για να βρούμε το φάκελο των plots
            model_details_for_deletion = next((record for record in all_models_data if record[0] == selected_model_id), None)
            if model_details_for_deletion:
                model_name_from_db = model_details_for_deletion[1] # Όνομα μοντέλου
                # Προσπαθούμε να εξάγουμε το UUID-like μέρος από το όνομα του μοντέλου
                uuid_part_start = model_name_from_db.rfind(" - ")
                if uuid_part_start != -1:
                    # Υποθέτουμε τη μορφή "Όνομα μοντέλου - UUID - Ημερομηνία Ώρα"
                    uuid_like_part = model_name_from_db[uuid_part_start + 3:].split(" - ")[0] 
                    # Ανακατασκευάζουμε το όνομα του φακέλου των plots με βάση το μοτίβο
                    plots_dir_to_delete = os.path.join(MODELS_DIR, "model_plots_" + uuid_like_part)
                    if os.path.exists(plots_dir_to_delete):
                        shutil.rmtree(plots_dir_to_delete)
                        print(f"Διαγράφηκε ο φάκελος των plots: {plots_dir_to_delete}")
                    else:
                        print(f"Ο φάκελος των plots '{plots_dir_to_delete}' δεν βρέθηκε.")
                else:
                    print("Αδυναμία εύρεσης ονόματος φακέλου plots από το όνομα του μοντέλου.")


            print("\nΗ διαγραφή ολοκληρώθηκε με επιτυχία.")

        except Exception as e:
            print(f"Προέκυψε σφάλμα κατά τη διαγραφή: {e}")
        



class User:
    def __init__(self, fullName, userName, password, role, telephone, email, address, description):
        self.id = -1
        self.fullName = fullName
        self.userName = userName
        self.password = password
        self.role = role
        self.telephone = telephone
        self.email = email
        self.address = address
        self.description = description

    def storeUser(self):
        idP = -1
        while True:
            self.id, idP, self.password = createUser(self.fullName, self.userName, self.password, self.role, self.telephone, self.email, self.address, self.description)
            if self.id != -1 and idP != -1:
                break
        return idP
    
    def print(self):
        print(f"idU -> {self.id} , fN -> {self.fullName} , uN -> {self.userName} , role -> {self.role}\n")

    




class Admin(User):
    def __init__(self, fullName, userName, password, role, telephone, email, address, description, idP, id):
        super().__init__(fullName, userName, password, role, telephone, email, address, description)
        self.idP = idP
        self.id = id
        if self.idP == -1 or self.id == -1:
            self.idP = super().storeUser()


    
    # def __del__(self):
    #     deleteUser(self.id,self.idP,self.role)
    #     print("User deleted successfully\n")

    def delete(self,idU):
        deleteUser(idU)
        print("User deleted successfully\n")


    def getUsers(self, fullName, id):
        rows = getUsers(fullName, id)

        conn = connect()
        cursor = conn.cursor()

        if rows == -1:
            return []

        users = []
        for row in rows:

            if row[4] == "admin":
                query = 'SELECT clearance FROM administrators WHERE user_id = %s'
            else:
                query = 'SELECT specialization FROM medical_personnel WHERE user_id = %s'

            cursor.execute(query, (row[0],))
            result = cursor.fetchone()

            if result is None:
                result[0] = ""


            user = User(
                fullName=row[1],
                userName=row[2],
                password=row[3],
                role=row[4],
                telephone=row[5],
                email=row[6],
                address=row[7],
                description=result[0]
            )
            user.id = row[0] 
            users.append(user)

        cursor.close()
        conn.close()
        return users
    
    def updateUser(self, userObj, updates: dict):
        updateUserInfo(userObj, updates)


    
    def printy(self):
        print(f"idP -> {self.idP}")
        super().print( )





class Medical(User):
    def __init__(self, fullName, userName, password, role, telephone, email, address, description, idP, id):
        super().__init__(fullName, userName, password, role, telephone, email, address, description)
        self.idP = idP
        self.id = id
        if self.idP == -1 or self.id == -1:
            self.idP = super().storeUser()


    # def delete(self):
    #     deleteUser(self.id,self.idP,self.role)
    #     print("User deleted successfully\n")


    def getResults(self, pName):
        raw_results = getResults(pName, self.idP)

        if raw_results == -1:
            return []

        results_list = []
        for row in raw_results:
            result = Results(
                patientName=row[1],
                fetalHealth=row[2],
                parameters=row[4],
                idMedical=row[3],
                idMo=row[6]
            )
            result.id = row[0]
            results_list.append(result)

        return results_list
    


    def getModels(self,modelID,name):
        raw_results = get_models_from_db(modelID,name, self.idP)

        if raw_results == -1:
            return []

        conn = connect()
        cursor = conn.cursor()

        results_list = []
        for row in raw_results:
            model = FetalHealthModel(
                id = row[0],
                name = row[1],
                parameters=row[2],
                idM=row[3],
                model_data=row[4]
            )

            query = "SELECT user_id FROM medical_personnel WHERE id = %s"

            cursor.execute(query, (model.idM,))
            result = cursor.fetchone()

            model.idM = result[0]

            results_list.append(model)

        cursor.close()
        conn.close()


        return results_list


    def printy(self):
        print(f"idP -> {self.idP}")
        super().print( )






class Results:
    def __init__(self,patientName,fetalHealth,parameters,idMedical,idMo):
        self.id = -1
        self.patientName = patientName
        self.fetalHealth = fetalHealth
        self.parameters = parameters
        self.idMedical = idMedical
        self.idMo = idMo


    def storeResult(self):
        while True:
            self.id = postResults(self.idMedical,self.patientName,self.fetalHealth,self.parameters,self.idMo)
            if self.id != -1:
                break

    def __str__(self):
        return f"Result(id={self.id}, patientName='{self.patientName}', fetalHealth={self.fetalHealth}, parameters='{self.parameters}', idMedical={self.idMedical})"

    def __repr__(self):
        return self.__str__()



# -----------------------------------------------------------
# Καθολικές Συναρτήσεις Βάσης Δεδομένων
# -----------------------------------------------------------

def connect():
    try:
        conn = sql.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "fedet"
        )
        return conn

    except Error as e:
        print("Connection failed:", e)
        return None


def login(username,password):
         
    conn = connect()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM users WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchone()

    if result is None:
        print(result)
        return 'Wrong'
    
    idU = result[0]
    fN = result[1]
    passwordDB = result[3].encode('utf-8')
    role = result[4]
    telephone = result[5]
    email = result[6]
    adress = result[7]

    print(result)
    
    if bcrypt.checkpw(password.encode('utf-8'), passwordDB):
        if role == "admin":
            query = 'SELECT id,clearance FROM administrators WHERE user_id = %s'
        else:
            query = 'SELECT id,specialization FROM medical_personnel WHERE user_id = %s'

        cursor.execute(query, (idU,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        idP = result[0]
        description = result[1]

        if role == "admin":
            user = Admin(fN,username,password,role,telephone,email,adress,description,idP,idU)
        else:
            user = Medical(fN,username,password,role,telephone,email,adress,description,idP,idU)

        return user
    else:
        cursor.close()
        conn.close()
        return 'Wrong'


def getResults(pName,idM):
         
    conn = connect()
    cursor = conn.cursor()

    if pName.strip(): 
        query = 'SELECT * FROM results WHERE Patient_Name = %s AND medical_supervisor = %s'
        cursor.execute(query, (pName, idM,))
    else:  
        query = 'SELECT * FROM results WHERE medical_supervisor = %s'
        cursor.execute(query, (idM,))

    result = cursor.fetchall()
    cursor.close()
    conn.close()

    if not result:
        return -1

    return result


def postResults(idP,pName,fH,parameters,idMo): # Αφαίρεση image1, image2 από εδώ, καθώς δεν χρησιμοποιούνται στην INSERT
         
    conn = connect()
    cursor = conn.cursor()

    query = """
        INSERT INTO results (Patient_Name, Fetal_Health, medical_supervisor, parameters, model_id) 
        VALUES (%s, %s, %s, %s, %s)
    """

    data = (pName,fH,idP,json.dumps(parameters),idMo)
    cursor.execute(query, data)
    conn.commit()

    id = cursor.lastrowid

    cursor.close()
    conn.close()

    print("Result inserted successfully!")

    return id


def createUser(fullName, username, password, role, telephone, email, adress, description):
    
    conn = connect()
    cursor = conn.cursor()

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    query = """
        INSERT INTO users (fullName, username, password, role, telephone, email, address) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    data = (fullName, username, hashed_password.decode('utf-8'), role, telephone, email, adress)
    cursor.execute(query, data)
    conn.commit()


    query = 'SELECT id FROM users WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchone()

    if result is None:
        return -1
    
    idU = result[0]


    if role == "admin":
        query = """
            INSERT INTO administrators (user_id, clearance) 
            VALUES (%s, %s)
        """
    else:
        query = """
            INSERT INTO medical_personnel (user_id, specialization) 
            VALUES (%s, %s)
        """

    data = (idU,description)
    cursor.execute(query, data)
    conn.commit()

    if role == "admin":
        query = 'SELECT id FROM administrators WHERE user_id = %s'
    else:
        query = 'SELECT id FROM medical_personnel WHERE user_id = %s'

    cursor.execute(query, (idU,))
    result = cursor.fetchone()

    if result is None:
        return -1

    idP = result[0]

    cursor.close()
    conn.close()

    print("User inserted successfully.")
    return idU,idP,hashed_password


def getUsers(fullName,id):
         
    conn = connect()
    cursor = conn.cursor()

    if fullName.strip() and id != -1: 
        query = 'SELECT * FROM users WHERE fullName = %s AND id = %s'
        cursor.execute(query, (fullName, id,))
    elif id != -1:  
        query = 'SELECT * FROM users WHERE id = %s'
        cursor.execute(query, (id,))
    elif fullName.strip():
        query = 'SELECT * FROM users WHERE fullName = %s'
        cursor.execute(query, (fullName,))
    else:
        query = 'SELECT * FROM users'
        cursor.execute(query)

    result = cursor.fetchall()
    cursor.close()
    conn.close()

    if not result:
        return -1

    return result


def updateUserInfo(userObj, updates: dict):

    print(updates)

    for field, value in updates.items():
        if field == "password":
            hashed_password = bcrypt.hashpw(value.encode('utf-8'), bcrypt.gensalt())
            value = hashed_password.decode('utf-8')
            updates[field] = value 

        if field != "description":
            if hasattr(userObj, field):
                setattr(userObj, field, value)

    if "description" in updates:
        userObj.description = updates["description"]
        del updates["description"]


    conn = connect()
    cursor = conn.cursor()


    if updates:
        set_clause = ", ".join(f"{field} = %s" for field in updates.keys())
        values = list(updates.values())
        values.append(userObj.id) 

        query = f"UPDATE users SET {set_clause} WHERE id = %s"

        cursor.execute(query, values)
        conn.commit()

    else:
        print("No fields to update in users.")


    if userObj.role == "admin":
        query = "UPDATE administrators SET clearance = %s WHERE user_id = %s"
    elif userObj.role == "medical":
        query = "UPDATE medical_personnel SET specialization = %s WHERE user_id = %s"
    else:
        query = None

    if query:
        cursor.execute(query, (userObj.description, userObj.id))
        conn.commit()

    cursor.close()
    conn.close()

    print(f"User (id={userObj.id}) updated successfully.")

    
def deleteUser(idU):
    conn = connect()
    cursor = conn.cursor()


    query = 'SELECT role FROM users WHERE id = %s'
    cursor.execute(query, (idU,))
    result = cursor.fetchone()

    role = result[0]

    query = "DELETE FROM users WHERE id = %s"
    cursor.execute(query, (idU,))
    conn.commit()

    query = "ALTER TABLE users AUTO_INCREMENT = %s"
    cursor.execute(query,(idU-1,))
    conn.commit()

    if role == "admin":
        query = "ALTER TABLE administrators AUTO_INCREMENT = %s"
    else:
        query = "ALTER TABLE medical_personnel AUTO_INCREMENT = %s"

    cursor.execute(query,(idU-1,))
    conn.commit()

    print("User deleted successfully\n")


def create_model_in_db(name, parameters_json_string, maker_id, model_data_blob):
    """
    Δημιουργεί μια νέα εγγραφή μοντέλου στη βάση δεδομένων με τα δυαδικά δεδομένα του μοντέλου.
    """
    conn = connect()
    cursor = conn.cursor()

    query = "SELECT id FROM medical_personnel WHERE user_id = %s"

    cursor.execute(query, (maker_id,))
    result = cursor.fetchone()

    maker_id = result[0]


    query = """
        INSERT INTO model (name, parameters, maker, model_data) 
        VALUES (%s, %s, %s, %s)
    """

    data = (name, parameters_json_string, maker_id, model_data_blob)
    cursor.execute(query, data)
    conn.commit()

    model_id = cursor.lastrowid

    cursor.close()
    conn.close()

    print(f"Model '{name}' inserted successfully with ID: {model_id}!")

    return model_id


def get_models_from_db(model_id=None, name=None, maker_id=None):
    """
    Ανακτά μοντέλα από τη βάση δεδομένων. Μπορεί να φιλτράρει με ID, όνομα ή ID κατασκευαστή.
    Επιστρέφει μια λίστα από tuples, όπου κάθε tuple περιέχει (id, name, parameters, maker, model_data).
    """
    conn = connect()
    cursor = conn.cursor()

    # query = "SELECT id FROM medical_personnel WHERE user_id = %s"

    # print(maker_id)

    # cursor.execute(query, (maker_id,))
    # result = cursor.fetchone()

    # maker_id = result[0]

    # print(maker_id)

    query = "SELECT id, name, parameters, maker, model_data FROM model WHERE 1=1"
    params = []

    if model_id is not None and model_id != -1:
        query += " AND id = %s"
        params.append(model_id)
    if name is not None and name.strip():
        query += " AND name LIKE %s"
        params.append(f"%{name}%")
    if maker_id is not None and maker_id != -1:
        query += " AND maker = %s"
        params.append(maker_id)

    cursor.execute(query, tuple(params))
    result = cursor.fetchall()
    

    cursor.close()
    conn.close()


    if not result:
        return []

    return result

def delete_model_from_db(model_id):
    """
    Διαγράφει μια εγγραφή μοντέλου από τη βάση δεδομένων με βάση το ID.
    """
    conn = connect()
    cursor = conn.cursor()

    query = "DELETE FROM model WHERE id = %s"
    cursor.execute(query, (model_id,))
    conn.commit()

    query = "ALTER TABLE model AUTO_INCREMENT = %s"
    cursor.execute(query,(model_id-1,))
    conn.commit()

    cursor.close()
    conn.close()
    print(f"Model ID: {model_id} deleted from database.")


# parameters = {
#     "parameter1": 100,
#     "parameter2": 100,
#     "parameter3": 90
# }
# user = login('george_ktist','123456789')
# user.printy()
# print(postResults("2","test subject 3","Normal",parameters))
# print(getData('test subject 1', '1'))

# print(getResults("",2))

# Admin = login("ilias_stath","123456789")
# users_list = Admin.getUsers("agasdgd",-1)
# for user in users_list:
    
#     # Update user info using admin
#     Admin.updateUser(user, {
#         "email": "ktist@",
#         "description": "A9",
#     })

# Admin.delete(5)


#-----Do the hash for every knew user


# conn = connect()
# cursor = conn.cursor()

# plain_password = "asd1sdd"
# hashed_password = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())

# query = """
#     INSERT INTO users (fullName, username, password, role, telephone, email, address) 
#     VALUES (%s, %s, %s, %s, %s, %s, %s)
# """

# data = ("agasdgd", "gasdqwef", hashed_password.decode('utf-8'), "admin", "+306916644999", "ecew@uowm.gr", "koza13ni")
# cursor.execute(query, data)
# conn.commit()

# print("User inserted successfully.")

# # Example SELECT query
# select_query = "SELECT * FROM users"
# cursor.execute(select_query)

# # Fetch all rows
# rows = cursor.fetchall()

# for row in rows:
#     print(row)

# # Close cursor and connection
# cursor.close()
# conn.close()
# # print("Connect",conn)

# lol = login("george_ktist","123456789")
# users_list = lol.getModels(-1,"")
# for user in users_list:
#     # Update user info using admin
#     print(user.id,user.model_name,user.idM)


#ALTER TABLE users AUTO_INCREMENT = 1;
