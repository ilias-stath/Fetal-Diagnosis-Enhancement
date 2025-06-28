import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# (τα υπόλοιπα imports παραμένουν)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import uuid
import os  # Προσθήκη του os για διαχείριση φακέλων
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class FetalHealthModel:
    """
    Μια κλάση για τη διαχείριση του μοντέλου ταξινόμησης της υγείας του εμβρύου.
    Περιλαμβάνει μεθόδους για εκπαίδευση, πρόβλεψη, αποθήκευση και φόρτωση.
    """
    def __init__(self):
        """
        Αρχικοποίηση των αντικειμένων της κλάσης.
        """
        # Αποδίδουμε τα χαρακτηριστικά που ζητήθηκαν
        self.model_id = None
        self.model_name = "Fetal Health Random Forest Classifier"
        self.parameters = [] # Λίστα με τα ονόματα των παραμέτρων (features)
        
        # Εδώ θα αποθηκεύσουμε τα "values", δηλαδή τα εκπαιδευμένα αντικείμενα.
        # Είναι κρίσιμο να αποθηκεύσουμε και το scaler μαζί με το μοντέλο!
        self.model_object = None # Το ίδιο το εκπαιδευμένο μοντέλο
        self.scaler = None # Ο scaler που χρησιμοποιήθηκε για την κανονικοποίηση

    def train_new_model(self, csv_path: str, models_dir: str):
        """
        Μέθοδος 2: Δημιουργία και εκπαίδευση ενός νέου μοντέλου.
        Διαβάζει δεδομένα από ένα CSV, τα επεξεργάζεται, εκπαιδεύει το μοντέλο
        και αποθηκεύει τα εκπαιδευμένα αντικείμενα μέσα στην κλάση.
        
        Args:
            csv_path (str): Η διαδρομή προς το αρχείο fetal_health.csv.
        """
        print(f"--- Ξεκινά η εκπαίδευση του μοντέλου από το αρχείο: {csv_path} ---")
        
        # 1. Φόρτωση και αρχική επεξεργασία δεδομένων
        data = pd.read_csv(csv_path)
        data = data.dropna() # Αφαίρεση γραμμών με ελλιπή δεδομένα (αν υπάρχουν)
        
        # 2. Διαχωρισμός παραμέτρων (X) και στόχου (y)
        X = data.drop('fetal_health', axis=1)
        y = data['fetal_health']
        self.parameters = X.columns.tolist() # Αποθήκευση ονομάτων παραμέτρων

        # 3. Διαχωρισμός σε σύνολα εκπαίδευσης και δοκιμής (train/test split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Κανονικοποίηση (Scaling) των δεδομένων - ΚΡΙΣΙΜΟ ΒΗΜΑ
        # Χρησιμοποιούμε StandardScaler για να έχουν οι παράμετροι μέση τιμή 0 και διακύμανση 1.
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train) # Εκπαίδευση του scaler ΜΟΝΟ στα train data
        X_test_scaled = self.scaler.transform(X_test) # Εφαρμογή του ίδιου scaler στα test data
        
        # 5. Εκπαίδευση του μοντέλου Random Forest
        self.model_object = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Εκπαιδεύεται το Random Forest Classifier...")
        self.model_object.fit(X_train_scaled, y_train)
        
        # 6. Δημιουργία ID και αξιολόγηση
        predictions = self.model_object.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        self.model_id = str(uuid.uuid4())
# Δημιουργία ενός φακέλου ειδικά για τα plots αυτού του μοντέλου
        model_plots_dir = os.path.join(models_dir, self.model_id)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        
        print(f"\nΑποθήκευση οπτικοποιήσεων αξιολόγησης στον φάκελο: {model_plots_dir}")
        
        # 7.1 Plot: Πίνακας Σύγχυσης (Confusion Matrix)
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Suspect', 'Pathological'], 
                    yticklabels=['Normal', 'Suspect', 'Pathological'])
        plt.title('Πίνακας Σύγχυσης (Confusion Matrix)')
        plt.ylabel('Πραγματική Κλάση (Actual)')
        plt.xlabel('Προβλεπόμενη Κλάση (Predicted)')
        # Αλλάζουμε το plt.show() σε plt.savefig()
        confusion_matrix_path = os.path.join(model_plots_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close() # Κλείνουμε τη φιγούρα για να απελευθερώσουμε μνήμη

        # 7.2 Plot: Σημαντικότητα Παραμέτρων (Feature Importance)
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
        # Αλλάζουμε το plt.show() σε plt.savefig()
        feature_importance_path = os.path.join(model_plots_dir, 'feature_importance.png')
        plt.savefig(feature_importance_path)
        plt.close() # Κλείνουμε τη φιγούρα
        
        print("--- Η εκπαίδευση ολοκληρώθηκε! ---")
        print(f"Model ID: {self.model_id}")
        print(f"Ακρίβεια στο σύνολο δοκιμής: {accuracy:.4f}")
        
    def predict_health_status(self, input_data: dict):
        """
        Μέθοδος 1: Εισαγωγή στοιχείων για εξαγωγή αποτελέσματος.
        Παίρνει δεδομένα για ένα νέο περιστατικό και επιστρέφει την πρόβλεψη.

        Args:
            input_data (dict): Ένα λεξικό με τις τιμές των παραμέτρων.
                                π.χ. {'baseline value': 120, 'accelerations': 0.006, ...}

        Returns:
            int: Η πρόβλεψη (1: Normal, 2: Suspect, 3: Pathological).
        """
        if self.model_object is None or self.scaler is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί ή φορτωθεί. Καλέστε πρώτα τη μέθοδο train_new_model() ή load_model().")

        # Μετατροπή του λεξικού σε Pandas DataFrame
        df = pd.DataFrame([input_data])
        
        # Διασφάλιση ότι οι στήλες είναι στη σωστή σειρά
        df = df[self.parameters]
        
        # Εφαρμογή του ΙΔΙΟΥ scaler που χρησιμοποιήθηκε στην εκπαίδευση
        scaled_data = self.scaler.transform(df)
        
        # Πρόβλεψη
        prediction = self.model_object.predict(scaled_data)
        
        return prediction[0]

    def save_model(self, filepath: str):
        """Αποθηκεύει τα εκπαιδευμένα αντικείμενα (μοντέλο και scaler) σε ένα αρχείο."""
        if not self.model_id:
            raise ValueError("Δεν υπάρχει εκπαιδευμένο μοντέλο για αποθήκευση.")
            
        # Αποθηκεύουμε όλα τα απαραίτητα στοιχεία σε ένα λεξικό
        model_pack = {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'parameters': self.parameters,
            'model_object': self.model_object,
            'scaler': self.scaler
        }
        joblib.dump(model_pack, filepath)
        print(f"Το μοντέλο αποθηκεύτηκε με επιτυχία στο: {filepath}")

    def load_model(self, filepath: str):
        """Φορτώνει τα εκπαιδευμένα αντικείμενα από ένα αρχείο."""
        model_pack = joblib.load(filepath)
        self.model_id = model_pack['model_id']
        self.model_name = model_pack['model_name']
        self.parameters = model_pack['parameters']
        self.model_object = model_pack['model_object']
        self.scaler = model_pack['scaler']
        print(f"Το μοντέλο '{self.model_name}' (ID: {self.model_id}) φορτώθηκε από το: {filepath}")
        
    