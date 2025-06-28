import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# (τα υπόλοιπα imports παραμένουν)
import os
import json
import datetime
from model import FetalHealthModel
import Database as DB

# Σταθερές για εύκολη διαχείριση
MODELS_DIR = "trained_models"  # Ένας φάκελος για να αποθηκεύουμε τα μοντέλα
REGISTRY_FILE = os.path.join(MODELS_DIR, "models_registry.json")
DATASET_CSV = 'fetal_health.csv'

def setup_environment():
    """Διασφαλίζει ότι ο φάκελος για τα μοντέλα υπάρχει."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

def load_registry():
    """Φορτώνει τον κατάλογο των μοντέλων. Αν δεν υπάρχει, επιστρέφει ένα άδειο λεξικό."""
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_registry(registry):
    """Αποθηκεύει τον κατάλογο των μοντέλων."""
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)

def train_and_save_new_model():
    """Εκπαιδεύει ένα νέο μοντέλο και το αποθηκεύει με μοναδικό όνομα."""
    print("\n--- Εκπαίδευση Νέου Μοντέλου ---")
    if not os.path.exists(DATASET_CSV):
        print(f"Σφάλμα: Το αρχείο dataset '{DATASET_CSV}' δεν βρέθηκε.")
        return

    model_trainer = FetalHealthModel()
    model_trainer.train_new_model(DATASET_CSV)
    
    # Δημιουργία μοναδικού ονόματος αρχείου βάσει του ID του μοντέλου
    model_filename = f"model_{model_trainer.model_id}.pkl"
    model_filepath = os.path.join(MODELS_DIR, model_filename)
    model_trainer.save_model(model_filepath)
    
    # Ενημέρωση του καταλόγου
    registry = load_registry()
    registry[model_trainer.model_id] = {
        "filename": model_filename,
        "name": model_trainer.model_name,
        "trained_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_registry(registry)
    
    print(f"\nΤο νέο μοντέλο με ID: {model_trainer.model_id} εκπαιδεύτηκε και αποθηκεύτηκε.")

def select_and_predict(pName, new_patient_data,idM):
    """Επιτρέπει στον χρήστη να επιλέξει ένα μοντέλο και να κάνει πρόβλεψη."""
    print("\n--- Πρόβλεψη με Επιλογή Μοντέλου ---")
    registry = load_registry()
    if not registry:
        print("Σφάλμα: Δεν υπάρχουν εκπαιδευμένα μοντέλα. Παρακαλώ εκπαιδεύστε ένα πρώτα (Επιλογή 2).")
        return

    print("Διαθέσιμα Μοντέλα:")
    model_map = {}
    for i, (model_id, details) in enumerate(registry.items(), 1):
        print(f"{i}. ID: {model_id} (Εκπαιδεύτηκε: {details['trained_on']})")
        model_map[str(i)] = model_id

    choice = input(f"Επιλέξτε μοντέλο (1-{len(registry)}): ")
    

    if choice not in model_map:
        print("Μη έγκυρη επιλογή.")
        return
        
    selected_model_id = model_map[choice]
    selected_model_filename = registry[selected_model_id]['filename']
    model_filepath = os.path.join(MODELS_DIR, selected_model_filename)

    model_predictor = FetalHealthModel()
    model_predictor.load_model(model_filepath)

    # Δεδομένα για το pilot/demo περιστατικό
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
    health_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    predicted_text = health_map.get(predicted_class, "Άγνωστη Κατηγορία")

    print(f"\nΧρησιμοποιώντας το μοντέλο ID: {selected_model_id}")
    print(f"Η πρόβλεψη του μοντέλου είναι: {predicted_text} (Κλάση: {predicted_class})")

    result = DB.Results(
                patientName=pName,
                fetalHealth=predicted_text,
                parameters=new_patient_data,
                idMedical=idM,
                image1=image1, # Create png from plots
                image2=image2,
                idMo=selected_model_id # Use the id of the database
            )
    
    result.storeResult()
    


def plot_class_distribution():
    """Plot: Κατανομή Κλάσεων"""
    print("\nΔημιουργία γραφήματος: Κατανομή Κλάσεων...")
    data = pd.read_csv(DATASET_CSV)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='fetal_health', data=data)
    plt.title('Κατανομή Κλάσεων Υγείας Εμβρύου')
    plt.xlabel('Κατηγορία Υγείας (1: Normal, 2: Suspect, 3: Pathological)')
    plt.ylabel('Αριθμός Περιστατικών')
    plt.show()

def plot_correlation_heatmap():
    """Plot: Πίνακας Συσχέτισης"""
    print("\nΔημιουργία γραφήματος: Πίνακας Συσχέτισης...")
    data = pd.read_csv(DATASET_CSV)
    plt.figure(figsize=(20, 16))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm') # annot=False για ταχύτητα/καθαρότητα
    plt.title('Πίνακας Συσχέτισης Παραμέτρων')
    plt.show()

def plot_feature_distribution():
    """Plot: Κατανομή Παραμέτρου ανά Κλάση (Box Plot)"""
    data = pd.read_csv(DATASET_CSV)
    feature_name = input("Δώστε το ακριβές όνομα της παραμέτρου που θέλετε να αναλύσετε (π.χ. 'abnormal_short_term_variability'): ")
    if feature_name not in data.columns:
        print("Σφάλμα: Αυτό το όνομα παραμέτρου δεν υπάρχει.")
        return
    
    print(f"\nΔημιουργία γραφήματος: Κατανομή της '{feature_name}'...")
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='fetal_health', y=feature_name, data=data)
    plt.title(f'Κατανομή της παραμέτρου "{feature_name}" ανά Κλάση')
    plt.xlabel('Κατηγορία Υγείας')
    plt.ylabel(f'Τιμή της {feature_name}')
    plt.show()

def eda_submenu():
    """Υπο-μενού για τις οπτικοποιήσεις ανάλυσης δεδομένων."""
    if not os.path.exists(DATASET_CSV):
        print(f"Σφάλμα: Το αρχείο dataset '{DATASET_CSV}' δεν βρέθηκε.")
        return
        
    while True:
        print("\n--- Μενού Ανάλυσης Δεδομένων (EDA) ---")
        print("1. Γράφημα Κατανομής Κλάσεων")
        print("2. Πίνακας Συσχέτισης (Heatmap)")
        print("3. Γράφημα Κατανομής Παραμέτρου (Box Plot)")
        print("4. Επιστροφή στο Κύριο Μενού")
        
        choice = input("Επιλέξτε γράφημα (1-4): ")
        if choice == '1':
            plot_class_distribution()
        elif choice == '2':
            plot_correlation_heatmap()
        elif choice == '3':
            plot_feature_distribution()
        elif choice == '4':
            break
        else:
            print("Μη έγκυρη επιλογή.")
            
# ====================================================================
# === ΤΕΛΟΣ ΝΕΩΝ ΣΥΝΑΡΤΗΣΕΩΝ ===
# ====================================================================

def main_menu():
    """Κύριο μενού της εφαρμογής (ΕΝΗΜΕΡΩΜΕΝΟ)."""
    setup_environment() 
    while True:
        print("\n" + "="*40)
        print("   Κύριο Μενού - Fetal Health (v3 with Plots)")
        print("="*40)
        print("1. Πρόβλεψη (με επιλογή μοντέλου)")
        print("2. Εκπαίδευση νέου μοντέλου")
        print("3. Ανάλυση Δεδομένων & Οπτικοποιήσεις (EDA)") # <-- ΝΕΑ ΕΠΙΛΟΓΗ
        print("4. Έξοδος") # <-- ΕΓΙΝΕ 4
        
        choice = input("Παρακαλώ επιλέξτε (1-4): ")

        if choice == '1':
            select_and_predict()
        elif choice == '2':
            train_and_save_new_model()
        elif choice == '3':
            eda_submenu() # <-- ΚΑΛΕΙ ΤΟ ΝΕΟ ΥΠΟ-ΜΕΝΟΥ
        elif choice == '4':
            print("Έξοδος από το πρόγραμμα.")
            break
        else:
            print("Μη έγκυρη επιλογή. Παρακαλώ προσπαθήστε ξανά.")

if __name__ == "__main__":
    main_menu()