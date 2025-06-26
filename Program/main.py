import os
import json
import datetime
from model import FetalHealthModel

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

def select_and_predict():
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
    health_map = {1: "Κανονική (Normal)", 2: "Ύποπτη (Suspect)", 3: "Παθολογική (Pathological)"}
    predicted_text = health_map.get(predicted_class, "Άγνωστη Κατηγορία")

    print(f"\nΧρησιμοποιώντας το μοντέλο ID: {selected_model_id}")
    print(f"Η πρόβλεψη του μοντέλου είναι: {predicted_text} (Κλάση: {predicted_class})")

def main_menu():
    """Κύριο μενού της εφαρμογής."""
    setup_environment() # Δημιουργία φακέλου αν δεν υπάρχει
    while True:
        print("\n" + "="*40)
        print("   Κύριο Μενού - Fetal Health (v2)")
        print("="*40)
        print("1. Πρόβλεψη (με επιλογή μοντέλου)")
        print("2. Εκπαίδευση νέου μοντέλου")
        print("3. Έξοδος")
        
        choice = input("Παρακαλώ επιλέξτε (1-3): ")

        if choice == '1':
            select_and_predict()
        elif choice == '2':
            train_and_save_new_model()
        elif choice == '3':
            print("Έξοδος από το πρόγραμμα.")
            break
        else:
            print("Μη έγκυρη επιλογή. Παρακαλώ προσπαθήστε ξανά.")

if __name__ == "__main__":
    main_menu()