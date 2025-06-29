import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# (τα υπόλοιπα imports παραμένουν)
import os
import json
import datetime
from model import FetalHealthModel
import shutil # Νέο import για διαγραφή φακέλων

# Σταθερές για εύκολη διαχείριση
MODELS_DIR = "trained_models"
EDA_PLOTS_DIR = "eda_plots" # Νέος φάκελος για τα EDA plots
REGISTRY_FILE = os.path.join(MODELS_DIR, "models_registry.json")
DATASET_CSV = 'fetal_health.csv'

def setup_environment():
    """Διασφαλίζει ότι οι απαραίτητοι φάκελοι υπάρχουν."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(EDA_PLOTS_DIR):
        os.makedirs(EDA_PLOTS_DIR)

def generate_eda_plots():
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

def plot_prediction_probabilities(probabilities, health_map, model_id):
    """Δημιουργεί και αποθηκεύει το γράφημα πιθανοτήτων."""
    labels = list(health_map.values())
    probs = list(probabilities)
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=labels, y=probs, palette="viridis")
    plt.title(f'Πιθανότητες Κλάσης για την Πρόβλεψη (Μοντέλο ID: {model_id[:8]}...)')
    plt.ylabel('Πιθανότητα')
    plt.xlabel('Κατηγορία Υγείας')
    plt.ylim(0, 1)

    # Προσθήκη τιμών πάνω από τις ράβδους
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, 
                 bar.get_height() + 0.02, 
                 f'{bar.get_height():.2%}', 
                 ha='center', 
                 color='black')

    # Δημιουργία μοναδικού ονόματος αρχείου για την εικόνα
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_probabilities_{timestamp}.png"
    # Αποθήκευση σε έναν γενικό φάκελο προβλέψεων
    predictions_dir = "prediction_plots"
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    filepath = os.path.join(predictions_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Το γράφημα πιθανοτήτων αποθηκεύτηκε στο: {filepath}")

def train_and_save_new_model():
    """Εκπαιδεύει ένα νέο μοντέλο και το αποθηκεύει με μοναδικό όνομα."""
    print("\n--- Εκπαίδευση Νέου Μοντέλου ---")
    if not os.path.exists(REGISTRY_FILE) or not load_registry():
        generate_eda_plots()

    model_trainer = FetalHealthModel()
    # Περνάμε το όνομα του φακέλου στη μέθοδο
    model_trainer.train_new_model(DATASET_CSV, MODELS_DIR) 
    
    model_filename = f"model_{model_trainer.model_id}.pkl"
    model_filepath = os.path.join(MODELS_DIR, model_filename)
    model_trainer.save_model(model_filepath)
    
    registry = load_registry()
    registry[model_trainer.model_id] = {
        "filename": model_filename,
        "trained_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_registry(registry)
    
    print(f"\nΤο νέο μοντέλο με ID: {model_trainer.model_id} εκπαιδεύτηκε και αποθηκεύτηκε.")
    print(f"Τα plots αξιολόγησης βρίσκονται στον φάκελο: {os.path.join(MODELS_DIR, model_trainer.model_id)}")

def plot_patient_comparison(patient_data, class_averages, important_features, prediction_class):
    """
    Συγκρίνει τις τιμές του ασθενή με τις μέσες τιμές της προβλεπόμενης κλάσης
    για τις πιο σημαντικές παραμέτρους.
    """
    if class_averages is None:
        return # Δεν κάνουμε τίποτα αν δεν υπάρχουν αποθηκευμένοι μέσοι όροι

    # Παίρνουμε τις μέσες τιμές για την κλάση που προέβλεψε το μοντέλο
    avg_values = class_averages.loc[prediction_class][important_features]
    patient_values = {feat: patient_data[feat] for feat in important_features}

    df_plot = pd.DataFrame({'Patient': patient_values, 'Class Average': avg_values})

    df_plot.plot(kind='bar', figsize=(12, 7), rot=45)
    plt.title(f'Σύγκριση Ασθενή με Μέσο Όρο "Κλάσης {int(prediction_class)}"')
    plt.ylabel('Τιμή')
    plt.xticks(ha='right')
    plt.tight_layout()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patient_comparison_{timestamp}.png"
    predictions_dir = "prediction_plots" # Χρησιμοποιούμε τον ίδιο φάκελο
    filepath = os.path.join(predictions_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Το γράφημα σύγκρισης αποθηκεύτηκε στο: {filepath}")

def select_and_predict():
    print("\n--- Πρόβλεψη με Επιλογή Μοντέλου ---")
    registry = load_registry()
    if not registry:
        print("Σφάλμα: Δεν υπάρχουν εκπαιδευμένα μοντέλα.")
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
    predicted_class, class_probabilities = model_predictor.predict_health_status(new_patient_data)
    health_map = {1: "Κανονική (Normal)", 2: "Ύποπτη (Suspect)", 3: "Παθολογική (Pathological)"}
    predicted_text = health_map.get(predicted_class, "Άγνωστη Κατηγορία")
    print(f"\nΧρησιμοποιώντας το μοντέλο ID: {selected_model_id}")
    print(f"Η πρόβλεψη του μοντέλου είναι: {predicted_text} (Κλάση: {predicted_class})")
    plot_prediction_probabilities(class_probabilities, health_map, selected_model_id)
    important_features_to_compare = [
        'abnormal_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'histogram_mean',
        'accelerations'
    ]
    plot_patient_comparison(new_patient_data, 
                            model_predictor.class_averages, 
                            important_features_to_compare,
                            predicted_class)


def delete_model():
    """Επιτρέπει στον χρήστη να επιλέξει και να διαγράψει ένα εκπαιδευμένο μοντέλο."""
    print("\n--- Διαγραφή Μοντέλου ---")
    registry = load_registry()
    if not registry:
        print("Δεν υπάρχουν εκπαιδευμένα μοντέλα για διαγραφή.")
        return

    print("Επιλέξτε το μοντέλο που θέλετε να διαγράψετε:")
    model_map = {}
    for i, (model_id, details) in enumerate(registry.items(), 1):
        print(f"{i}. ID: {model_id} (Εκπαιδεύτηκε: {details['trained_on']})")
        model_map[str(i)] = model_id

    choice = input(f"Επιλέξτε μοντέλο προς διαγραφή (1-{len(registry)}) ή 'q' για ακύρωση: ")
    
    if choice.lower() == 'q':
        print("Η διαγραφή ακυρώθηκε.")
        return
        
    if choice not in model_map:
        print("Μη έγκυρη επιλογή.")
        return

    # Επιβεβαίωση από τον χρήστη
    confirm = input(f"Είστε σίγουροι ότι θέλετε να διαγράψετε το μοντέλο {choice}; Αυτή η ενέργεια δεν αναιρείται. (y/n): ")
    if confirm.lower() != 'y':
        print("Η διαγραφή ακυρώθηκε.")
        return

    # Λήψη των πληροφοριών του μοντέλου
    selected_model_id = model_map[choice]
    model_details = registry[selected_model_id]
    
    try:
        # 1. Διαγραφή του αρχείου .pkl
        pkl_path = os.path.join(MODELS_DIR, model_details['filename'])
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            print(f"Διαγράφηκε το αρχείο: {pkl_path}")

        # 2. Διαγραφή του φακέλου με τα plots
        plots_dir = os.path.join(MODELS_DIR, selected_model_id)
        if os.path.exists(plots_dir):
            shutil.rmtree(plots_dir)
            print(f"Διαγράφηκε ο φάκελος: {plots_dir}")

        # 3. Αφαίρεση της εγγραφής από τον κατάλογο
        del registry[selected_model_id]
        save_registry(registry)
        print(f"Η εγγραφή για το μοντέλο {selected_model_id} αφαιρέθηκε από τον κατάλογο.")
        print("\nΗ διαγραφή ολοκληρώθηκε με επιτυχία.")

    except Exception as e:
        print(f"Προέκυψε σφάλμα κατά τη διαγραφή: {e}")

# ====================================================================
# === ΤΕΛΟΣ ΝΕΑΣ ΣΥΝΑΡΤΗΣΗΣ ===
# ====================================================================

def main_menu():
    """Κύριο μενού της εφαρμογής (ΕΝΗΜΕΡΩΜΕΝΟ)."""
    setup_environment() 
    while True:
        print("\n" + "="*40)
        print("   Κύριο Μενού - Fetal Health (v5)")
        print("="*40)
        print("1. Πρόβλεψη (με επιλογή μοντέλου)")
        print("2. Εκπαίδευση νέου μοντέλου")
        print("3. Διαγραφή υπάρχοντος μοντέλου") # <-- ΝΕΑ ΕΠΙΛΟΓΗ
        print("4. Έξοδος")
        
        choice = input("Παρακαλώ επιλέξτε (1-4): ")

        if choice == '1':
            select_and_predict()
        elif choice == '2':
            train_and_save_new_model()
        elif choice == '3':
            delete_model() # <-- ΚΑΛΕΙ ΤΗ ΝΕΑ ΣΥΝΑΡΤΗΣΗ
        elif choice == '4':
            print("Έξοδος από το πρόγραμμα.")
            break
        else:
            print("Μη έγκυρη επιλογή. Παρακαλώ προσπαθήστε ξανά.")

if __name__ == "__main__":
    main_menu()