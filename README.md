# Fetal Diagnosis Enhancement

Το **Fetal Diagnosis Enhancement** είναι μια desktop εφαρμογή γραμμένη σε Python, σχεδιασμένη για να υποστηρίζει το ιατρικό προσωπικό στην ταξινόμηση της υγείας του εμβρύου. Η εφαρμογή αναλύει δεδομένα από καρδιοτοκογραφήματα (Cardiotocography - CTG) χρησιμοποιώντας ένα μοντέλο μηχανικής μάθησης (Random Forest Classifier) για να προβλέψει την κατάσταση του εμβρύου σε μία από τις τρεις κατηγορίες: **Κανονική (Normal)**, **Ύποπτη (Suspect)** ή **Παθολογική (Pathological)**.

Το σύστημα διαθέτει ένα ολοκληρωμένο γραφικό περιβάλλον χρήστη (GUI) και υποστηρίζει δύο διακριτούς ρόλους χρηστών: **Διαχειριστή (Admin)** για τη διαχείριση των λογαριασμών και **Ιατρό (Medical)** για τις κύριες λειτουργίες πρόβλεψης και διαχείρισης μοντέλων.

## Ομάδα Ανάπτυξης

*   **Κωνσταντίνος Παπαθανασίου**
*   **Ηλίας Σταθάκος**
*   **Γιώργος Κτιστάκης**

## Βασικές Λειτουργίες

Το σύστημα είναι δομημένο γύρω από δύο βασικούς ρόλους:

### 🔑 Λειτουργίες Διαχειριστή (Admin)

Ο διαχειριστής έχει πλήρη έλεγχο στη διαχείριση των λογαριασμών χρηστών της εφαρμογής.
-   **Προβολή Όλων των Χρηστών:** Εμφάνιση μιας λίστας με όλους τους εγγεγραμμένους χρήστες και τα στοιχεία τους.
-   **Δημιουργία Νέου Χρήστη:** Δυνατότητα δημιουργίας νέων λογαριασμών (είτε Admin είτε Medical).
-   **Επεξεργασία Χρήστη:** Τροποποίηση των στοιχείων ενός υπάρχοντος χρήστη.
-   **Διαγραφή Χρήστη:** Οριστική διαγραφή ενός χρήστη από το σύστημα.

### 👨‍⚕️ Λειτουργίες Ιατρού (Medical User)

Ο ιατρός έχει πρόσβαση στις κύριες λειτουργίες που αφορούν τη μηχανική μάθηση.
-   **Εκπαίδευση Νέου Μοντέλου:** Δυνατότητα εκπαίδευσης ενός νέου μοντέλου `RandomForestClassifier` παρέχοντας ένα αρχείο `.csv` με νέα δεδομένα. Η εφαρμογή οπτικοποιεί την απόδοση του μοντέλου (Confusion Matrix, Feature Importance).
-   **Προβολή Προηγούμενων Αποτελεσμάτων:** Αναζήτηση και προβολή των αποτελεσμάτων από προηγούμενες προβλέψεις που έχουν γίνει.
-   **Εισαγωγή Τιμών για Πρόβλεψη:**
    1.  Ο χρήστης ανεβάζει ένα αρχείο `.csv` με τα δεδομένα ενός νέου περιστατικού.
    2.  Επιλέγει ένα από τα ήδη εκπαιδευμένα μοντέλα που είναι αποθηκευμένα στη βάση.
    3.  Στη συνέχεια, έχει δύο επιλογές:
        *   **Πρόβλεψη Υγείας Εμβρύου:** Εκτέλεση της πρόβλεψης με το επιλεγμένο μοντέλο. Το αποτέλεσμα εμφανίζεται μαζί με ένα γράφημα που δείχνει τις πιθανότητες για κάθε κλάση.
        *   **Διαγραφή Μοντέλου:** Δυνατότητα διαγραφής ενός εκπαιδευμένου μοντέλου από τη βάση δεδομένων.

## Τεχνολογίες & Βιβλιοθήκες

-   **Γλώσσα Προγραμματισμού:** Python 3
-   **Γραφικό Περιβάλλον (GUI):** `Tkinter` & `ttk`
-   **Βάση Δεδομένων:** MySQL Server
-   **Σύνδεση με Βάση:** `mysql-connector-python`
-   **Επεξεργασία Δεδομένων:** `pandas`
-   **Μηχανική Μάθηση:** `scikit-learn` (RandomForestClassifier, StandardScaler, train_test_split)
-   **Οπτικοποίηση Δεδομένων:** `matplotlib` & `seaborn`
-   **Ασφάλεια Κωδικών:** `bcrypt` για hashing των passwords
-   **Διαχείριση Εικόνων:** `Pillow (PIL)`

## Αρχιτεκτονική Συστήματος

Το project αποτελείται από δύο βασικά αρχεία:
-   **`gui.py`**: Περιέχει όλη τη λογική για τη δημιουργία και διαχείριση του γραφικού περιβάλλοντος χρήστη. χειρίζεται τις αλληλεπιδράσεις του χρήστη και καλεί τις κατάλληλες μεθόδους από το `Database.py`.
-   **`Database.py`**: Περιέχει τις κλάσεις που μοντελοποιούν τις οντότητες του συστήματος (`User`, `Admin`, `Medical`, `FetalHealthModel`, `Results`) καθώς και την κλάση `Database` που υλοποιεί όλη την επικοινωνία με τη βάση δεδομένων MySQL (CRUD operations, login, κ.λπ.).

## Εγκατάσταση & Ρύθμιση

Για να εκτελέσετε το project τοπικά, ακολουθήστε τα παρακάτω βήματα.

### 1. Απαιτήσεις

-   Εγκατεστημένη **Python 3.8+**.
-   Ένας ενεργός **MySQL Server**.

### 2. Ρύθμιση Project

1.  **Κλωνοποιήστε το repository:**
    ```bash
    git clone https://github.com/ilias-stath/Fetal-Diagnosis-Enhancement
    cd fetal-health-classification
    ```

2.  **Δημιουργήστε ένα virtual environment (προτείνεται):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Σε Linux/macOS
    # venv\Scripts\activate   # Σε Windows
    ```

3.  **Εγκαταστήστε τις απαραίτητες βιβλιοθήκες:**
    Δημιουργήστε ένα αρχείο `requirements.txt` με το παρακάτω περιεχόμενο:
    ```
    mysql-connector-python
    bcrypt
    matplotlib
    seaborn
    pandas
    scikit-learn
    Pillow
    ```
    Και εκτελέστε την εντολή:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Ρύθμιση Βάσης Δεδομένων

1.  Συνδεθείτε στον MySQL server σας.
2.  Δημιουργήστε τη βάση δεδομένων και έναν χρήστη (αντικαταστήστε το `'your_password'` με έναν ασφαλή κωδικό):
    ```sql
    CREATE DATABASE fedet;
    CREATE USER 'fedet_user'@'localhost' IDENTIFIED BY 'your_password';
    GRANT ALL PRIVILEGES ON fedet.* TO 'fedet_user'@'localhost';
    FLUSH PRIVILEGES;
    ```
3.  Εκτελέστε τα παρακάτω SQL scripts για να δημιουργήσετε τους απαραίτητους πίνακες:
    ```sql
    USE fedet;

    CREATE TABLE users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        fullName VARCHAR(255) NOT NULL,
        username VARCHAR(100) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        role ENUM('admin', 'medical') NOT NULL,
        telephone VARCHAR(20),
        email VARCHAR(255),
        address TEXT
    );

    CREATE TABLE administrators (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        clearance VARCHAR(255),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE medical_personnel (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        specialization VARCHAR(255),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE model (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        parameters JSON,
        maker INT,
        model_data LONGBLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (maker) REFERENCES medical_personnel(id) ON DELETE SET NULL
    );

    CREATE TABLE results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Patient_Name VARCHAR(255) NOT NULL,
        Fetal_Health VARCHAR(50),
        medical_supervisor INT,
        parameters JSON,
        model_id INT,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (medical_supervisor) REFERENCES medical_personnel(id) ON DELETE SET NULL,
        FOREIGN KEY (model_id) REFERENCES model(id) ON DELETE SET NULL
    );
    ```

4.  **Ενημερώστε τα στοιχεία σύνδεσης:**
    Ανοίξτε το αρχείο `gui.py` και εντοπίστε τη γραμμή `self.DB = DTB.Database(...)` μέσα στη συνάρτηση `login`. Αλλάξτε τα στοιχεία ώστε να ταιριάζουν με τη ρύθμισή σας:
    ```python
    self.DB = DTB.Database("localhost", "fedet_user", "your_password", "fedet")
    ```

## Εκτέλεση της Εφαρμογής

Αφού ολοκληρώσετε τη ρύθμιση, εκτελέστε την εφαρμογή από το τερματικό:
```bash
python gui.py
```
Θα εμφανιστεί η οθόνη σύνδεσης. Μπορείτε να συνδεθείτε με έναν χρήστη που έχετε δημιουργήσει ή να δημιουργήσετε τον πρώτο σας χρήστη (Admin) απευθείας από τη βάση δεδομένων για να ξεκινήσετε.
