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
import uuid
import os 
import shutil
import datetime
import pickle

# Σταθερές για εύκολη διαχείριση

DATASET_CSV = 'fetal_health.csv'
PREDICTION_PLOTS_DIR = 'prediction_plots'

# Δημιουργία φακέλων αν δεν υπάρχουν

if not os.path.exists(PREDICTION_PLOTS_DIR): 
    os.makedirs(PREDICTION_PLOTS_DIR)



class FetalHealthModel:
    """
    Μια κλάση για τη διαχείριση του μοντέλου ταξινόμησης της υγείας του εμβρύου.
    Περιλαμβάνει μεθόδους για εκπαίδευση, πρόβλεψη, αποθήκευση και φόρτωση.
    """
    def __init__(self,id=None,name=None,parameters=None,idM=None,model_data=None,DB=None): 
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
        self.class_averages = None # Νέα μεταβλητή για τους μέσους όρους
        self.DB = DB
        print(self.DB)


    def clear_folder(self,folder_path):
        """Διαγράφει όλα τα αρχεία και τους υποφακέλους μέσα σε έναν φάκελο."""
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Αποτυχία διαγραφής του {file_path}.')


    def is_json_format(self,variable):
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



    def plot_prediction_probabilities(self,probabilities, model_name):
        """Δημιουργεί και αποθηκεύει το γράφημα πιθανοτήτων."""
        health_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}
        labels = list(health_map.values())
        probs = list(probabilities)
        
        plt.figure(figsize=(10, 6))
        bars = sns.barplot(x=labels, y=probs, hue=labels, palette="viridis", legend=False)
        plt.title(f'Πιθανότητες Κλάσης για την Πρόβλεψη\n(Μοντέλο: {model_name[:50]}...)')
        plt.ylabel('Πιθανότητα')
        plt.xlabel('Κατηγορία Υγείας')
        plt.ylim(0, 1)

        for bar in bars.patches:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{bar.get_height():.2%}', ha='center', color='black')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_probabilities_{timestamp}.png"
        filepath = os.path.join(PREDICTION_PLOTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Το γράφημα πιθανοτήτων αποθηκεύτηκε στο: {filepath}")
        

    def storeModel(self):
        """
        Αποθηκεύει το εκπαιδευμένο μοντέλο (και scaler) ως δυαδικά δεδομένα στη βάση δεδομένων.
        """
        if self.model_object is None or self.scaler is None or self._model_binary_data is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί ή τα δυαδικά δεδομένα δεν έχουν δημιουργηθεί για αποθήκευση.")

        # Μετατροπή της λίστας παραμέτρων σε JSON string για αποθήκευση στη βάση δεδομένων
        if not self.is_json_format(self.parameters):
            parameters_json = json.dumps(self.parameters)

        while True:
            # Κλήση της συνάρτησης create_model_in_db για αποθήκευση του μοντέλου
            self.id = self.DB.create_model_in_db(self.model_name, parameters_json, self.idM, self._model_binary_data)
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

        ### ΑΛΛΑΓΗ: Υπολογισμός μέσων όρων ###
        print("Υπολογισμός μέσων όρων ανά κλάση...")
        try:
            self.class_averages = data.groupby('fetal_health').mean()
        except KeyError as err :
            return -1
        
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
            'parameters': self.parameters,
            'class_averages': self.class_averages 
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
        model_plots_dir_path = os.path.join("model_plots",uuid_part_for_folder)
        
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
        return 1


    def predict_health_status(self, csv_path: str, idP, pName):

        print("predict")
        print(self.DB)
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
            self.class_averages = model_pack.get('class_averages') 
            print("Μοντέλο και εξαρτήσεις αποκωδικοποιήθηκαν επιτυχώς.")
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

        # patient_data_dict = df.iloc[0].to_dict()

        # 3. Διασφάλιση ότι οι στήλες του DataFrame είναι στη σωστή σειρά
        if not all(col in df.columns for col in self.parameters):
            missing_cols = [col for col in self.parameters if col not in df.columns]
            raise ValueError(f"Το CSV αρχείο λείπει από απαραίτητες παραμέτρους για το μοντέλο: {missing_cols}")
        
        # Χρησιμοποιούμε το df απευθείας, αλλά το βάζουμε στη σωστή σειρά
        df_ordered = df[self.parameters]
        
        # 4. Εφαρμογή του ΙΔΙΟΥ scaler που χρησιμοποιήθηκε στην εκπαίδευση
        scaled_data = self.scaler.transform(df_ordered)
        
        # 5. Πρόβλεψη
        prediction_class = self.model_object.predict(scaled_data)[0]

                ### ΝΕΟΣ ΚΩΔΙΚΑΣ: Λήψη πιθανοτήτων ###
        probabilities = self.model_object.predict_proba(scaled_data)[0]
        
        self.clear_folder(PREDICTION_PLOTS_DIR)

        ### ΝΕΟΣ ΚΩΔΙΚΑΣ: Κλήση των συναρτήσεων για δημιουργία plots ###
        print("\nΔημιουργία οπτικοποιήσεων πρόβλεψης...")
        self.plot_prediction_probabilities(probabilities, self.model_name)
        # plot_patient_comparison(patient_data_dict, self.class_averages, prediction_class)
        ### ΤΕΛΟΣ ΝΕΟΥ ΚΩΔΙΚΑ ###
        
        health_status_map = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}
        prediction_string = health_status_map.get(prediction_class, "Unknown")
        
        result = Results(pName, prediction_string, self.parameters, idP, self.id, self.DB)
        result.storeResult()

        return prediction_string


        



class User:
    def __init__(self, fullName, userName, password, role, telephone, email, address, description,DB=None):
        self.id = -1
        self.fullName = fullName
        self.userName = userName
        self.password = password
        self.role = role
        self.telephone = telephone
        self.email = email
        self.address = address
        self.description = description
        self.DB = DB

    def storeUser(self):
        idP = -1
        print(self.DB)
        while True:
            self.id, idP, self.password = self.DB.createUser(self.fullName, self.userName, self.password, self.role, self.telephone, self.email, self.address, self.description)
            if self.id != -1 and idP != -1:
                break
            return idP
        return idP
    
    def print(self):
        print(f"idU -> {self.id} , fN -> {self.fullName} , uN -> {self.userName} , role -> {self.role}\n")

    




class Admin(User):
    def __init__(self, fullName, userName, password, role, telephone, email, address, description, idP, id, DB=None):
        super().__init__(fullName, userName, password, role, telephone, email, address, description, DB)
        self.idP = idP
        self.id = id
        if self.idP == -1 or self.id == -1:
            self.idP = super().storeUser()


    
    # def __del__(self):
    #     deleteUser(self.id,self.idP,self.role)
    #     print("User deleted successfully\n")

    def delete(self,idU):
        self.DB.deleteUser(idU)
        print("User deleted successfully\n")


    def getUsers(self, fullName, id):
        rows = self.DB.getUsers(fullName, id)

        conn = self.DB.connect()
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
                description=result[0],
                DB = self.DB
            )
            user.id = row[0] 
            users.append(user)

        cursor.close()
        conn.close()
        return users
    
    def updateUser(self, userObj, updates: dict):
        self.DB.updateUserInfo(userObj, updates)


    
    def printy(self):
        print(f"idP -> {self.idP}")
        super().print( )





class Medical(User):
    def __init__(self, fullName, userName, password, role, telephone, email, address, description, idP, id, DB=None):
        super().__init__(fullName, userName, password, role, telephone, email, address, description, DB)
        self.idP = idP
        self.id = id
        if self.idP == -1 or self.id == -1:
            self.idP = super().storeUser()


    # def delete(self):
    #     deleteUser(self.id,self.idP,self.role)
    #     print("User deleted successfully\n")


    def getResults(self, pName):
        raw_results = self.DB.getResults(pName, self.idP)

        if raw_results == -1:
            return []

        results_list = []
        for row in raw_results:
            result = Results(
                patientName=row[1],
                fetalHealth=row[2],
                parameters=row[4],
                idMedical=row[3],
                idMo=row[6],
                DB = self.DB
            )
            result.id = row[0]
            results_list.append(result)

        return results_list
    


    def getModels(self,modelID,name):
        raw_results = self.DB.get_models_from_db(modelID,name, self.idP)

        if raw_results == -1:
            return []

        conn = self.DB.connect()
        cursor = conn.cursor()

        results_list = []
        for row in raw_results:
            model = FetalHealthModel(
                id = row[0],
                name = row[1],
                parameters=row[2],
                idM=row[3],
                model_data=row[4],
                DB = self.DB
            )

            if model.idM is not None:
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
    def __init__(self,patientName,fetalHealth,parameters,idMedical,idMo,DB):
        self.id = -1
        self.patientName = patientName
        self.fetalHealth = fetalHealth
        self.parameters = parameters
        self.idMedical = idMedical
        self.idMo = idMo
        self.DB = DB


    def storeResult(self):
        while True:
            self.id = self.DB.postResults(self.idMedical,self.patientName,self.fetalHealth,self.parameters,self.idMo)
            if self.id != -1:
                break

    def __str__(self):
        return f"Result(id={self.id}, patientName='{self.patientName}', fetalHealth={self.fetalHealth}, parameters='{self.parameters}', idMedical={self.idMedical})"

    def __repr__(self):
        return self.__str__()



# -----------------------------------------------------------
# Καθολικές Συναρτήσεις Βάσης Δεδομένων
# -----------------------------------------------------------

class Database:
    def __init__(self,host,user,password,database):
        self.id = 1
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect(self):
        try:
            conn = sql.connect(
                host = self.host,
                user = self.user,
                password = self.password,
                database = self.database
            )
            return conn

        except Error as e:
            print("Connection failed:", e)
            return None


    def login(self,username,password):
            
        conn = self.connect()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM users WHERE username = %s'
        cursor.execute(query, (username,))
        result = cursor.fetchone()

        if result is None:
            print(result)
            cursor.close()
            conn.close()
            return 'Wrong'
        
        idU = result[0]
        fN = result[1]
        passwordDB = result[3].encode('utf-8')
        role = result[4]
        telephone = result[5]
        email = result[6]
        adress = result[7]

        
        if bcrypt.checkpw(password.encode('utf-8'), passwordDB):
            print(result)
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
                user = Admin(fN,username,password,role,telephone,email,adress,description,idP,idU,None)
            else:
                user = Medical(fN,username,password,role,telephone,email,adress,description,idP,idU,None)

            return user
        else:
            cursor.close()
            conn.close()
            return 'Wrong'


    def getResults(self,pName,idM):
            
        conn = self.connect()
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


    def postResults(self,idP,pName,fH,parameters,idMo): # Αφαίρεση image1, image2 από εδώ, καθώς δεν χρησιμοποιούνται στην INSERT
            
        conn = self.connect()
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


    def createUser(self,fullName, username, password, role, telephone, email, adress, description):
        
        conn = self.connect()
        cursor = conn.cursor()

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        query = """
            INSERT INTO users (fullName, username, password, role, telephone, email, address) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        data = (fullName, username, hashed_password.decode('utf-8'), role, telephone, email, adress)
        try:
            cursor.execute(query, data)
            conn.commit()

        except sql.errors.IntegrityError as err :
            # Check if the error message specifically mentions a duplicate entry for the 'username' key
            if "Duplicate entry" in str(err) and "for key 'username'" in str(err):
                print(f"Error: Username '{username}' already exists. Please choose a different username.")
                # Optionally, display this message in your Tkinter app using messagebox.showerror
                # messagebox.showerror("Error", f"Username '{username}' already exists. Please choose a different username.")

                query = "ALTER TABLE users AUTO_INCREMENT = %s"
                cursor.execute(query,(1,))
                query = "ALTER TABLE administrators AUTO_INCREMENT = %s"
                cursor.execute(query,(1,))
                query = "ALTER TABLE medical_personnel AUTO_INCREMENT = %s"
                cursor.execute(query,(1,))
                cursor.close()
                conn.close()
                return -1,-1,-1 # Indicate failure due to duplicate username



        query = 'SELECT id FROM users WHERE username = %s'
        cursor.execute(query, (username,))
        result = cursor.fetchone()

        if result is None:
            cursor.close()
            conn.close()
            return -1,-1,-1
        
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
            cursor.close()
            conn.close()
            return -1,-1,-1

        idP = result[0]

        cursor.close()
        conn.close()

        print("User inserted successfully.")
        return idU,idP,hashed_password


    def getUsers(self,fullName,id):
            
        conn = self.connect()
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


    def is_bcrypt_hash(self,s):
        """
        Heuristically checks if a string appears to be a bcrypt hash.
        This is NOT a foolproof method and should be used with caution.
        """
        if not isinstance(s, str):
            return False
        
        # Bcrypt hashes are usually 60 characters long (for a cost of 12)
        # and start with $2a$, $2b$, or $2y$ followed by the cost factor.
        # A common length for a bcrypt hash is 60 characters.
        if len(s) == 60 and s.startswith(("$2a$", "$2b$", "$2y$")):
            parts = s.split('$')
            if len(parts) == 4 and parts[0] == '' and parts[1] in ('2a', '2b', '2y'):
                try:
                    # Check if the cost factor is a valid integer
                    cost = int(parts[2])
                    if 4 <= cost <= 31: # Bcrypt cost factors typically range from 4 to 31
                        return True
                except ValueError:
                    pass
        return False


    def updateUserInfo(self,userObj, updates: dict):

        print(updates)

        for field, value in updates.items():
            if field == "password":
                if not self.is_bcrypt_hash(value): # Only hash if it's not already hashed
                    hashed_password = bcrypt.hashpw(value.encode('utf-8'), bcrypt.gensalt())
                    updates[field] = hashed_password.decode('utf-8')
                else:
                    # If it's already a bcrypt hash, assume it's valid and use it as is.
                    # In a real-world scenario, you might want to log this or have a policy
                    # that updates always provide plaintext for hashing.
                    print("HASHED")
                    pass # The value is already hashed, no need to re-hash.

            if field != "description":
                if hasattr(userObj, field):
                    setattr(userObj, field, value)

        if "description" in updates:
            userObj.description = updates["description"]
            del updates["description"]


        conn = self.connect()
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

        
    def deleteUser(self,idU):
        conn = self.connect()
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

        conn.commit()

        print("User deleted successfully\n")


    def create_model_in_db(self,name, parameters_json_string, maker_id, model_data_blob):
        """
        Δημιουργεί μια νέα εγγραφή μοντέλου στη βάση δεδομένων με τα δυαδικά δεδομένα του μοντέλου.
        """
        conn = self.connect()
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


    def get_models_from_db(self,model_id=None, name=None, maker_id=None):
        """
        Ανακτά μοντέλα από τη βάση δεδομένων. Μπορεί να φιλτράρει με ID, όνομα ή ID κατασκευαστή.
        Επιστρέφει μια λίστα από tuples, όπου κάθε tuple περιέχει (id, name, parameters, maker, model_data).
        """
        conn = self.connect()
        cursor = conn.cursor()


        query = "SELECT id, name, parameters, maker, model_data FROM model WHERE 1=1"
        params = []

        if model_id is not None and model_id != -1:
            query += " AND id = %s " 
            params.append(model_id)
        if name is not None and name.strip():
            query += " AND name LIKE %s"
            params.append(f"%{name}%")
        if maker_id is not None and maker_id != -1:
            query += " AND maker = %s"
            params.append(maker_id)
        if model_id is None or model_id == -1:
            query += " OR id = 1"

        cursor.execute(query, tuple(params))
        result = cursor.fetchall()
        

        cursor.close()
        conn.close()


        if not result:
            return []

        return result

    def delete_model_from_db(self,model_id):
        """
        Διαγράφει μια εγγραφή μοντέλου από τη βάση δεδομένων με βάση το ID.
        """
        conn = self.connect()
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



#ALTER TABLE users AUTO_INCREMENT = 1;
