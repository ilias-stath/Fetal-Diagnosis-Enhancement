import mysql.connector as sql
from mysql.connector import Error
import bcrypt
import json
# import classes


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

    def delete(self):
        deleteUser(self.id,self.idP,self.role)
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


    def delete(self):
        deleteUser(self.id,self.idP,self.role)
        print("User deleted successfully\n")


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
                idMedical=row[3]
            )
            result.id = row[0]
            results_list.append(result)

        return results_list


    def printy(self):
        print(f"idP -> {self.idP}")
        super().print( )






class Results:
    def __init__(self,patientName,fetalHealth,parameters,idMedical):
        self.id = -1
        self.patientName = patientName
        self.fetalHealth = fetalHealth
        self.parameters = parameters
        self.idMedical = idMedical


    def storeResult(self):
        while True:
            self.id = postResults(self.idMedical,self.patientName,self.fetalHealth,self.parameters)
            if self.id != -1:
                break

    def __str__(self):
        return f"Result(id={self.id}, patientName='{self.patientName}', fetalHealth={self.fetalHealth}, parameters='{self.parameters}', idMedical={self.idMedical})"

    def __repr__(self):
        return self.__str__()




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

    # if not conn :
    #     return 'NO_CONN'    #Cannot establise connection to DB
    
    cursor = conn.cursor()
    
    query = 'SELECT * FROM users WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchone()

    
    if result is None:
        print(result)
        return 'Wrong'
    
    idU = result[0]
    fN = result[1]
    #2nd is username, we already have it
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

    # if not conn :
    #     return 'NO_CONN'    #Cannot establise connection to DB
    
    cursor = conn.cursor()

    # query = 'SELECT id FROM medical_personnel WHERE user_id = %s'
    # cursor.execute(query, (idM,))
    # result = cursor.fetchone()

    # if result is None:
    #     return -1

    # idM = result[0]

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


# def postData(mName,pName,fH,parameters): #parameters is json data
def postResults(idM,pName,fH,parameters):
     
    conn = connect()

    # if not conn :
    #     return -1    #Cannot establise connection to DB
    
    cursor = conn.cursor()


    # query = 'SELECT id FROM users WHERE username = %s'
    # cursor.execute(query, (mName,))
    # result = cursor.fetchone()

    # if result is None:
    #     return False

    # idM = result[0]

    # # print("id of user -> ",idM)

    # query = 'SELECT id FROM medical_personnel WHERE user_id = %s'
    # cursor.execute(query, (idM,))
    # result = cursor.fetchone()

    # if result is None:
    #     return -1

    # idM = result[0]


    # query = """
    # SELECT medical_personnel.id
    # FROM users
    # JOIN medical_personnel ON users.id = medical_personnel.user_id
    # WHERE users.username = %s
    # """

    # cursor.execute(query, (mName,))
    # result = cursor.fetchone()

    # if result is None:
    #     return False

    # idM = result[0]

    # print("id of doctor -> ",idM)

    query = """
        INSERT INTO results (Patient_Name, Fetal_Health, medical_supervisor, parameters) 
        VALUES (%s, %s, %s, %s)
    """

    data = (pName,fH,idM,json.dumps(parameters))
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

    #password = "123456789"
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    query = """
        INSERT INTO users (fullName, username, password, role, telephone, email, adress) 
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

    # if not conn :
    #     return 'NO_CONN'    #Cannot establise connection to DB
    
    cursor = conn.cursor()

    # query = 'SELECT id FROM medical_personnel WHERE user_id = %s'
    # cursor.execute(query, (idM,))
    # result = cursor.fetchone()

    # if result is None:
    #     return -1

    # idM = result[0]

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
    # Update user object's attributes
    for field, value in updates.items():
        if field == "password":
            # Hash the password
            hashed_password = bcrypt.hashpw(value.encode('utf-8'), bcrypt.gensalt())
            value = hashed_password.decode('utf-8')
            updates[field] = value  # update the dictionary with the hashed password
            
        if hasattr(userObj, field):
            setattr(userObj, field, value)

    # Prepare the SQL query
    set_clause = ", ".join(f"{field} = %s" for field in updates.keys())
    values = list(updates.values())
    values.append(userObj.id)  # For WHERE clause

    query = f"UPDATE users SET {set_clause} WHERE id = %s"

    conn = connect()
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()

    # Update the description in the corresponding table
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


    
def deleteUser(idU,idP,role):
    conn = connect()
    cursor = conn.cursor()

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

# Admin = login('ilias_stath','123456789')
# users_list = Admin.getUsers("Konstantinos Papathanasiou",-1)
# for user in users_list:
    
#     # Update user info using admin
#     Admin.updateUser(user, {
#         "address": "Kozani",
#     })



#-----Do the hash for every knew user


# conn = connect()
# cursor = conn.cursor()

# plain_password = "123456789"
# hashed_password = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())

# query = """
#     INSERT INTO users (fullName, username, password, role, telephone, email, adress) 
#     VALUES (%s, %s, %s, %s, %s, %s, %s)
# """

# data = ("Ilias Stathakos", "ilias_stath", hashed_password.decode('utf-8'), "admin", "+306999999", "ece002017@uowm.gr", "kozani")
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







#ALTER TABLE users AUTO_INCREMENT = 1;
