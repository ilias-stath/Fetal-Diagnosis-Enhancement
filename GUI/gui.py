import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import Database as DB
from PIL import Image, ImageTk
import os
import glob

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Login Page")
        self.root.attributes('-fullscreen', True)

        self.bg_color = "#1e3f66"
        self.root.configure(bg=self.bg_color)
        self.current_page = "login"
        self.csv_path = None


        self.setup_style()

        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(expand=True, fill='both')

        self.create_widgets()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 12), padding=10)
        style.configure("Exit.TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Title.TLabel", font=("Arial", 48, "bold"), background="white", foreground=self.bg_color)
        style.configure("Header.TLabel", font=("Arial", 36), background="#4a4a4a", foreground="white")
        style.configure("UserHeader.TLabel", font=("Arial", 36), background="#2d6a4f", foreground="white")

    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.main_frame.configure(bg=self.bg_color)

    def create_widgets(self):
        self.clear_frame()
        self.current_page = "login"

        center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(center_frame, text="FEDET", font=("Arial", 48, "bold"),
                bg="white", fg=self.bg_color, padx=20, pady=10).pack(pady=(0, 30))

        tk.Label(center_frame, text="Username", bg=self.bg_color, fg="white").pack()
        self.username_entry = tk.Entry(center_frame, width=30)
        self.username_entry.pack(pady=(0, 10))

        tk.Label(center_frame, text="Password", bg=self.bg_color, fg="white").pack()
        self.password_entry = tk.Entry(center_frame, show="*", width=30)
        self.password_entry.pack(pady=(0, 10))

        tk.Button(center_frame, text="Login", command=self.login, width=20, height=1).pack(pady=10)
        tk.Button(center_frame, text="Exit", command=self.root.quit, width=20, height=1).pack(pady=(5, 0))

        self.root.bind("<Escape>", lambda event: self.root.attributes('-fullscreen', False))

    def login(self):
        user = self.username_entry.get()
        pw = self.password_entry.get()

        if user == "1":
            user = "ilias_stath"
            pw = "123456789"
        elif user == "2":
            user = "george_ktist"
            pw = "123456789"

        self.User = DB.login(user, pw)
        if type(self.User) != str:
            if self.User.role == "admin":
                self.show_admin_page()
            elif self.User.role == "medical":
                self.show_user_page()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    def show_admin_page(self):
        self.clear_frame()
        center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(center_frame, text="Welcome, "+self.User.fullName, font=("Arial", 36, "bold"),
                 bg="white", fg=self.bg_color, padx=30, pady=20, bd=2, relief="groove").pack(pady=(0, 30))
        

        ttk.Button(center_frame, text="Look up user data", width=30, command=self.lookup_user_data).pack(pady=10)
        ttk.Button(center_frame, text="Create New User",width=30, command=self.open_create_user_form).pack(pady=10)
        ttk.Button(center_frame, text="Exit", style="Exit.TButton", command=self.confirm_exit).pack(pady=30)

    def show_user_page(self):
        self.clear_frame()
        center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(center_frame, text="Welcome, "+self.User.fullName, font=("Arial", 36, "bold"),
                 bg="white", fg=self.bg_color, padx=30, pady=20, bd=2, relief="groove").pack(pady=(0, 30))

        ttk.Button(center_frame, text="Insert values for estimation", width=40, command=self.insert_values).pack(pady=10)
        ttk.Button(center_frame, text="Search previous results", width=40, command=self.search_results).pack(pady=10)
        ttk.Button(center_frame, text="Train model with new parameters", width=40, command=self.train_model_screen).pack(pady=10)
        ttk.Button(center_frame, text="Exit", style="Exit.TButton", command=self.confirm_exit).pack(pady=30)

    def insert_values(self):
        self.show_csv_screen(
            title="Insert CSV for Estimation",
            run_command=self.run_model,
            back_command=self.show_user_page,
            training=False
        )

    def train_model_screen(self):
        self.show_csv_screen(
            title="Train Model with CSV",
            run_command=self.train_model,
            back_command=self.show_user_page
        )

    def show_csv_screen(self, title, run_command, back_command,training=True):
        self.clear_frame()
        self.csv_path = None

        center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(center_frame, text=title, font=("Arial", 30, "bold"),
                 bg="white", fg=self.bg_color, padx=30, pady=20, bd=2, relief="groove").pack(pady=(0, 20))

        ttk.Button(center_frame, text="Browse CSV File", width=30, command=lambda:self.browse_csv(training)).pack(pady=10)

        self.csv_label = tk.Label(center_frame, text="", bg=self.bg_color, fg="white",
                                  font=("Segoe UI", 12), wraplength=800)
        self.csv_label.pack(pady=(20, 10))

        if training:
            self.run_model_button = ttk.Button(center_frame, text="Run", width=30, command=run_command)
            self.run_model_button_is_visible = False

        self.back_button = ttk.Button(center_frame, text="Back", style="Exit.TButton", command=back_command)
        self.back_button.pack(pady=10)

        self.center_csv_frame = center_frame  # store reference

    def browse_csv(self,training):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_path = file_path
            self.csv_label.config(text=f"Selected File: {file_path}")
            if training:
                if not self.run_model_button_is_visible:
                    self.run_model_button.pack(pady=10)
                    self.run_model_button_is_visible = True
            else:
                self.display_model_table()  # Show models when CSV is selected

    def display_model_table(self):
        # Destroy previous model table if it exists
        if hasattr(self, 'model_table_container') and self.model_table_container.winfo_exists():
            self.model_table_container.destroy()

        # Outer container
        self.model_table_container = tk.Frame(self.center_csv_frame, bg="white")
        self.model_table_container.pack(padx=20, pady=10, fill="both", expand=True)

        # Canvas with vertical scrollbar
        canvas = tk.Canvas(self.model_table_container, width=750, height=200, bg="white")
        scrollbar = ttk.Scrollbar(self.model_table_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Table headers
        headers = ["ID", "Model Name", "Author ID", "Actions"]
        for col, header in enumerate(headers):
            tk.Label(scrollable_frame, text=header, font=("Segoe UI", 10, "bold"),
                    bg="#dbeafe", fg="black", borderwidth=1, relief="solid", padx=5, pady=5).grid(row=0, column=col, sticky="nsew")

        # Table rows
        model_list = self.User.getModels(-1, "")

        for row_idx, model in enumerate(model_list, start=1):
            fields = [model.id, model.model_name, model.idM]
            for col_idx, field in enumerate(fields):
                tk.Label(scrollable_frame, text=str(field), bg="white", fg="black",
                        borderwidth=1, relief="solid", padx=4, pady=4, anchor="w", justify="left").grid(row=row_idx, column=col_idx, sticky="nsew")

            action_frame = tk.Frame(scrollable_frame, bg="white")
            action_frame.grid(row=row_idx, column=len(fields), padx=4, pady=4)

            tk.Button(action_frame, text="Delete", width=6,
                    command=lambda mid=model: self.delete_model(mid)).pack(side="left", padx=2)

            tk.Button(action_frame, text="Run", width=6,
                    command=lambda mid=model: self.run_model_with_csv(mid)).pack(side="left", padx=2)
            
    def delete_model(self, model):
        if model.idM is not None:
            confirm = messagebox.askyesno("Delete Model", f"Are you sure you want to delete model ID {model.id}?")
            if confirm:
                DB.delete_model_from_db(model.id)
                print(f"Deleted model with ID {model.id}.")
                messagebox.showinfo("Model Deleted", f"Model ID {model.id} was successfully deleted.")
                self.display_model_table()  # Refresh the model table
        else:
            messagebox.showinfo("Error!","Original model cannot be deleted!!")

    def run_model_with_csv(self, model):
        try:
            res = model.predict_health_status(self.csv_path, self.User.idP)
            print(f"Run model ID {model.id} on selected CSV {self.csv_path} result -> {res}.")

            # Clear the screen
            self.clear_frame()

            # Center frame like in show_admin_page
            center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
            center_frame.place(relx=0.5, rely=0.5, anchor='center')

            # Result label styled similarly
            tk.Label(
                center_frame,
                text=f"Model prediction result:\n{res}",
                font=("Arial", 20, "bold"),
                bg="white",
                fg=self.bg_color,
                padx=30,
                pady=20,
                bd=2,
                relief="groove"
            ).pack(pady=(0, 30))

            # Image frame
            img_frame = tk.Frame(center_frame, bg="white")
            img_frame.pack()

            # Load and display two images side by side
            image_dir = "prediction_plots"
            png_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

            if png_files:
                img = Image.open(png_files[0]).resize((600, 600))

            tk_img = ImageTk.PhotoImage(img)

            label_img = tk.Label(img_frame, image=tk_img, bg="white", bd=2, relief="solid")
            label_img.image = tk_img
            label_img.pack(side="left", padx=10)

            # Home button below images
            def go_home():
                self.clear_frame()
                self.show_user_page()

            ttk.Button(center_frame, text="Home", style="Exit.TButton", command=go_home).pack(pady=30)

        except Exception as e:
            self.clear_frame()

            error_frame = tk.Frame(self.main_frame, bg=self.bg_color)
            error_frame.place(relx=0.5, rely=0.5, anchor='center')

            tk.Label(error_frame, text=f"An error occurred:\n{e}",
                    font=("Arial", 16), fg="red", bg="white",
                    padx=20, pady=20, bd=2, relief="groove").pack(pady=10)

            ttk.Button(error_frame, text="Back", style="Exit.TButton", command=self.show_user_page).pack(pady=20)


    def confirm_exit(self):
        answer = messagebox.askyesno("Exit Confirmation", "Do you want to log out?")
        if answer:
            self.create_widgets()

    def search_results(self):
        print("Search previous results - placeholder")
        users_list = self.User.getResults("")
        for user in users_list:
            # Update user info using admin
            fields = [user.id, user.patientName, user.fetalHealth, user.parameters, user.idMo]
        

    def train_model(self):
        if self.csv_path:
            model = DB.FetalHealthModel(-1, "", "", self.User.id, "")
            model.train_new_model(self.csv_path)

            # Clear screen before showing UI
            self.clear_frame()

            # Centered content frame
            center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
            center_frame.place(relx=0.5, rely=0.5, anchor='center')

            # Success label
            tk.Label(
                center_frame,
                text="Model trained successfully!",
                font=("Arial", 20, "bold"),
                bg="white",
                fg=self.bg_color,
                padx=30,
                pady=20,
                bd=2,
                relief="groove"
            ).pack(pady=(0, 30))

            # Image frame
            img_frame = tk.Frame(center_frame, bg="white")
            img_frame.pack()

            # Load and resize images
            # Get list of PNG files in the directory (sorted alphabetically)
            png_files = sorted(glob.glob(os.path.join("model_plots", "*.png")))

            # Make sure at least two images exist
            if len(png_files) >= 2:
                img1 = Image.open(png_files[0])
                img2 = Image.open(png_files[1])
            img1 = img1.resize((600, 600))
            img2 = img2.resize((600, 600))

            tk_img1 = ImageTk.PhotoImage(img1)
            tk_img2 = ImageTk.PhotoImage(img2)

            label_img1 = tk.Label(img_frame, image=tk_img1, bg="white", bd=2, relief="solid")
            label_img1.image = tk_img1
            label_img1.pack(side="left", padx=10)

            label_img2 = tk.Label(img_frame, image=tk_img2, bg="white", bd=2, relief="solid")
            label_img2.image = tk_img2
            label_img2.pack(side="left", padx=10)

            # Home button
            def go_home():
                self.clear_frame()
                self.show_user_page()

            ttk.Button(center_frame, text="Home", style="Exit.TButton", command=go_home).pack(pady=30)


    def run_model(self):
        if self.csv_path:
            messagebox.showinfo("Estimation", f"Running model with:\n{self.csv_path}")
        else:
            messagebox.showwarning("No File", "Please select a CSV file first.")

    def open_create_user_form(self):
        self.clear_frame()
        self.current_page = "create_user"

        center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(center_frame, text="Create New User", font=("Arial", 28, "bold"),
                bg="white", fg=self.bg_color, padx=30, pady=10, bd=2, relief="groove").pack(pady=(0, 20))

        fields = ["Full Name", "Username", "Password", "Telephone", "Email", "Address", "Description"]
        self.create_user_entries = {}

        for field in fields:
            tk.Label(center_frame, text=field + ":", bg=self.bg_color, fg="white", font=("Segoe UI", 12)).pack()
            entry = tk.Entry(center_frame, width=40, show="*" if field == "Password" else "")
            entry.pack(pady=(0, 10))
            self.create_user_entries[field] = entry

        # Role dropdown
        tk.Label(center_frame, text="Role:", bg=self.bg_color, fg="white", font=("Segoe UI", 12)).pack()
        self.role_combobox = ttk.Combobox(center_frame, values=["admin", "medical"], state="readonly", width=37)
        self.role_combobox.set("medical")
        self.role_combobox.pack(pady=(0, 20))

        ttk.Button(center_frame, text="Create User", command=self.submit_create_user).pack(pady=(10, 10))
        ttk.Button(center_frame, text="Back", style="Exit.TButton", command=self.show_admin_page).pack(pady=(0, 10))


    def submit_create_user(self):
        values = {field: self.create_user_entries[field].get().strip() for field in self.create_user_entries}
        values["Role"] = self.role_combobox.get()

        # Basic validation
        if not all(values.values()):
            messagebox.showerror("Input Error", "All fields are required.")
            return
        if "@" not in values["Email"]:
            messagebox.showerror("Input Error", "Invalid email format.")
            return
        if len(values["Password"]) < 6:
            messagebox.showerror("Input Error", "Password must be at least 6 characters.")
            return

        # Create the appropriate user object
        if values["Role"] == 'admin':
            x = DB.Admin(values["Full Name"], values["Username"], values["Password"],
                        values["Role"], values["Telephone"], values["Email"],
                        values["Address"], values["Description"], -1, -1)
        else:
            x = DB.Medical(values["Full Name"], values["Username"], values["Password"],
                            values["Role"], values["Telephone"], values["Email"],
                            values["Address"], values["Description"], -1, -1)

        # Optional: Save or process the user object
        messagebox.showinfo("Success", f"New {values['Role']} user created!")
        self.show_admin_page()

    def search_results(self):
        self.clear_frame()

        center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')
        self.center_frame = center_frame  # store it for reuse

        tk.Label(center_frame, text="Search Results", font=("Arial", 28, "bold"),
                bg="white", fg=self.bg_color, padx=30, pady=10, bd=2, relief="groove").pack(pady=(0, 20))

        search_frame = tk.Frame(center_frame, bg=self.bg_color)
        search_frame.pack(pady=(0, 10))

        tk.Label(search_frame, text="Search by Patient Name:", bg=self.bg_color,
                fg="white", font=("Segoe UI", 12)).pack(side="left", padx=(0, 10))

        self.results_search_entry = tk.Entry(search_frame, width=30)
        self.results_search_entry.pack(side="left", padx=(0, 10))

        ttk.Button(search_frame, text="Search", command=lambda: self.display_results_table(self.results_search_entry.get())).pack(side="left")

        if not hasattr(self, 'results_back_button'):
            self.results_back_button = ttk.Button(center_frame, text="Back", style="Exit.TButton", command=self.show_admin_page)
            self.results_back_button.pack(pady=20)

        self.display_results_table("")  # Show all results initially

    def display_results_table(self, filter_text):
        # Destroy previous table frame if exists
        if hasattr(self, 'results_table_container') and self.results_table_container.winfo_exists():
            self.results_table_container.destroy()

        # Create a container with canvas + scrollbar
        self.results_table_container = tk.Frame(self.center_frame, bg="white")
        self.results_table_container.pack(padx=20, pady=10, fill="both", expand=True)

        canvas = tk.Canvas(self.results_table_container, width=545, height=400, bg="white")
        scrollbar = tk.Scrollbar(self.results_table_container, orient="vertical", command=canvas.yview)
        self.results_table_frame = tk.Frame(canvas, bg="white")

        self.results_table_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.results_table_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Table headers
        headers = ["ID", "Patient Name", "Fetal Health", "Parameters", "ID Mo"]
        for col, header in enumerate(headers):
            tk.Label(self.results_table_frame, text=header, font=("Segoe UI", 10, "bold"),
                    bg="#dbeafe", fg="black", borderwidth=1, relief="solid", padx=5, pady=5).grid(row=0, column=col, sticky="nsew")

        results_list = self.User.getResults("") if not filter_text else self.User.getResults(filter_text)

        for row_idx, result in enumerate(results_list, start=1):
            fields = [result.id, result.patientName, result.fetalHealth, result.parameters, result.idMo]
            for col_idx, field in enumerate(fields):
                if headers[col_idx] == "Parameters":
                    text = str(field)
                    tk.Label(self.results_table_frame, text=text, bg="white", fg="black", borderwidth=1,
                            relief="solid", padx=4, pady=4, anchor="w", justify="left",
                            wraplength=300, width=40).grid(row=row_idx, column=col_idx, sticky="nsew")
                else:
                    tk.Label(self.results_table_frame, text=str(field), bg="white", fg="black", borderwidth=1,
                            relief="solid", padx=4, pady=4, anchor="w", justify="left").grid(row=row_idx, column=col_idx, sticky="nsew")

            action_frame = tk.Frame(self.results_table_frame, bg="white")
            action_frame.grid(row=row_idx, column=len(fields), padx=4, pady=4)


        # Destroy previous back button if exists
        if hasattr(self, 'results_back_button') and self.results_back_button.winfo_exists():
            self.results_back_button.destroy()

        # Recreate Back button
        self.results_back_button = ttk.Button(self.center_frame, text="Back", style="Exit.TButton", command=self.show_user_page)
        self.results_back_button.pack(pady=20)
     


    def lookup_user_data(self):
        self.clear_frame()

        center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')
        self.center_frame = center_frame  # store it for reuse

        tk.Label(center_frame, text="User Database", font=("Arial", 28, "bold"),
                bg="white", fg=self.bg_color, padx=30, pady=10, bd=2, relief="groove").pack(pady=(0, 20))

        search_frame = tk.Frame(center_frame, bg=self.bg_color)
        search_frame.pack(pady=(0, 10))

        tk.Label(search_frame, text="Search by Name:", bg=self.bg_color, fg="white", font=("Segoe UI", 12)).pack(side="left", padx=(0, 10))

        self.search_entry = tk.Entry(search_frame, width=30)
        self.search_entry.pack(side="left", padx=(0, 10))

        ttk.Button(search_frame, text="Search", command=lambda: self.display_user_table(self.search_entry.get())).pack(side="left")

        # Create Back button once
        if not hasattr(self, 'lookup_back_button'):
            self.lookup_back_button = ttk.Button(center_frame, text="Back", style="Exit.TButton", command=self.show_admin_page)
            self.lookup_back_button.pack(pady=20)

        self.display_user_table("")  # show all users initially

    def display_user_table(self, filter_text):
        # Destroy previous table frame if exists
        if hasattr(self, 'table_container') and self.table_container.winfo_exists():
            self.table_container.destroy()

        # Outer container
        self.table_container = tk.Frame(self.center_frame, bg="white")
        self.table_container.pack(padx=20, pady=10, fill="both", expand=True)

        # Canvas with scrollbar
        canvas = tk.Canvas(self.table_container, width=875, height=400, bg="white")
        scrollbar = ttk.Scrollbar(self.table_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Table headers
        headers = ["ID", "Name", "Phone", "Username", "Role", "Address", "Email", "Description", "Actions"]
        for col, header in enumerate(headers):
            tk.Label(scrollable_frame, text=header, font=("Segoe UI", 10, "bold"),
                    bg="#dbeafe", fg="black", borderwidth=1, relief="solid", padx=5, pady=5).grid(row=0, column=col, sticky="nsew")

        # User rows
        users_list = self.User.getUsers("", -1) if not filter_text else self.User.getUsers(filter_text, -1)

        for row_idx, user in enumerate(users_list, start=1):
            fields = [user.id, user.fullName, user.telephone, user.userName, user.role,
                    user.address, user.email, user.description]
            for col_idx, field in enumerate(fields):
                tk.Label(scrollable_frame, text=str(field), bg="white", fg="black", borderwidth=1,
                        relief="solid", padx=4, pady=4, anchor="w", justify="left").grid(row=row_idx, column=col_idx, sticky="nsew")

            action_frame = tk.Frame(scrollable_frame, bg="white")
            action_frame.grid(row=row_idx, column=len(fields), padx=4, pady=4)

            tk.Button(action_frame, text="Edit", width=6,
                    command=lambda userObj=user: self.editUser(userObj)).pack(side="left", padx=2)
            tk.Button(action_frame, text="Delete", width=6,
                    command=lambda uid=user.id: self.deleteUser(uid)).pack(side="left", padx=2)

        # Destroy previous back button if exists
        if hasattr(self, 'lookup_back_button') and self.lookup_back_button.winfo_exists():
            self.lookup_back_button.destroy()

        # Back button
        self.lookup_back_button = ttk.Button(self.center_frame, text="Back", style="Exit.TButton", command=self.show_admin_page)
        self.lookup_back_button.pack(pady=20)



    def editUser(self, userObj):
        self.clear_frame()
        self.current_page = "edit_user"

        user = userObj

        center_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(center_frame, text="Edit User", font=("Arial", 28, "bold"),
                bg="white", fg=self.bg_color, padx=30, pady=10, bd=2, relief="groove").pack(pady=(0, 20))

        fields = ["fullName", "username", "password", "telephone", "email", "address", "description"]
        field_attr_map = {
            "fullName": "fullName",
            "username": "userName",
            "password": "password",
            "telephone": "telephone",
            "email": "email",
            "address": "address",
            "description": "description"
        }

        self.edit_user_entries = {}

        for field in fields:
            tk.Label(center_frame, text=field + ":", bg=self.bg_color, fg="white", font=("Segoe UI", 12)).pack()
            entry = tk.Entry(center_frame, width=40, show="*" if field == "password" else "")
            entry.pack(pady=(0, 10))

            value = getattr(user, field_attr_map[field], "")
            entry.insert(0, value)
            self.edit_user_entries[field] = entry

        # Role dropdown
        tk.Label(center_frame, text="role:", bg=self.bg_color, fg="white", font=("Segoe UI", 12)).pack()
        self.edit_role_combobox = ttk.Combobox(center_frame, values=["admin", "medical"], state="readonly", width=37)
        self.edit_role_combobox.set(user.role)
        self.edit_role_combobox.pack(pady=(0, 20))

        ttk.Button(center_frame, text="Save Changes", command=lambda: self.update_user(user)).pack(pady=(10, 10))
        ttk.Button(center_frame, text="Back", style="Exit.TButton", command=self.lookup_user_data).pack(pady=(0, 10))
    
    def update_user(self, user):
        values = {field: self.edit_user_entries[field].get().strip() for field in self.edit_user_entries}
        values["role"] = self.edit_role_combobox.get()

        # Basic validation
        if not all(values.values()):
            messagebox.showerror("Input Error", "All fields are required.")
            return
        if "@" not in values["email"]:
            messagebox.showerror("Input Error", "Invalid email format.")
            return
        if len(values["password"]) < 6:
            messagebox.showerror("Input Error", "Password must be at least 6 characters.")
            return

        # Update the user
        self.User.updateUser(user,values)

        # Example saving logic (uncomment if applicable)
        # x.update()

        messagebox.showinfo("Success", f"User ID {user.id} updated successfully!")
        self.lookup_user_data()


    def deleteUser(self, user_id):
        print(f"Delete user {user_id}")
        confirm = messagebox.askyesno("Delete User", f"Are you sure you want to delete user ID {user_id}?")
        if confirm:
            self.User.delete(user_id)
            messagebox.showinfo("Success", f"User ID {user_id} successfully deleted.")
            self.display_user_table("")  # Replace this with your actual function that shows the users page

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
