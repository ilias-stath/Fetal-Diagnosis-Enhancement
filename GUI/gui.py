import tkinter as tk
from tkinter import messagebox, filedialog, ttk

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Login Page")
        self.root.attributes('-fullscreen', True)

        self.bg_color = "#1e3f66"
        self.root.configure(bg=self.bg_color)
        self.current_page = "login"
        self.csv_path = None

        self.ADMIN_CREDENTIALS = ("admin", "admin")
        self.USER_CREDENTIALS = ("user", "user")

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

        tk.Label(center_frame, text="FEDE", font=("Arial", 48, "bold"),
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

        if (user, pw) == self.ADMIN_CREDENTIALS:
            self.show_admin_page()
        elif (user, pw) == self.USER_CREDENTIALS:
            self.show_user_page()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    def show_admin_page(self):
        self.clear_frame()
        self.current_page = "admin"

        tk.Label(self.main_frame, text="Welcome, Admin!", font=("Arial", 36, "bold"),
                bg="white", fg=self.bg_color, padx=30, pady=20, bd=2, relief="groove").pack(pady=(40, 30))

        ttk.Button(self.main_frame, text="Look up user data", width=30, command=self.lookup_user_data).pack(pady=10)
        ttk.Button(self.main_frame, text="Exit", style="Exit.TButton", command=self.confirm_exit).pack(pady=30)

    def show_user_page(self):
        self.clear_frame()
        self.current_page = "user"

        tk.Label(self.main_frame, text="Welcome, User!", font=("Arial", 36, "bold"),
                bg="white", fg=self.bg_color, padx=30, pady=20, bd=2, relief="groove").pack(pady=(40, 30))

        ttk.Button(self.main_frame, text="Insert values for estimation", width=40, command=self.insert_values).pack(pady=10)
        ttk.Button(self.main_frame, text="Search previous results", width=40, command=self.search_results).pack(pady=10)
        ttk.Button(self.main_frame, text="Train model with new parameters", width=40, command=self.train_model_screen).pack(pady=10)
        ttk.Button(self.main_frame, text="Exit", style="Exit.TButton", command=self.confirm_exit).pack(pady=30)

    def insert_values(self):
        self.show_csv_screen(
            title="Insert CSV for Estimation",
            run_command=self.run_model,
            back_command=self.show_user_page
        )

    def train_model_screen(self):
        self.show_csv_screen(
            title="Train Model with CSV",
            run_command=self.train_model,
            back_command=self.show_user_page
        )

    def show_csv_screen(self, title, run_command, back_command):
        self.clear_frame()
        self.csv_path = None

        tk.Label(self.main_frame, text=title, font=("Arial", 30, "bold"),
                 bg="white", fg=self.bg_color, padx=30, pady=20, bd=2, relief="groove").pack(pady=(30, 20))

        ttk.Button(self.main_frame, text="Browse CSV File", width=30, command=self.browse_csv).pack(pady=10)

        self.csv_label = tk.Label(self.main_frame, text="", bg=self.bg_color, fg="white",
                                  font=("Segoe UI", 12), wraplength=800)
        self.csv_label.pack(pady=(20, 10))

        self.run_model_button = ttk.Button(self.main_frame, text="Run", width=30, command=run_command)
        self.run_model_button_is_visible = False

        self.back_button = ttk.Button(self.main_frame, text="Back", style="Exit.TButton", command=back_command)
        self.back_button.pack(pady=10)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_path = file_path
            self.csv_label.config(text=f"Selected File: {file_path}")
            if not self.run_model_button_is_visible:
                self.run_model_button.pack(before=self.back_button, pady=10)
                self.run_model_button_is_visible = True

    def confirm_exit(self):
        answer = messagebox.askyesno("Exit Confirmation", "Do you want to log out?")
        if answer:
            self.create_widgets()

    def search_results(self):
        print("Search previous results - placeholder")

    def train_model(self):
        if self.csv_path:
            messagebox.showinfo("Training", f"Training model with:\n{self.csv_path}")
        else:
            messagebox.showwarning("No File", "Please select a CSV file first.")

    def run_model(self):
        if self.csv_path:
            messagebox.showinfo("Estimation", f"Running model with:\n{self.csv_path}")
        else:
            messagebox.showwarning("No File", "Please select a CSV file first.")

    def lookup_user_data(self):
        print("Look up user data - placeholder")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
