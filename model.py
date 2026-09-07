import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# TRAINING DATA
# 10 categories × 40 examples = 400 examples
# =========================================================

data = [

    # -----------------------------------------------------
    # NETWORK ISSUE - 40 examples
    # -----------------------------------------------------

    ("My Wi-Fi is not connecting", "Network Issue"),
    ("Internet is not working", "Network Issue"),
    ("I cannot connect to the Wi-Fi", "Network Issue"),
    ("My network connection keeps dropping", "Network Issue"),
    ("The internet connection is very slow", "Network Issue"),
    ("I have no internet access", "Network Issue"),
    ("Wi-Fi keeps disconnecting", "Network Issue"),
    ("My computer cannot connect to the network", "Network Issue"),
    ("The office network is down", "Network Issue"),
    ("Internet connection stopped working", "Network Issue"),
    ("Network connectivity problem", "Network Issue"),
    ("Wi-Fi signal is weak", "Network Issue"),
    ("I cannot access the internet", "Network Issue"),
    ("Network connection is unstable", "Network Issue"),
    ("My laptop lost network connection", "Network Issue"),
    ("Ethernet connection is not working", "Network Issue"),
    ("LAN connection failed", "Network Issue"),
    ("Internet keeps disconnecting", "Network Issue"),
    ("Unable to connect to office Wi-Fi", "Network Issue"),
    ("The network is unavailable", "Network Issue"),
    ("Wi-Fi says connected but internet does not work", "Network Issue"),
    ("My internet keeps going offline", "Network Issue"),
    ("Network access is failing", "Network Issue"),
    ("Cannot connect my laptop to Wi-Fi", "Network Issue"),
    ("The wireless network is not available", "Network Issue"),
    ("Internet speed is very slow", "Network Issue"),
    ("My network connection is poor", "Network Issue"),
    ("Wi-Fi connection is unstable", "Network Issue"),
    ("Computer is not getting network access", "Network Issue"),
    ("Office internet is not working", "Network Issue"),
    ("I am unable to connect to the network", "Network Issue"),
    ("The Wi-Fi connection stopped suddenly", "Network Issue"),
    ("Internet access is unavailable", "Network Issue"),
    ("My LAN cable is not providing internet", "Network Issue"),
    ("Network keeps timing out", "Network Issue"),
    ("Unable to browse websites because of network", "Network Issue"),
    ("Wireless connection keeps failing", "Network Issue"),
    ("My PC has no network connection", "Network Issue"),
    ("Internet connection is interrupted", "Network Issue"),
    ("Cannot reach the internet from my computer", "Network Issue"),


    # -----------------------------------------------------
    # PERFORMANCE ISSUE - 40 examples
    # -----------------------------------------------------

    ("My computer is very slow", "Performance Issue"),
    ("My laptop is running slowly", "Performance Issue"),
    ("The computer keeps freezing", "Performance Issue"),
    ("My system is lagging", "Performance Issue"),
    ("Applications are running very slowly", "Performance Issue"),
    ("The laptop takes too long to start", "Performance Issue"),
    ("My computer hangs frequently", "Performance Issue"),
    ("The system performance is poor", "Performance Issue"),
    ("Programs are responding slowly", "Performance Issue"),
    ("My PC freezes when I open applications", "Performance Issue"),
    ("Computer takes a long time to load", "Performance Issue"),
    ("Laptop is extremely slow", "Performance Issue"),
    ("System is constantly freezing", "Performance Issue"),
    ("My computer has become very slow", "Performance Issue"),
    ("Applications keep becoming unresponsive", "Performance Issue"),
    ("The system takes too long to respond", "Performance Issue"),
    ("My laptop is lagging badly", "Performance Issue"),
    ("Computer performance has decreased", "Performance Issue"),
    ("The PC is slow after startup", "Performance Issue"),
    ("Programs take forever to open", "Performance Issue"),
    ("My system is using too much memory", "Performance Issue"),
    ("Computer CPU usage is very high", "Performance Issue"),
    ("My laptop becomes slow during work", "Performance Issue"),
    ("The computer frequently hangs", "Performance Issue"),
    ("System response is very slow", "Performance Issue"),
    ("My PC is taking too long to perform tasks", "Performance Issue"),
    ("Applications are lagging", "Performance Issue"),
    ("Computer is freezing repeatedly", "Performance Issue"),
    ("Laptop startup is very slow", "Performance Issue"),
    ("My system has become unresponsive", "Performance Issue"),
    ("Computer performance is terrible", "Performance Issue"),
    ("My PC runs slowly when multiple programs are open", "Performance Issue"),
    ("The system is slow and keeps freezing", "Performance Issue"),
    ("Laptop is not performing properly", "Performance Issue"),
    ("My computer responds very slowly", "Performance Issue"),
    ("Programs are taking too long to load", "Performance Issue"),
    ("My PC is experiencing severe lag", "Performance Issue"),
    ("The system slows down frequently", "Performance Issue"),
    ("Computer freezes while working", "Performance Issue"),
    ("My laptop performance is poor", "Performance Issue"),


    # -----------------------------------------------------
    # PRINTER ISSUE - 40 examples
    # -----------------------------------------------------

    ("My printer is not printing", "Printer Issue"),
    ("The printer does not work", "Printer Issue"),
    ("I cannot print my document", "Printer Issue"),
    ("Printer is offline", "Printer Issue"),
    ("The printer is not responding", "Printer Issue"),
    ("Print job is stuck", "Printer Issue"),
    ("Printer keeps showing an error", "Printer Issue"),
    ("My document is not printing", "Printer Issue"),
    ("Printer is not detected", "Printer Issue"),
    ("The office printer is unavailable", "Printer Issue"),
    ("Printer connection failed", "Printer Issue"),
    ("I cannot connect to the printer", "Printer Issue"),
    ("Printer stopped working", "Printer Issue"),
    ("Print queue is stuck", "Printer Issue"),
    ("Printer prints blank pages", "Printer Issue"),
    ("Printer is showing offline status", "Printer Issue"),
    ("The printer is not responding to print commands", "Printer Issue"),
    ("My print job never starts", "Printer Issue"),
    ("Cannot send a document to printer", "Printer Issue"),
    ("Printer has a paper error", "Printer Issue"),
    ("Printer has no paper", "Printer Issue"),
    ("Printer ink is empty", "Printer Issue"),
    ("Printer toner is low", "Printer Issue"),
    ("Printer is producing incorrect output", "Printer Issue"),
    ("My printer cannot be found", "Printer Issue"),
    ("Printing stopped suddenly", "Printer Issue"),
    ("Printer queue is not processing", "Printer Issue"),
    ("Computer cannot communicate with printer", "Printer Issue"),
    ("The printer keeps disconnecting", "Printer Issue"),
    ("Printer driver is not working", "Printer Issue"),
    ("I am unable to print from my laptop", "Printer Issue"),
    ("Office printer is not working", "Printer Issue"),
    ("Network printer cannot be accessed", "Printer Issue"),
    ("Printer is stuck in offline mode", "Printer Issue"),
    ("Print command does nothing", "Printer Issue"),
    ("My printer is showing an error message", "Printer Issue"),
    ("Printer is not printing correctly", "Printer Issue"),
    ("Unable to print files", "Printer Issue"),
    ("Printer is unavailable", "Printer Issue"),
    ("The printer stopped responding", "Printer Issue"),


    # -----------------------------------------------------
    # ACCOUNT ISSUE - 40 examples
    # -----------------------------------------------------

    ("I forgot my password", "Account Issue"),
    ("I cannot login to my account", "Account Issue"),
    ("My password is not working", "Account Issue"),
    ("My account is locked", "Account Issue"),
    ("I cannot sign in", "Account Issue"),
    ("Login failed", "Account Issue"),
    ("I forgot my username", "Account Issue"),
    ("My account password has expired", "Account Issue"),
    ("I need to reset my password", "Account Issue"),
    ("Unable to access my account", "Account Issue"),
    ("My login credentials are not working", "Account Issue"),
    ("I cannot log into the system", "Account Issue"),
    ("Account login problem", "Account Issue"),
    ("My account has been locked", "Account Issue"),
    ("Password reset is not working", "Account Issue"),
    ("I am unable to sign in to my account", "Account Issue"),
    ("My username and password are rejected", "Account Issue"),
    ("Login credentials failed", "Account Issue"),
    ("I need help recovering my account", "Account Issue"),
    ("Cannot access my user account", "Account Issue"),
    ("My password stopped working", "Account Issue"),
    ("The system will not accept my password", "Account Issue"),
    ("I cannot authenticate my account", "Account Issue"),
    ("My account login keeps failing", "Account Issue"),
    ("I need a new password", "Account Issue"),
    ("My account is inaccessible", "Account Issue"),
    ("I am locked out of my account", "Account Issue"),
    ("Login is not possible", "Account Issue"),
    ("My credentials are invalid", "Account Issue"),
    ("I cannot enter my account", "Account Issue"),
    ("The login page rejects my password", "Account Issue"),
    ("My user account is locked", "Account Issue"),
    ("I forgot my login password", "Account Issue"),
    ("I cannot access the employee account", "Account Issue"),
    ("Password authentication failed", "Account Issue"),
    ("My account sign in is failing", "Account Issue"),
    ("Unable to log in with my credentials", "Account Issue"),
    ("Account access problem", "Account Issue"),
    ("I need assistance with my account login", "Account Issue"),
    ("My password needs to be reset", "Account Issue"),


    # -----------------------------------------------------
    # SECURITY ISSUE - 40 examples
    # -----------------------------------------------------

    ("I received a suspicious email", "Security Issue"),
    ("My computer may have a virus", "Security Issue"),
    ("I clicked a suspicious link", "Security Issue"),
    ("I think my account was hacked", "Security Issue"),
    ("There is malware on my computer", "Security Issue"),
    ("I found a phishing email", "Security Issue"),
    ("My computer has been infected", "Security Issue"),
    ("Someone may have accessed my account", "Security Issue"),
    ("I received a phishing message", "Security Issue"),
    ("My password may have been stolen", "Security Issue"),
    ("There is suspicious activity on my account", "Security Issue"),
    ("My computer is showing malware warnings", "Security Issue"),
    ("I opened a suspicious attachment", "Security Issue"),
    ("I think I clicked a malicious link", "Security Issue"),
    ("My account may be compromised", "Security Issue"),
    ("I suspect a security breach", "Security Issue"),
    ("Someone is trying to access my account", "Security Issue"),
    ("I received a suspicious login alert", "Security Issue"),
    ("My device is infected with malware", "Security Issue"),
    ("There is a possible virus on my laptop", "Security Issue"),
    ("I think this email is phishing", "Security Issue"),
    ("My account has suspicious activity", "Security Issue"),
    ("I downloaded a suspicious file", "Security Issue"),
    ("My computer is behaving like it has a virus", "Security Issue"),
    ("I suspect malware infection", "Security Issue"),
    ("There is an unauthorized login", "Security Issue"),
    ("My credentials may have been compromised", "Security Issue"),
    ("I clicked an unknown email link", "Security Issue"),
    ("I received a fake login email", "Security Issue"),
    ("My system detected a security threat", "Security Issue"),
    ("There may be ransomware on my computer", "Security Issue"),
    ("My antivirus detected malware", "Security Issue"),
    ("I suspect someone hacked my account", "Security Issue"),
    ("There is unauthorized access to my account", "Security Issue"),
    ("I found a suspicious attachment", "Security Issue"),
    ("My computer has a possible security infection", "Security Issue"),
    ("I received an unexpected password reset email", "Security Issue"),
    ("My account received an unknown login notification", "Security Issue"),
    ("I think my credentials were stolen", "Security Issue"),
    ("There is suspicious activity on my computer", "Security Issue"),


    # -----------------------------------------------------
    # HARDWARE ISSUE - 40 examples
    # -----------------------------------------------------

    ("My keyboard is not working", "Hardware Issue"),
    ("My mouse is not working", "Hardware Issue"),
    ("Laptop screen is damaged", "Hardware Issue"),
    ("Computer will not turn on", "Hardware Issue"),
    ("My laptop is overheating", "Hardware Issue"),
    ("The monitor is not displaying anything", "Hardware Issue"),
    ("My keyboard stopped working", "Hardware Issue"),
    ("USB port is not working", "Hardware Issue"),
    ("My laptop battery is not charging", "Hardware Issue"),
    ("The computer power button does not work", "Hardware Issue"),
    ("My mouse stopped responding", "Hardware Issue"),
    ("Laptop screen is flickering", "Hardware Issue"),
    ("Computer is making strange noises", "Hardware Issue"),
    ("My monitor is blank", "Hardware Issue"),
    ("The laptop charger is not working", "Hardware Issue"),
    ("My keyboard keys are not responding", "Hardware Issue"),
    ("Computer hardware failure", "Hardware Issue"),
    ("Laptop battery drains quickly", "Hardware Issue"),
    ("The webcam is not working", "Hardware Issue"),
    ("My microphone hardware is not working", "Hardware Issue"),
    ("Laptop is overheating badly", "Hardware Issue"),
    ("The screen keeps flickering", "Hardware Issue"),
    ("My external monitor is not detected", "Hardware Issue"),
    ("USB device is not recognized", "Hardware Issue"),
    ("Computer shuts down unexpectedly", "Hardware Issue"),
    ("My laptop does not power on", "Hardware Issue"),
    ("The display is broken", "Hardware Issue"),
    ("My charging port is damaged", "Hardware Issue"),
    ("The computer fan is making noise", "Hardware Issue"),
    ("Laptop charger stopped working", "Hardware Issue"),
    ("My keyboard is unresponsive", "Hardware Issue"),
    ("Mouse cursor is not moving", "Hardware Issue"),
    ("Laptop battery is not charging", "Hardware Issue"),
    ("My computer screen is black", "Hardware Issue"),
    ("The monitor has no signal", "Hardware Issue"),
    ("Laptop hardware is malfunctioning", "Hardware Issue"),
    ("My computer is overheating", "Hardware Issue"),
    ("The webcam stopped working", "Hardware Issue"),
    ("My USB ports are not working", "Hardware Issue"),
    ("Computer power problem", "Hardware Issue"),


    # -----------------------------------------------------
    # EMAIL ISSUE - 40 examples
    # -----------------------------------------------------

    ("I cannot send emails", "Email Issue"),
    ("I cannot receive emails", "Email Issue"),
    ("My email is not working", "Email Issue"),
    ("Email is not syncing", "Email Issue"),
    ("My mailbox is full", "Email Issue"),
    ("Email messages are not arriving", "Email Issue"),
    ("I cannot open my email", "Email Issue"),
    ("Email login is not working", "Email Issue"),
    ("My emails are delayed", "Email Issue"),
    ("The email application is not responding", "Email Issue"),
    ("I cannot send an email attachment", "Email Issue"),
    ("My inbox is not updating", "Email Issue"),
    ("Email synchronization failed", "Email Issue"),
    ("I am not receiving new messages", "Email Issue"),
    ("Outgoing emails are failing", "Email Issue"),
    ("Email server connection problem", "Email Issue"),
    ("My email account is not syncing", "Email Issue"),
    ("Emails are stuck in the outbox", "Email Issue"),
    ("I cannot access my mailbox", "Email Issue"),
    ("Email messages are missing", "Email Issue"),
    ("My email is very slow", "Email Issue"),
    ("Email client keeps crashing", "Email Issue"),
    ("I cannot download email attachments", "Email Issue"),
    ("My inbox is not refreshing", "Email Issue"),
    ("Email delivery failed", "Email Issue"),
    ("Emails are not being sent", "Email Issue"),
    ("I cannot receive messages from coworkers", "Email Issue"),
    ("Email application stopped working", "Email Issue"),
    ("My email password is not accepted", "Email Issue"),
    ("Mailbox storage is full", "Email Issue"),
    ("Email account is not connecting", "Email Issue"),
    ("The mail server is unavailable", "Email Issue"),
    ("Email messages are delayed", "Email Issue"),
    ("My inbox is empty unexpectedly", "Email Issue"),
    ("Cannot attach files to email", "Email Issue"),
    ("Email keeps disconnecting", "Email Issue"),
    ("I am unable to access company email", "Email Issue"),
    ("Email sync keeps failing", "Email Issue"),
    ("My outgoing mail is not working", "Email Issue"),
    ("I have a problem with my email", "Email Issue"),


    # -----------------------------------------------------
    # SOFTWARE ISSUE - 40 examples
    # -----------------------------------------------------

    ("The software is not opening", "Software Issue"),
    ("An application keeps crashing", "Software Issue"),
    ("I cannot install the software", "Software Issue"),
    ("The application is not working", "Software Issue"),
    ("Software installation failed", "Software Issue"),
    ("The program crashes when I open it", "Software Issue"),
    ("An application stopped working", "Software Issue"),
    ("The software shows an error", "Software Issue"),
    ("I need help installing an application", "Software Issue"),
    ("The program is not responding", "Software Issue"),
    ("My application keeps closing", "Software Issue"),
    ("Software update failed", "Software Issue"),
    ("The application will not start", "Software Issue"),
    ("I cannot update the software", "Software Issue"),
    ("The program has stopped responding", "Software Issue"),
    ("Application installation problem", "Software Issue"),
    ("The software is showing an error message", "Software Issue"),
    ("My application is broken", "Software Issue"),
    ("The program crashes frequently", "Software Issue"),
    ("I cannot launch the application", "Software Issue"),
    ("Software is not functioning correctly", "Software Issue"),
    ("The application freezes", "Software Issue"),
    ("Program installation is failing", "Software Issue"),
    ("The software cannot be updated", "Software Issue"),
    ("Application startup failed", "Software Issue"),
    ("The program does not open", "Software Issue"),
    ("My software keeps crashing", "Software Issue"),
    ("Application is displaying errors", "Software Issue"),
    ("I am unable to install a program", "Software Issue"),
    ("Software stopped working after update", "Software Issue"),
    ("The application is unresponsive", "Software Issue"),
    ("My program is not working properly", "Software Issue"),
    ("Software installation error", "Software Issue"),
    ("The application cannot start", "Software Issue"),
    ("My software is malfunctioning", "Software Issue"),
    ("Program keeps freezing", "Software Issue"),
    ("I cannot open the application", "Software Issue"),
    ("The application crashes on startup", "Software Issue"),
    ("Software update is not installing", "Software Issue"),
    ("My program has an error", "Software Issue"),


    # -----------------------------------------------------
    # ACCESS / PERMISSION ISSUE - 40 examples
    # -----------------------------------------------------

    ("I do not have permission to access the folder", "Access/Permission Issue"),
    ("I cannot access the shared drive", "Access/Permission Issue"),
    ("Access is denied", "Access/Permission Issue"),
    ("I need permission to access a file", "Access/Permission Issue"),
    ("I cannot open the shared folder", "Access/Permission Issue"),
    ("My account does not have access", "Access/Permission Issue"),
    ("Permission denied when opening a file", "Access/Permission Issue"),
    ("I need access to the application", "Access/Permission Issue"),
    ("I cannot access the company portal", "Access/Permission Issue"),
    ("The system says access denied", "Access/Permission Issue"),
    ("I need authorization to access a resource", "Access/Permission Issue"),
    ("My access request was denied", "Access/Permission Issue"),
    ("I cannot access a shared file", "Access/Permission Issue"),
    ("Permission is missing", "Access/Permission Issue"),
    ("I need administrator permission", "Access/Permission Issue"),
    ("Access to the folder is blocked", "Access/Permission Issue"),
    ("I cannot access the network drive", "Access/Permission Issue"),
    ("My user account has insufficient permissions", "Access/Permission Issue"),
    ("I need access rights for a system", "Access/Permission Issue"),
    ("The application says I do not have permission", "Access/Permission Issue"),
    ("I cannot access a restricted folder", "Access/Permission Issue"),
    ("Permission error when opening a document", "Access/Permission Issue"),
    ("My access rights are not working", "Access/Permission Issue"),
    ("I need authorization for a file", "Access/Permission Issue"),
    ("Access denied to shared resources", "Access/Permission Issue"),
    ("I cannot open the company shared drive", "Access/Permission Issue"),
    ("The system is refusing access", "Access/Permission Issue"),
    ("I need permission for a database", "Access/Permission Issue"),
    ("My account cannot access the application", "Access/Permission Issue"),
    ("I cannot access a restricted resource", "Access/Permission Issue"),
    ("Permission denied on the server", "Access/Permission Issue"),
    ("I need additional access rights", "Access/Permission Issue"),
    ("My request for access was rejected", "Access/Permission Issue"),
    ("I cannot access the shared network folder", "Access/Permission Issue"),
    ("The file says permission denied", "Access/Permission Issue"),
    ("I need access approval", "Access/Permission Issue"),
    ("My account has no permission to open this file", "Access/Permission Issue"),
    ("Access rights are missing", "Access/Permission Issue"),
    ("I cannot access the internal portal", "Access/Permission Issue"),
    ("Permission problem with shared folder", "Access/Permission Issue"),


    # -----------------------------------------------------
    # DATABASE ISSUE - 40 examples
    # -----------------------------------------------------

    ("I cannot connect to the database", "Database Issue"),
    ("Database connection failed", "Database Issue"),
    ("The database is not responding", "Database Issue"),
    ("I cannot access the database", "Database Issue"),
    ("Database server is down", "Database Issue"),
    ("Database connection is very slow", "Database Issue"),
    ("My database login is failing", "Database Issue"),
    ("Database query is not working", "Database Issue"),
    ("The database server is unavailable", "Database Issue"),
    ("I am getting a database connection error", "Database Issue"),
    ("Unable to connect to SQL database", "Database Issue"),
    ("Database timeout error", "Database Issue"),
    ("Database is offline", "Database Issue"),
    ("SQL connection failed", "Database Issue"),
    ("Cannot execute database query", "Database Issue"),
    ("Database access problem", "Database Issue"),
    ("Database server connection problem", "Database Issue"),
    ("The database is running slowly", "Database Issue"),
    ("Database credentials are not working", "Database Issue"),
    ("I cannot log into the database", "Database Issue"),
    ("Database query failed", "Database Issue"),
    ("SQL server is not responding", "Database Issue"),
    ("Database service is unavailable", "Database Issue"),
    ("My application cannot reach the database", "Database Issue"),
    ("Database connection keeps timing out", "Database Issue"),
    ("I am unable to access SQL server", "Database Issue"),
    ("Database authentication failed", "Database Issue"),
    ("The database connection was lost", "Database Issue"),
    ("Database server stopped responding", "Database Issue"),
    ("Unable to retrieve data from database", "Database Issue"),
    ("Database request is failing", "Database Issue"),
    ("SQL database is unavailable", "Database Issue"),
    ("Database query takes too long", "Database Issue"),
    ("Cannot connect to MySQL database", "Database Issue"),
    ("Database server has an error", "Database Issue"),
    ("My database session is disconnected", "Database Issue"),
    ("Unable to run database commands", "Database Issue"),
    ("Database service stopped", "Database Issue"),
    ("The SQL database connection is not working", "Database Issue"),
    ("I have a database connectivity problem", "Database Issue"),
]


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(
    data,
    columns=["text", "category"]
)

print("\n========================================")
print("AI-IT HELPDESK MODEL TRAINING")
print("========================================")

print(f"\nTotal training examples: {len(df)}")
print(f"Total categories: {df['category'].nunique()}")

print("\nExamples per category:")
print(df["category"].value_counts())


# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["category"],
    test_size=0.20,
    random_state=42,
    stratify=df["category"]
)


# =========================================================
# TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    class_weight="balanced"
)

model.fit(
    X_train_vectorized,
    y_train
)


# =========================================================
# MODEL EVALUATION
# =========================================================

y_pred = model.predict(
    X_test_vectorized
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# TEST SAMPLE PROBLEMS
# =========================================================

test_problems = [
    "My Wi-Fi is not connecting to the internet.",
    "My laptop is extremely slow and keeps freezing.",
    "The printer is not printing my document.",
    "I forgot my password and cannot login.",
    "I received a suspicious phishing email.",
    "My laptop battery is not charging.",
    "I cannot send or receive emails.",
    "The application keeps crashing.",
    "I cannot access the shared folder.",
    "I cannot connect to the database."
]

print("\n========================================")
print("SAMPLE PREDICTIONS")
print("========================================")

for problem in test_problems:

    problem_vector = vectorizer.transform(
        [problem]
    )

    prediction = model.predict(
        problem_vector
    )[0]

    probabilities = model.predict_proba(
        problem_vector
    )[0]

    confidence = max(probabilities) * 100

    print("\nProblem:", problem)
    print("Predicted Category:", prediction)
    print(f"Confidence: {confidence:.2f}%")


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "helpdesk_model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print("\nFiles created/updated:")
print("1. helpdesk_model.pkl")
print("2. vectorizer.pkl")

print("\nAI Helpdesk model training completed!")
# =========================================================
# PREDICTION FUNCTION FOR TOOLS / AGENT
# =========================================================

def predict_issue(user_message):
    """
    Predict the IT issue category and confidence.
    Used by the AI Helpdesk Agent tools.
    """

    problem_vector = vectorizer.transform(
        [user_message]
    )

    prediction = model.predict(
        problem_vector
    )[0]

    probabilities = model.predict_proba(
        problem_vector
    )[0]

    confidence = max(probabilities) * 100

    return prediction, confidence
