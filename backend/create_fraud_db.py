import sqlite3
import datetime

conn = sqlite3.connect("fraud_cases.db")
cursor = conn.cursor()

# --------------------
# Create fraud_cases table
# --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS fraud_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userName TEXT,
    securityIdentifier TEXT,
    securityQuestion TEXT,
    securityAnswer TEXT,
    maskedCard TEXT,
    transactionAmount TEXT,
    transactionMerchant TEXT,
    transactionLocation TEXT,
    transactionTime TEXT,
    transactionCategory TEXT,
    status TEXT,
    note TEXT,
    updated_at TEXT
);
""")

# --------------------
# Insert multiple sample cases
# --------------------

cases = [
    (
        "John Doe",
        "ID12345",
        "What is your favorite color?",
        "blue",
        "**** 4242",
        "₹5,499",
        "ABC Electronics",
        "Mumbai",
        "2025-11-26 15:45",
        "e-commerce",
        "pending_review",
        "",
        datetime.datetime.utcnow().isoformat()
    ),
    (
        "Anita Sharma",
        "ID99887",
        "What city were you born in?",
        "delhi",
        "**** 8821",
        "₹12,250",
        "TravelGo Flights",
        "Bangalore",
        "2025-11-26 09:12",
        "travel",
        "pending_review",
        "",
        datetime.datetime.utcnow().isoformat()
    ),
    (
        "Ravi Patel",
        "ID55901",
        "What is your pet's name?",
        "tiger",
        "**** 1199",
        "₹799",
        "FitLife Gym",
        "Ahmedabad",
        "2025-11-25 20:05",
        "fitness",
        "pending_review",
        "",
        datetime.datetime.utcnow().isoformat()
    ),
    (
        "Sara Khan",
        "ID44321",
        "What is your favorite fruit?",
        "mango",
        "**** 3390",
        "₹2,349",
        "StyleStreet Fashion",
        "Hyderabad",
        "2025-11-26 14:30",
        "shopping",
        "pending_review",
        "",
        datetime.datetime.utcnow().isoformat()
    ),
    (
        "Arjun Mehta",
        "ID22119",
        "What is your best friend's name?",
        "rahul",
        "**** 7712",
        "₹15,999",
        "TechMart Gadgets",
        "Pune",
        "2025-11-27 10:18",
        "electronics",
        "pending_review",
        "",
        datetime.datetime.utcnow().isoformat()
    ),
]

cursor.executemany("""
INSERT INTO fraud_cases (
    userName, securityIdentifier, securityQuestion, securityAnswer,
    maskedCard, transactionAmount, transactionMerchant,
    transactionLocation, transactionTime, transactionCategory,
    status, note, updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", cases)

conn.commit()
conn.close()

print("fraud_cases.db created with 5 sample fraud cases.")
