import pandas as pd
import random
from datetime import datetime, timedelta

# Payment modes and constants
modes = ['UPI', 'Cash', 'Credit Card', 'Netbanking', 'Debit Card']
currency = 'INR'
income_expense = ['Expense'] * 95 + ['Income'] * 5

# Category/Subcategory pairs
categories_subcategories = [
    ('Apparel', 'Clothing'), ('Apparel', 'Footwear'), ('Beauty', 'grooming'),
    ('Culture', 'Movie'), ('Education', 'Stationary'), ('Family', 'Pocket money'),
    ('Festivals', 'Diwali'), ('Festivals', 'Holi'), ('Food', 'Milk'), 
    ('Food', 'snacks'), ('Food', 'Grocery'), ('Food', 'Lunch'), ('Food', 'Dinner'),
    ('Gift', 'Gift'), ('Grooming', 'Saloon'), ('Health', 'Medicine'),
    ('Household', 'Kirana'), ('Household', 'Appliances'), ('Investment', 'Mutual fund'),
    ('Money transfer', 'Money transfer'), ('Other', 'Donation'),
    ('Public Provident Fund', 'Public Provident Fund'), ('Recurring Deposit', 'Recurring Deposit'),
    ('Rent', 'Rent'), ('Self-development', 'Self-development'),
    ('Social Life', 'Leisure'), ('Tourism', 'Trip'), ('Transportation', 'auto'),
    ('Transportation', 'Train'), ('Transportation', 'Taxi'),
    ('subscription', 'Netflix'), ('subscription', 'Amazon Prime'), ('subscription', 'Spotify'),
    ('subscription', 'Tata Sky'), ('subscription', 'Edtech Course'), ('subscription', 'Airtel'), ('subscription', 'DishTV'),
    ('water (jar /tanker)', 'water (jar /tanker)')
]

# Subscription-specific plans and prices
subscription_plans = {
    'Netflix': [149, 199, 499, 649],
    'Amazon Prime': [299, 599, 1499],
    'Spotify': [7, 25, 119, 149, 179],
    'Hotstar': [149, 299, 499],
    'Audible': [199],
    'Wifi Internet Service': [399, 599, 799],
    'Kindle unlimited': [169],
}

notes = ['Paid bill', 'Bought groceries', 'UPI transfer', 'Lunch at cafe', 'Movie night',
         'Online course', 'Subscription renewal', 'Festival shopping', 'Train ticket', 'Medical bill']

# Date setup
start_date = datetime.today() - timedelta(days=180)
def random_date():
    return (start_date + timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d')

# List to collect data
synthetic_data = []

# Step 1: Ensure at least 15 entries for each subcategory
for category, subcategory in categories_subcategories:
    for _ in range(20):
        date = random_date()
        mode = random.choice(modes)
        note = random.choice(notes)
        income_or_expense = random.choice(income_expense)
        
        if category == 'subscription' and subcategory in subscription_plans:
            amount = random.choice(subscription_plans[subcategory])
        else:
            amount = round(random.uniform(50, 5000), 2)
        
        synthetic_data.append({
            "Date": date,
            "Mode": mode,
            "Category": category,
            "Subcategory": subcategory,
            "Note": note,
            "Amount": amount,
            "Income/Expense": income_or_expense,
            "Currency": currency
        })

# Step 2: Add random entries to reach total num_records
min_records = len(synthetic_data)
num_records = 5500

while len(synthetic_data) < num_records:
    category, subcategory = random.choice(categories_subcategories)
    date = random_date()
    mode = random.choice(modes)
    note = random.choice(notes)
    income_or_expense = random.choice(income_expense)
    
    if category == 'subscription' and subcategory in subscription_plans:
        amount = random.choice(subscription_plans[subcategory])
    else:
        amount = round(random.uniform(50, 5000), 2)

    synthetic_data.append({
        "Date": date,
        "Mode": mode,
        "Category": category,
        "Subcategory": subcategory,
        "Note": note,
        "Amount": amount,
        "Income/Expense": income_or_expense,
        "Currency": currency
    })

# Save to CSV
df = pd.DataFrame(synthetic_data)
csv_path = "artifacts/synthetic-data.csv"
df.to_csv(csv_path, index=False)
print(f"Saved {len(df)} records to: {csv_path}")
