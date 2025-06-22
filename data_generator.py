import pandas as pd
import random
from datetime import datetime, timedelta

# Define parameters
NUM_ENTRIES = 1000

modes = ['Cash', 'Credit Card', 'Saving Bank account 1']
categories = {
    'Food': ['snacks', 'Milk', 'Grocery', 'Lunch'],
    'Transportation': ['Train', 'Bus', 'Auto'],
    'subscription': ['Netflix', 'Spotify', 'Mobile Service Provider', 'Amazon Prime'],
    'Festivals': ['Diwali', 'Holi', 'Ganesh Pujan'],
    'Apparel': ['Laundry'],
    'Other': ['Gift', 'Donation', 'Misc'],
    'Income': ['Salary', 'Freelance', 'Bonus', 'Gpay Reward'],
    'Investment': ['Equity Mutual Fund E', 'Small Cap fund 2'],
}
notes = {
    'snacks': ['Vadapav & Tea', 'Chips & Cola', 'Burger', 'Sandwich'],
    'Milk': ['1L Amul milk', 'Half litre milk'],
    'Train': ['To Office', 'Place A to B'],
    'Bus': ['Commute to Market', 'To Tuition'],
    'Auto': ['Home to Station', 'Office to Home'],
    'Netflix': ['Monthly plan', 'Annual plan'],
    'Salary': ['Monthly salary'],
    'Laundry': ['5 clothes washed', 'Blanket cleaned'],
    'Gift': ['Birthday Gift', 'Festival Gift'],
    'Donation': ['To Temple', 'Street Charity'],
    'Grocery': ['Fruits & Veggies', 'Monthly stock'],
    'Lunch': ['Home Delivery', 'Restaurant'],
    'Freelance': ['Client payment', 'Upwork payout'],
    'Bonus': ['Year-end bonus'],
    'Gpay Reward': ['Cashback'],
}

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

start_date = datetime(2018, 9, 1)
end_date = datetime(2022, 12, 31)

# Generate rows
data = []
for _ in range(NUM_ENTRIES):
    date = random_date(start_date, end_date).strftime("%d/%m/%Y %H:%M:%S")
    mode = random.choice(modes)
    category = random.choice(list(categories.keys()))
    subcategory = random.choice(categories[category])
    note = random.choice(notes.get(subcategory, ['']))
    
    if category == 'Income':
        amount = round(random.uniform(1000, 50000), 2)
        income_expense = 'Income'
    elif category == 'Investment':
        amount = round(random.uniform(500, 10000), 2)
        income_expense = 'Transfer-Out'
    else:
        amount = round(random.uniform(10, 1000), 2)
        income_expense = 'Expense'
    
    data.append([
        date,
        mode,
        category,
        subcategory,
        note,
        amount,
        income_expense,
        'INR'
    ])

# Convert to DataFrame
df = pd.DataFrame(data, columns=[
    'Date', 'Mode', 'Category', 'Subcategory', 'Note',
    'Amount', 'Income/Expense', 'Currency'
])

# Save to CSV
df.to_csv("artifacts/synthetic-data.csv", index=False)
print("✅ Generated 1000 synthetic records to 'synthetic-data.csv'")
