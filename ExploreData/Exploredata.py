import pandas as pd

# Load the data, skipping the first row (index 0) and using the second row (index 1) as header
file_path = 'default_of_credit_card_clients.csv'

# Since the user says first row is redundant and second is header:
# Usually, pd.read_csv(..., skiprows=1) makes the new first row the header.
df = pd.read_csv(file_path, skiprows=1)

# Inspect the first few rows and info
print(df.head())
print(df.info())
print(df.describe())

...