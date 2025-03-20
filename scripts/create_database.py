import sqlite3
import seaborn as sns
import pandas as pd

# Load dataset
penguins = sns.load_dataset("penguins").dropna()

# Create SQLite connection
conn = sqlite3.connect("data/penguins.db")
cursor = conn.cursor()

# Create table schema
cursor.execute("""
    CREATE TABLE IF NOT EXISTS penguins (
        species TEXT,
        bill_length_mm REAL,
        bill_depth_mm REAL,
        flipper_length_mm REAL,
        body_mass_g REAL
    )
""")

# Insert data
penguins[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].to_sql("penguins", conn, if_exists="replace", index=False)

# Close connection
conn.close()

print("Database created successfully!")