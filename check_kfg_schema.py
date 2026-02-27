import sqlite3

conn = sqlite3.connect('data/kfg/khipu_database.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("KFG Database Tables:")
print("=" * 80)
for table in tables:
    table_name = table[0]
    print(f"\n{table_name}:")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

conn.close()
