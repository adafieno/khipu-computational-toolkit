import sqlite3
conn = sqlite3.connect('khipu.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall() if not r[0].startswith('sqlite_')]
print("Database tables:")
for t in tables:
    print(f"  - {t}")
conn.close()
