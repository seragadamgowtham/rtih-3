import sqlite3
def clean():
    conn = sqlite3.connect('medmap_ai.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM medicines WHERE mid='MED021'")
    conn.commit()
    conn.close()
    
if __name__ == '__main__':
    clean()
    print("Cleaned.")
