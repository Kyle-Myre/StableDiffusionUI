from rich.console import Console
from paths import DATABASE_URL
import sqlite3

console = Console()

def load_session():
    try:
        path = DATABASE_URL
        connection = sqlite3.connect(path)
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS session (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                TaskDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                Steps INTEGER,
                Guidance INTEGER,
                PositivePrompt TEXT,
                NegativePrompt TEXT
            );
        """)
        return connection
    except sqlite3.Error as error:
        console.print(error)

def latest_records():
    try:
        database = load_session()
        cursor   = database.cursor()
        query = cursor.execute("SELECT * FROM session ORDER BY TaskDate DESC LIMIT 5;").fetchall()
        result = [item for item in query]
        data = [("Task Date", "Time Task", "Steps", "Guidance", "Positive Prompt" , "Negative Prompt")]
        return data + result
    except sqlite3.Error as error:
        console.print(error)


def insert_record(positive_prompt:str , negative_prompt:str , steps:int , guidance:int):
    try:
        database = load_session()
        cursor   = database.cursor()
        cursor.execute(
            "INSERT INTO session(ID , Steps , Guidance , PositivePrompt , NegativePrompt) VALUES(?,?,?,?,?)" , 
            [None , steps , guidance , positive_prompt , negative_prompt]
        )
        database.commit()
    except sqlite3.Error as error:
        console.print(error)