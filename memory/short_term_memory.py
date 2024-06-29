"""
short_term_memory.py - utils for saving stms

Memoir+ a persona extension for Text Gen Web UI. 
MIT License

Copyright (c) 2024 brucepro
"""
import sqlite3
import os


class ShortTermMemory:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.create_sqlite_db_if_missing()

    def create_sqlite_db_if_missing(self):
        if not os.path.exists(self.db_name):
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Create ShortTermMemory table
            cursor.execute('''CREATE TABLE "short_term_memory" (
                                "id" INTEGER NOT NULL UNIQUE,
                                "memory_text" TEXT NOT NULL,
                                "DateTime" DATETIME DEFAULT CURRENT_TIMESTAMP,
                                "people" TEXT NOT NULL,
                                "memory_type" TEXT,
                                "initiated_by" TEXT,
                                "roleplay" TEXT,
                                "saved_to_longterm" INTEGER,
                                PRIMARY KEY("id" AUTOINCREMENT)
                            )''')

            conn.commit()
            conn.close()

    def connect(self):
        try:
            self.conn = sqlite3.connect(f'{self.db_name}')
            self.cursor = self.conn.cursor()
            # print('Connected to the database!')
        except Exception as e:
            print(e)

    def disconnect(self):
        try:
            if self.conn is not None and self.cursor is not None:
                self.cursor = None
                self.conn.close()
                # print('Disconnected from the database!')
        except Exception as e:
            print(e)

    def save_memory(self, memory_text, people, memory_type, initiated_by, roleplay, saved_to_longterm=0):
        try:
            self.connect()
            sql = "INSERT INTO short_term_memory (memory_text, people, memory_type, initiated_by, roleplay, saved_to_longterm) VALUES (?, ?, ?, ?, ?, 0)"
            values = (memory_text, people, memory_type, initiated_by, roleplay)
            self.cursor.execute(sql, values)
            self.conn.commit()
            # print('Memory saved successfully!')
        except Exception as e:
            print(e)
        finally:
            self.disconnect()

    def update_mem_saved_to_longterm(self, id):
        try:
            self.connect()
            sql = f"UPDATE short_term_memory SET saved_to_longterm=1 WHERE id={id}"
            self.cursor.execute(sql)
            self.conn.commit()
            # print('Memory updated successfully!')
        except Exception as e:
            print(e)
        finally:
            self.disconnect()
