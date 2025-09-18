import contextlib
import sqlite3

import matplotlib.pyplot as plt

# need to have a files table
# and a metrics table



with sqlite3.connect('file.db') as conn, contextlib.closing(conn.cursor()) as cur:
    cur.execute(f'''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            path TEXT NOT NULL,
            series_name TEXT NOT NULL
        );
   ''')

    cur.execute(f'''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f','now')),
            metric REAL NOT NULL,
            series_name TEXT NOT NULL
        );
   ''')

    conn.commit()


def add_metric(series_name, metric):
    with sqlite3.connect('file.db') as conn, contextlib.closing(conn.cursor()) as cur:
        cur.execute(f'INSERT INTO metrics (metric, series_name) values (?, ?)', (metric, series_name))
       # conn.commit()


def add_file(series_name, path):
    with sqlite3.connect('file.db') as conn, contextlib.closing(conn.cursor()) as cur:
        cur.execute(f'INSERT INTO files (path, series_name) values (?, ?)', (path, series_name))


def get_metrics(series_name):
    with sqlite3.connect('file.db') as conn, contextlib.closing(conn.cursor()) as cur:
        cur.execute(f'SELECT id, timestamp, metric FROM metrics WHERE series_name = ?', (series_name,))
        return cur.fetchall()

def get_files(series_name):
    with sqlite3.connect('file.db') as conn, contextlib.closing(conn.cursor()) as cur:
        cur.execute(f'SELECT id, timestamp, path FROM files WHERE series_name = ?', (series_name,))
        return cur.fetchall()


value = 30

add_metric('loss', value)
add_metric('loss', value)
add_metric('loss', value)
add_metric('loss', value)
add_metric('loss', value)
add_metric('loss', value)
add_metric('loss', value)
add_metric('loss', value)
print(get_metrics('loss'))

plt.plot([m[1] for m in get_metrics('loss')], [m[2] for m in get_metrics('loss')])
plt.title('loss')
plt.show()
