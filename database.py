import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'whatsapp_analytics.db')

def get_connection():
    """Returns an active SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_name TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_messages INTEGER,
        total_words INTEGER,
        media_count INTEGER,
        links_count INTEGER,
        users_count INTEGER,
        overall_sentiment REAL,
        dominant_sentiment TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        user_name TEXT NOT NULL,
        message_count INTEGER,
        avg_sentiment REAL,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chatbot_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        user_query TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        query_type TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()

def save_analysis_session(session_name: str, stats: dict, sentiment_data: dict, user_percent_df: pd.DataFrame) -> int:
    """Saves a completed chat analysis session and its user breakdowns."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    # Determine dominant sentiment label
    pos = sentiment_data.get('pos_percent', 0)
    neu = sentiment_data.get('neu_percent', 0)
    neg = sentiment_data.get('neg_percent', 0)
    if pos >= neu and pos >= neg:
        dom_sent = 'Positive'
    elif neg >= pos and neg >= neu:
        dom_sent = 'Negative'
    else:
        dom_sent = 'Neutral'

    cursor.execute("""
        INSERT INTO chat_sessions (session_name, total_messages, total_words, media_count, links_count, users_count, overall_sentiment, dominant_sentiment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_name,
        stats.get('total_messages', 0),
        stats.get('total_words', 0),
        stats.get('media_count', 0),
        stats.get('links_count', 0),
        stats.get('users_count', 0),
        sentiment_data.get('overall_score', 0.0),
        dom_sent
    ))
    
    session_id = cursor.lastrowid

    # Insert user statistics
    if not user_percent_df.empty:
        user_sentiment_map = {}
        if 'user_sentiment' in sentiment_data and not sentiment_data['user_sentiment'].empty:
            for _, row in sentiment_data['user_sentiment'].iterrows():
                user_sentiment_map[row['user']] = row['avg_polarity']

        for _, row in user_percent_df.iterrows():
            u_name = row['user']
            cursor.execute("""
                INSERT INTO user_stats (session_id, user_name, message_count, avg_sentiment)
                VALUES (?, ?, ?, ?)
            """, (
                session_id,
                u_name,
                int(row.get('count', 0)) if 'count' in row else 0,
                float(user_sentiment_map.get(u_name, 0.0))
            ))

    conn.commit()
    conn.close()
    return session_id

def log_chatbot_query(session_id: int, query: str, response: str, query_type: str = 'General'):
    """Logs a chatbot interaction into the database."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chatbot_history (session_id, user_query, bot_response, query_type)
        VALUES (?, ?, ?, ?)
    """, (session_id, query, response, query_type))
    conn.commit()
    conn.close()

def get_all_sessions() -> pd.DataFrame:
    """Fetches all recorded analysis sessions."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM chat_sessions ORDER BY uploaded_at DESC", conn)
    conn.close()
    return df

def get_session_users(session_id: int) -> pd.DataFrame:
    """Fetches user stats for a specific session."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM user_stats WHERE session_id = ?", conn, params=(session_id,))
    conn.close()
    return df

def get_chatbot_history(session_id: int = None) -> pd.DataFrame:
    """Fetches chatbot interaction history."""
    init_db()
    conn = get_connection()
    if session_id:
        df = pd.read_sql_query("SELECT * FROM chatbot_history WHERE session_id = ? ORDER BY timestamp DESC", conn, params=(session_id,))
    else:
        df = pd.read_sql_query("SELECT * FROM chatbot_history ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()
    return df

def delete_session(session_id: int):
    """Deletes a session and related records."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    """Resets all database tables."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chatbot_history")
    cursor.execute("DELETE FROM user_stats")
    cursor.execute("DELETE FROM chat_sessions")
    conn.commit()
    conn.close()
