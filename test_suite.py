import unittest
import pandas as pd
import os
import shutil

from preprocessor import parse_chat
from analyzer import (
    fetch_stats, most_busy_users, most_common_words, emoji_helper,
    monthly_timeline, daily_timeline, week_activity_map,
    month_activity_map, activity_heatmap, analyze_sentiment
)
import database
from chatbot import answer_question, identify_intent

SAMPLE_RAW_CHAT = """15/01/24, 09:30 - Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.
15/01/24, 09:30 - Alice: Good morning team! Let's get started on the AI project 🚀
15/01/24, 09:31 - Bob: Morning Alice! I loved the new UI design. Check out https://github.com/example/ui
15/01/24, 09:32 - Charlie: <Media omitted>
15/01/24, 09:35 - Alice: Great progress. Sentiment analysis is super positive today! 😊
16/01/24, 14:20 - Bob: What time is our deployment review?
16/01/24, 14:22 - Alice: 4 PM works best.
"""

class TestWhatsAppAnalyzer(unittest.TestCase):

    def setUp(self):
        self.df = parse_chat(SAMPLE_RAW_CHAT)

    def test_preprocessor(self):
        self.assertFalse(self.df.empty, "DataFrame should not be empty")
        self.assertIn('user', self.df.columns)
        self.assertIn('message', self.df.columns)
        self.assertIn('is_media', self.df.columns)
        self.assertIn('url_count', self.df.columns)
        self.assertIn('emoji_count', self.df.columns)

        # Users parsed
        users = set(self.df['user'].unique())
        self.assertIn('Alice', users)
        self.assertIn('Bob', users)
        self.assertIn('Charlie', users)

        # Check media detection
        charlie_msgs = self.df[self.df['user'] == 'Charlie']
        self.assertTrue(charlie_msgs['is_media'].iloc[0])

        # Check URL detection
        bob_msgs = self.df[self.df['user'] == 'Bob']
        self.assertGreater(bob_msgs['url_count'].sum(), 0)

    def test_analyzer_stats(self):
        stats = fetch_stats('Overall', self.df)
        self.assertEqual(stats['users_count'], 3)
        self.assertEqual(stats['media_count'], 1)
        self.assertGreater(stats['total_messages'], 0)
        self.assertGreater(stats['total_words'], 0)

    def test_user_activity(self):
        busy_users, user_pct = most_busy_users(self.df)
        self.assertFalse(busy_users.empty)
        self.assertIn('Alice', busy_users.index)

    def test_emoji_and_words(self):
        emojis = emoji_helper('Overall', self.df)
        self.assertFalse(emojis.empty)
        self.assertIn('🚀', emojis['emoji'].values)

        words = most_common_words('Overall', self.df)
        self.assertFalse(words.empty)

    def test_sentiment_analysis(self):
        sent = analyze_sentiment('Overall', self.df)
        self.assertIn('overall_score', sent)
        self.assertIn('pos_percent', sent)
        self.assertIn('user_sentiment', sent)
        self.assertGreaterEqual(sent['pos_percent'], 0)

    def test_database_operations(self):
        database.init_db()
        stats = fetch_stats('Overall', self.df)
        sent = analyze_sentiment('Overall', self.df)
        _, user_pct = most_busy_users(self.df)

        session_id = database.save_analysis_session("Test Session", stats, sent, user_pct)
        self.assertIsInstance(session_id, int)

        sessions = database.get_all_sessions()
        self.assertGreaterEqual(len(sessions), 1)

        database.log_chatbot_query(session_id, "Who sent the most messages?", "Alice sent the most.", "top_sender")
        chat_hist = database.get_chatbot_history(session_id)
        self.assertGreaterEqual(len(chat_hist), 1)

    def test_chatbot_engine(self):
        ans1, intent1 = answer_question("Who sent the most messages?", self.df)
        self.assertEqual(intent1, "top_sender")
        self.assertIn("Alice", ans1)

        ans2, intent2 = answer_question("How many messages are in the chat?", self.df)
        self.assertEqual(intent2, "total_messages")

        ans3, intent3 = answer_question("What is the overall sentiment?", self.df)
        self.assertEqual(intent3, "sentiment")

        ans4, intent4 = answer_question("What is the most used emoji?", self.df)
        self.assertEqual(intent4, "top_emoji")

if __name__ == '__main__':
    unittest.main()
