import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

from preprocessor import parse_chat
from analyzer import (
    fetch_stats, most_busy_users, create_wordcloud, most_common_words,
    emoji_helper, monthly_timeline, daily_timeline, week_activity_map,
    month_activity_map, activity_heatmap, analyze_sentiment
)
import database
from chatbot import answer_question

# Page Configuration
st.set_page_config(
    page_title="OmniRoute - AI WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Header & Badge Styling */
    .badge-positive {
        background-color: #059669;
        color: white;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neutral {
        background-color: #64748b;
        color: white;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-negative {
        background-color: #dc2626;
        color: white;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid rgba(56, 189, 248, 0.3);
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database Schema
try:
    database.init_db()
except Exception:
    pass

# Session State Initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_filename' not in st.session_state:
    st.session_state.chat_filename = None
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Sidebar Navigation & Upload
with st.sidebar:
    st.title("💬 Chat Analyzer")
    st.markdown("AI-Powered WhatsApp Analytics & Assistant")
    st.divider()

    uploaded_file = st.file_uploader("📂 Upload WhatsApp Chat (.txt)", type=["txt"])
    
    # 1-Click Sample Chat Loader
    if st.button("🚀 Load Sample Chat Data", use_container_width=True):
        sample_path = os.path.join(os.path.dirname(__file__), "sample_chat.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.session_state.df = parse_chat(content)
            st.session_state.chat_filename = "sample_chat.txt"
            st.session_state.chat_messages = []
            st.success("Loaded sample chat dataset!")
            st.rerun()

    if uploaded_file is not None and st.session_state.chat_filename != uploaded_file.name:
        bytes_data = uploaded_file.getvalue()
        raw_text = bytes_data.decode("utf-8", errors="ignore")
        st.session_state.df = parse_chat(raw_text)
        st.session_state.chat_filename = uploaded_file.name
        st.session_state.chat_messages = []
        st.success(f"Parsed {uploaded_file.name} successfully!")

    df = st.session_state.df

    selected_user = "Overall"
    if df is not None and not df.empty:
        user_list = [u for u in df['user'].unique() if u != 'group_notification']
        user_list.sort()
        user_list.insert(0, "Overall")
        selected_user = st.selectbox("👤 Select Analysis Scope", user_list)

    st.divider()
    st.markdown("""
    **Export Guide**:
    1. Open WhatsApp chat
    2. Tap **Menu (⋮)** > **More** > **Export Chat**
    3. Select **Without Media**
    4. Upload the exported `.txt` file here
    """)

# Main Content Layout
if df is None or df.empty:
    st.title("🌟 AI-Based WhatsApp Chat Analyzer")
    st.markdown("""
    ### Welcome to the WhatsApp Intelligence & Analytics Platform!
    Gain deep conversational insights, behavioral trends, NLP sentiment metrics, and chat with an integrated AI assistant.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🔍 Key Features:
        - **Comprehensive Metrics**: Messages, words, media attachments, links & participants.
        - **Activity Timelines**: Monthly, daily, day-of-week, and hourly 24h heatmap.
        - **Text & NLP Insights**: Word frequency analysis, stopword removal, and custom WordCloud.
        - **Emoji & Media Analysis**: Most frequent reactions and media sharing distribution.
        - **Sentiment Analysis**: Polarity breakdown (Positive, Neutral, Negative) and user mood scoring.
        - **AI Chatbot Assistant**: Ask questions like *"Who sent the most messages?"* and get instant data-backed answers.
        - **SQLite Database Persistence**: Store sessions and query logs locally.
        """)
    with col2:
        st.info("💡 **Quick Start**: Upload your WhatsApp chat export `.txt` file using the sidebar, or click the **'Load Sample Chat Data'** button to explore all features instantly!")
        
        # Display Sample Chat Preview Card
        sample_path = os.path.join(os.path.dirname(__file__), "sample_chat.txt")
        if os.path.exists(sample_path):
            with st.expander("📄 Preview Sample Chat Dataset"):
                with open(sample_path, "r", encoding="utf-8") as f:
                    st.code(f.read()[:600] + "\n...", language="text")

else:
    # Save session to SQLite if not already saved
    if st.session_state.current_session_id is None:
        try:
            stats = fetch_stats("Overall", df)
            sent = analyze_sentiment("Overall", df)
            _, user_pct = most_busy_users(df)
            sess_id = database.save_analysis_session(st.session_state.chat_filename or "Chat", stats, sent, user_pct)
            st.session_state.current_session_id = sess_id
        except Exception:
            pass

    # Navigation Tabs
    tab_overview, tab_users, tab_activity, tab_content, tab_sentiment, tab_chatbot, tab_db = st.tabs([
        "📊 Overview",
        "👥 User Analysis",
        "📈 Activity & Heatmaps",
        "🔤 Content & Emojis",
        "🧠 Sentiment Analysis",
        "🤖 AI Chatbot",
        "🗄️ Database & History"
    ])

    # ==================== TAB 1: OVERVIEW ====================
    with tab_overview:
        st.subheader(f"📊 Chat Overview - {selected_user}")
        stats = fetch_stats(selected_user, df)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Messages</div>
                <div class="metric-value">{stats['total_messages']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Words</div>
                <div class="metric-value">{stats['total_words']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Media Shared</div>
                <div class="metric-value">{stats['media_count']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Links Shared</div>
                <div class="metric-value">{stats['links_count']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Participants</div>
                <div class="metric-value">{stats['users_count']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>📋 Message Data Explorer</div>", unsafe_allow_html=True)
        
        # Filter dataframe for display
        display_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
        st.dataframe(
            display_df[['date', 'user', 'message', 'word_count', 'is_media', 'url_count', 'emoji_count']],
            use_container_width=True,
            height=300
        )

    # ==================== TAB 2: USER ANALYSIS ====================
    with tab_users:
        st.subheader("👥 User Contribution & Activity Breakdown")
        
        busy_users, user_percent = most_busy_users(df)
        
        if not busy_users.empty:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("#### 🏆 Most Active Users")
                fig, ax = plt.subplots(figsize=(8, 4.5))
                colors = sns.color_palette("viridis", len(busy_users))
                sns.barplot(x=busy_users.index, y=busy_users.values, ax=ax, palette=colors)
                plt.xticks(rotation=45, ha='right', color='white')
                plt.yticks(color='white')
                ax.set_ylabel("Messages Sent", color='white')
                ax.set_facecolor("#0E1117")
                fig.patch.set_facecolor("#0E1117")
                ax.spines['bottom'].set_color('#334155')
                ax.spines['left'].set_color('#334155')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### 📊 Percentage Share")
                st.dataframe(user_percent, use_container_width=True, height=350)
        else:
            st.info("No individual user messages found.")

    # ==================== TAB 3: ACTIVITY & HEATMAPS ====================
    with tab_activity:
        st.subheader(f"📈 Temporal Activity Patterns - {selected_user}")

        # Monthly & Daily Timelines
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📅 Monthly Timeline")
            timeline = monthly_timeline(selected_user, df)
            if not timeline.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(timeline['time'], timeline['message'], color='#38bdf8', marker='o', linewidth=2.5)
                plt.xticks(rotation=45, ha='right', color='white')
                plt.yticks(color='white')
                ax.set_ylabel("Message Volume", color='white')
                ax.set_facecolor("#0E1117")
                fig.patch.set_facecolor("#0E1117")
                ax.grid(True, linestyle='--', alpha=0.2)
                st.pyplot(fig)
            else:
                st.info("Not enough data for monthly timeline.")

        with c2:
            st.markdown("#### 📆 Daily Activity Timeline")
            daily = daily_timeline(selected_user, df)
            if not daily.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(daily['date'], daily['message_count'], color='#a855f7', linewidth=2)
                plt.xticks(rotation=45, ha='right', color='white')
                plt.yticks(color='white')
                ax.set_ylabel("Message Volume", color='white')
                ax.set_facecolor("#0E1117")
                fig.patch.set_facecolor("#0E1117")
                ax.grid(True, linestyle='--', alpha=0.2)
                st.pyplot(fig)
            else:
                st.info("Not enough data for daily timeline.")

        # Busiest Days & Months
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### 🗓️ Most Active Days of Week")
            week_map = week_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=week_map.index, y=week_map.values, ax=ax, palette='mako')
            plt.xticks(rotation=30, color='white')
            plt.yticks(color='white')
            ax.set_facecolor("#0E1117")
            fig.patch.set_facecolor("#0E1117")
            st.pyplot(fig)

        with c4:
            st.markdown("#### 🌙 Most Active Months")
            m_map = month_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=m_map.index, y=m_map.values, ax=ax, palette='rocket')
            plt.xticks(rotation=45, ha='right', color='white')
            plt.yticks(color='white')
            ax.set_facecolor("#0E1117")
            fig.patch.set_facecolor("#0E1117")
            st.pyplot(fig)

        # 24-Hour Activity Heatmap
        st.markdown("#### 🕒 Hourly Activity Heatmap (Day vs Hour)")
        heat_df = activity_heatmap(selected_user, df)
        if not heat_df.empty:
            fig, ax = plt.subplots(figsize=(14, 5))
            sns.heatmap(heat_df, cmap='crest', linewidths=0.5, linecolor='#1e293b', ax=ax, cbar_kws={'label': 'Messages'})
            plt.xticks(rotation=45, ha='right', color='white')
            plt.yticks(color='white')
            ax.set_facecolor("#0E1117")
            fig.patch.set_facecolor("#0E1117")
            st.pyplot(fig)

    # ==================== TAB 4: CONTENT & EMOJIS ====================
    with tab_content:
        st.subheader(f"🔤 Content, Vocabulary & Emojis - {selected_user}")

        # Wordcloud
        st.markdown("#### ☁️ Conversation Word Cloud")
        try:
            wc = create_wordcloud(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            fig.patch.set_facecolor("#0E1117")
            st.pyplot(fig)
        except Exception as e:
            st.warning("Could not generate WordCloud from available text.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔠 Most Frequent Words")
            common_words = most_common_words(selected_user, df, top_n=15)
            if not common_words.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.barplot(x='count', y='word', data=common_words, palette='cool', ax=ax)
                plt.xticks(color='white')
                plt.yticks(color='white')
                ax.set_facecolor("#0E1117")
                fig.patch.set_facecolor("#0E1117")
                st.pyplot(fig)
            else:
                st.info("No common words found.")

        with col2:
            st.markdown("#### 😊 Emoji Usage Breakdown")
            emoji_df = emoji_helper(selected_user, df)
            if not emoji_df.empty:
                st.dataframe(emoji_df.head(15), use_container_width=True, height=350)
            else:
                st.info("No emojis found in this chat selection.")

    # ==================== TAB 5: SENTIMENT ANALYSIS ====================
    with tab_sentiment:
        st.subheader(f"🧠 Sentiment & Tone Analysis - {selected_user}")
        sentiment_res = analyze_sentiment(selected_user, df)

        score = sentiment_res['overall_score']
        if score > 0.05:
            badge_html = f"<span class='badge-positive'>Positive 😊 (Score: {score})</span>"
        elif score < -0.05:
            badge_html = f"<span class='badge-negative'>Negative 😟 (Score: {score})</span>"
        else:
            badge_html = f"<span class='badge-neutral'>Neutral 😐 (Score: {score})</span>"

        st.markdown(f"**Overall Chat Polarity**: {badge_html}", unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### 🥧 Sentiment Distribution")
            labels = ['Positive', 'Neutral', 'Negative']
            sizes = [sentiment_res['pos_percent'], sentiment_res['neu_percent'], sentiment_res['neg_percent']]
            colors = ['#10b981', '#64748b', '#ef4444']

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'color': 'white'})
            ax.axis('equal')
            fig.patch.set_facecolor("#0E1117")
            st.pyplot(fig)

        with c2:
            st.markdown("#### 📈 Sentiment Trend Over Time")
            sent_time = sentiment_res['sentiment_timeline']
            if not sent_time.empty:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(sent_time['date'], sent_time['avg_polarity'], color='#38bdf8', marker='.', linewidth=2)
                ax.axhline(0, color='#94a3b8', linestyle='--', alpha=0.5)
                plt.xticks(rotation=45, ha='right', color='white')
                plt.yticks(color='white')
                ax.set_ylabel("Polarity Score (-1.0 to +1.0)", color='white')
                ax.set_facecolor("#0E1117")
                fig.patch.set_facecolor("#0E1117")
                st.pyplot(fig)

        st.markdown("#### 👥 User-Wise Sentiment Scores")
        if not sentiment_res['user_sentiment'].empty:
            st.dataframe(sentiment_res['user_sentiment'], use_container_width=True)

    # ==================== TAB 6: AI CHATBOT ====================
    with tab_chatbot:
        st.subheader("🤖 AI Chatbot Assistant")
        st.caption("Ask anything about this chat — senders, message counts, emojis, active times, or sentiment.")

        # Quick Question Buttons
        st.markdown("**Quick Inquiries**:")
        q_cols = st.columns(4)
        sample_queries = [
            "Who sent the most messages?",
            "How many messages are in the chat?",
            "Which day had the highest activity?",
            "What is the overall sentiment?"
        ]
        
        clicked_query = None
        for i, q_text in enumerate(sample_queries):
            with q_cols[i]:
                if st.button(q_text, key=f"quick_q_{i}", use_container_width=True):
                    clicked_query = q_text

        # Display Chat History
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Process Input
        user_input = st.chat_input("Type your question about the chat data here...")
        active_query = clicked_query or user_input

        if active_query:
            # User Message
            st.session_state.chat_messages.append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.markdown(active_query)

            # Bot Answer
            with st.chat_message("assistant"):
                ans, intent = answer_question(active_query, df, st.session_state.current_session_id)
                st.markdown(ans)
                st.session_state.chat_messages.append({"role": "assistant", "content": ans})

    # ==================== TAB 7: DATABASE & HISTORY ====================
    with tab_db:
        st.subheader("🗄️ SQLite Database Records & Saved Sessions")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("#### 📁 Saved Analysis Sessions")
        with c2:
            if st.button("🗑️ Clear All Database History"):
                database.clear_all_history()
                st.session_state.current_session_id = None
                st.success("Database records cleared.")
                st.rerun()

        sessions_df = database.get_all_sessions()
        if not sessions_df.empty:
            st.dataframe(sessions_df, use_container_width=True)
        else:
            st.info("No recorded sessions in database yet.")

        st.markdown("#### 💬 Chatbot Interaction Logs")
        chat_logs_df = database.get_chatbot_history(st.session_state.current_session_id)
        if not chat_logs_df.empty:
            st.dataframe(chat_logs_df[['timestamp', 'user_query', 'query_type', 'bot_response']], use_container_width=True)
        else:
            st.info("No chatbot questions logged for this session yet.")
