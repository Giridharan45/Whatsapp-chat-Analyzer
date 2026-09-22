import re
import pandas as pd
from collections import Counter
import emoji
from analyzer import (
    fetch_stats, most_busy_users, most_common_words, emoji_helper,
    week_activity_map, month_activity_map, analyze_sentiment
)
from database import log_chatbot_query

def identify_intent(query: str) -> str:
    """Identifies the intent of the user's natural language question."""
    q = query.lower().strip()

    if any(k in q for k in ['who sent the most', 'top sender', 'most active user', 'who talks the most', 'busiest user', 'top user']):
        return 'top_sender'
    elif any(k in q for k in ['how many message', 'total message', 'number of message', 'message count', 'count of message']):
        return 'total_messages'
    elif any(k in q for k in ['how many word', 'total word', 'word count']):
        return 'total_words'
    elif any(k in q for k in ['media', 'photos', 'videos', 'images', 'stickers', 'attachments']):
        return 'media_count'
    elif any(k in q for k in ['link', 'url', 'website', 'links shared']):
        return 'links_count'
    elif any(k in q for k in ['most active month', 'peak month', 'which month', 'busiest month']):
        return 'busiest_month'
    elif any(k in q for k in ['most active day', 'peak day', 'which day', 'busiest day', 'day of the week']):
        return 'busiest_day'
    elif any(k in q for k in ['peak hour', 'busiest hour', 'what time', 'active time', 'which hour']):
        return 'busiest_hour'
    elif any(k in q for k in ['emoji', 'emojis', 'smileys', 'reactions']):
        return 'top_emoji'
    elif any(k in q for k in ['common word', 'frequent word', 'most used word', 'top word', 'popular word']):
        return 'top_word'
    elif any(k in q for k in ['sentiment', 'mood', 'positive or negative', 'tone', 'feeling', 'vibe']):
        return 'sentiment'
    elif any(k in q for k in ['summary', 'summarize', 'overview', 'highlights', 'tell me about']):
        return 'summary'
    elif any(k in q for k in ['who are the users', 'list participants', 'members', 'people in chat']):
        return 'user_list'
    else:
        # Check if question is asking about a specific person
        return 'user_specific_or_general'

def answer_question(query: str, df: pd.DataFrame, session_id: int = None) -> tuple[str, str]:
    """
    Answers natural language questions about the analyzed chat dataframe.
    Returns (response_text, intent_name).
    """
    if df is None or df.empty:
        return "⚠️ Please upload or select a WhatsApp chat file first before asking questions.", "No Data"

    intent = identify_intent(query)
    q_lower = query.lower()

    # Pre-calculated common metrics
    clean_df = df[df['user'] != 'group_notification']
    stats = fetch_stats('Overall', df)

    if intent == 'top_sender':
        busy_users, user_pct = most_busy_users(df)
        if not busy_users.empty:
            top_user = busy_users.index[0]
            top_count = busy_users.iloc[0]
            pct = user_pct.iloc[0]['percent']
            response = (
                f"👑 **Top Sender**: **{top_user}**\n\n"
                f"- **Messages Sent**: {top_count:,}\n"
                f"- **Group Share**: {pct}% of all messages\n\n"
                f"**Top 3 Contributors**:\n"
            )
            for i, (user, count) in enumerate(busy_users.head(3).items(), 1):
                p = user_pct[user_pct['user'] == user]['percent'].values[0]
                response += f"{i}. **{user}**: {count:,} messages ({p}%)\n"
        else:
            response = "No user messages found."

    elif intent == 'total_messages':
        response = (
            f"📊 **Total Messages**: **{stats['total_messages']:,}** messages in this chat.\n"
            f"- Text Messages: {stats['total_messages'] - stats['media_count']:,}\n"
            f"- Media/Attachments: {stats['media_count']:,}\n"
            f"- Total Words: {stats['total_words']:,}"
        )

    elif intent == 'total_words':
        avg_words_per_msg = round(stats['total_words'] / max(1, stats['total_messages']), 1)
        response = (
            f"📝 **Total Words**: **{stats['total_words']:,}** words spoken.\n"
            f"- Average words per message: ~**{avg_words_per_msg}** words"
        )

    elif intent == 'media_count':
        pct = round((stats['media_count'] / max(1, stats['total_messages'])) * 100, 1)
        response = (
            f"📷 **Media Messages**: **{stats['media_count']:,}** media items were shared "
            f"(photos, videos, audio, documents, stickers).\n"
            f"- Media represents **{pct}%** of the total chat."
        )

    elif intent == 'links_count':
        response = (
            f"🔗 **Links Shared**: **{stats['links_count']:,}** URLs / web links were shared across the group."
        )

    elif intent == 'busiest_month':
        month_map = month_activity_map('Overall', df)
        top_month = month_map.idxmax()
        top_count = month_map.max()
        response = (
            f"📅 **Most Active Month**: **{top_month}** with **{top_count:,}** messages sent."
        )

    elif intent == 'busiest_day':
        week_map = week_activity_map('Overall', df)
        top_day = week_map.idxmax()
        top_count = week_map.max()
        response = (
            f"🗓️ **Most Active Day of Week**: **{top_day}** with **{top_count:,}** messages sent."
        )

    elif intent == 'busiest_hour':
        hour_counts = clean_df['hour'].value_counts()
        if not hour_counts.empty:
            top_hour = hour_counts.index[0]
            count = hour_counts.iloc[0]
            period_str = f"{top_hour:02d}:00 - {(top_hour + 1) % 24:02d}:00"
            response = (
                f"⏰ **Peak Activity Hour**: **{period_str}** with **{count:,}** messages sent."
            )
        else:
            response = "Could not calculate hourly activity."

    elif intent == 'top_emoji':
        emoji_df = emoji_helper('Overall', df)
        if not emoji_df.empty:
            top_e = emoji_df.iloc[0]['emoji']
            top_c = emoji_df.iloc[0]['count']
            total_e = emoji_df['count'].sum()
            response = (
                f"✨ **Most Used Emoji**: {top_e} (used **{top_c:,}** times)\n\n"
                f"**Top 5 Emojis**:\n"
            )
            for idx, row in emoji_df.head(5).iterrows():
                response += f"- {row['emoji']} : {row['count']} times\n"
            response += f"\nTotal emojis sent across chat: **{total_e:,}**"
        else:
            response = "No emojis were detected in this chat."

    elif intent == 'top_word':
        words_df = most_common_words('Overall', df, top_n=10)
        if not words_df.empty:
            top_w = words_df.iloc[0]['word']
            top_wc = words_df.iloc[0]['count']
            response = (
                f"🔤 **Most Common Word**: '**{top_w}**' (used **{top_wc:,}** times)\n\n"
                f"**Top 5 Words** (excluding common stopwords):\n"
            )
            for idx, row in words_df.head(5).iterrows():
                response += f"{idx+1}. **{row['word']}**: {row['count']} times\n"
        else:
            response = "Not enough text data to determine common words."

    elif intent == 'sentiment':
        sentiment_res = analyze_sentiment('Overall', df)
        score = sentiment_res['overall_score']
        tone = "Positive 😊" if score > 0.05 else ("Negative 😟" if score < -0.05 else "Neutral 😐")
        response = (
            f"🧠 **Overall Sentiment Analysis**:\n"
            f"- **General Tone**: **{tone}** (Polarity Score: {score})\n"
            f"- 🟢 **Positive**: {sentiment_res['pos_percent']}%\n"
            f"- ⚪ **Neutral**: {sentiment_res['neu_percent']}%\n"
            f"- 🔴 **Negative**: {sentiment_res['neg_percent']}%\n"
        )

    elif intent == 'user_list':
        unique_users = [u for u in df['user'].unique() if u != 'group_notification']
        response = (
            f"👥 **Chat Participants ({len(unique_users)})**:\n" +
            ", ".join(f"**{u}**" for u in unique_users)
        )

    elif intent == 'summary':
        sentiment_res = analyze_sentiment('Overall', df)
        busy_users, _ = most_busy_users(df)
        week_map = week_activity_map('Overall', df)
        top_user = busy_users.index[0] if not busy_users.empty else "N/A"
        response = (
            f"📋 **WhatsApp Chat Executive Summary**:\n\n"
            f"- 💬 **Total Messages**: {stats['total_messages']:,}\n"
            f"- 👥 **Total Participants**: {stats['users_count']}\n"
            f"- 👑 **Most Active User**: **{top_user}**\n"
            f"- 📷 **Media Attachments**: {stats['media_count']:,}\n"
            f"- 🔗 **Links Shared**: {stats['links_count']:,}\n"
            f"- 🗓️ **Peak Day**: {week_map.idxmax()}\n"
            f"- 🌟 **Overall Sentiment**: {sentiment_res['pos_percent']}% Positive, {sentiment_res['neu_percent']}% Neutral, {sentiment_res['neg_percent']}% Negative\n"
        )

    else:
        # Check if user mentioned a specific participant name
        all_users = [u for u in df['user'].unique() if u != 'group_notification']
        matched_user = None
        for u in all_users:
            if u.lower() in q_lower:
                matched_user = u
                break

        if matched_user:
            u_stats = fetch_stats(matched_user, df)
            u_sent = analyze_sentiment(matched_user, df)
            u_emojis = emoji_helper(matched_user, df)
            top_emoji_str = u_emojis.iloc[0]['emoji'] if not u_emojis.empty else "None"
            response = (
                f"👤 **Individual Analysis for {matched_user}**:\n\n"
                f"- **Messages Sent**: {u_stats['total_messages']:,}\n"
                f"- **Words Spoken**: {u_stats['total_words']:,}\n"
                f"- **Media Shared**: {u_stats['media_count']:,}\n"
                f"- **Links Shared**: {u_stats['links_count']:,}\n"
                f"- **Top Emoji**: {top_emoji_str}\n"
                f"- **Sentiment Polarity**: {u_sent['overall_score']} "
                f"({u_sent['pos_percent']}% Pos / {u_sent['neu_percent']}% Neu / {u_sent['neg_percent']}% Neg)"
            )
            intent = f"user_stats_{matched_user}"
        else:
            response = (
                f"🤖 I can answer questions like:\n"
                f"- *'Who sent the most messages?'*\n"
                f"- *'How many messages are in the chat?'*\n"
                f"- *'Which month was most active?'*\n"
                f"- *'Which day had the highest activity?'*\n"
                f"- *'What is the most used emoji?'*\n"
                f"- *'What is the most common word?'*\n"
                f"- *'How many media messages are there?'*\n"
                f"- *'What is the overall sentiment?'*\n"
                f"- *'Tell me about [User Name]'*\n"
                f"- *'Give me a summary of the chat'*."
            )

    if session_id:
        try:
            log_chatbot_query(session_id, query, response, intent)
        except Exception:
            pass

    return response, intent
