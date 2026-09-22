import re
import pandas as pd
from datetime import datetime
from urlextract import URLExtract
import emoji

extractor = URLExtract()

def clean_raw_text(data: str) -> str:
    """Cleans invisible unicode control characters commonly found in WhatsApp exports."""
    return data.replace('\u202f', ' ').replace('\xa0', ' ').replace('\u200e', '').replace('\ufeff', '')

def parse_chat(data: str) -> pd.DataFrame:
    """
    Parses exported WhatsApp chat text into a structured pandas DataFrame.
    Supports Android 12h/24h formats, iOS bracketed formats, multi-line messages,
    and system messages (like 'Messages and calls are end-to-end encrypted').
    """
    data = clean_raw_text(data)

    # Patterns for different WhatsApp export timestamp formats
    # 1. Standard Android 12h: dd/mm/yy, hh:mm am/pm -
    # 2. Standard Android 24h: dd/mm/yyyy, hh:mm -
    # 3. iOS bracketed: [dd/mm/yy, hh:mm:ss am/pm] or [dd/mm/yyyy, hh:mm:ss]
    patterns = [
        # Format: 12/05/23, 9:45 pm -  OR 12/05/2023, 09:45 am - OR 12/5/23, 9:45 AM -
        (r'(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[apAP][mM])?)\s*-\s*', '%d/%m/%y', True),
        # Format: [12/05/23, 9:45:10 PM] OR [12/05/2023, 21:45:10]
        (r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[apAP][mM])?)\]\s*', '%d/%m/%y', False),
        # Format: 12.05.23, 21:45 - OR 12.05.2023, 9:45 pm -
        (r'(\d{1,2}\.\d{1,2}\.\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[apAP][mM])?)\s*-\s*', '%d.%m.%y', True),
        # Format: 2023-05-12, 21:45 -
        (r'(\d{4}-\d{1,2}-\d{1,2},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[apAP][mM])?)\s*-\s*', '%Y-%m-%d', True),
    ]

    matched_pattern = None
    messages = []
    dates = []

    for pattern, dt_fmt, is_dash in patterns:
        split_messages = re.split(pattern, data)
        if len(split_messages) > 3:
            matched_pattern = pattern
            # split_messages alternates: [preamble, date1, msg1, date2, msg2, ...]
            # First element is preamble before the first message (or empty)
            raw_dates = split_messages[1::2]
            raw_msgs = split_messages[2::2]
            dates = raw_dates
            messages = raw_msgs
            break

    if not messages:
        # Fallback regex parser for line-by-line parsing
        lines = data.splitlines()
        current_date = None
        current_msg = []
        parsed_dates = []
        parsed_msgs = []
        
        fallback_re = re.compile(r'^\D*(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}.*?)(?: - |\] )')
        for line in lines:
            m = fallback_re.match(line)
            if m:
                if current_date is not None and current_msg:
                    parsed_dates.append(current_date)
                    parsed_msgs.append("\n".join(current_msg))
                current_date = m.group(1).strip('[]')
                body = line[m.end():]
                current_msg = [body]
            else:
                if current_msg:
                    current_msg.append(line)
        if current_date and current_msg:
            parsed_dates.append(current_date)
            parsed_msgs.append("\n".join(current_msg))
        
        dates = parsed_dates
        messages = parsed_msgs

    if not messages:
        return pd.DataFrame()

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Robust DateTime conversion with multiple potential date formats
    def parse_datetime(val_str):
        val_str = str(val_str).strip()
        formats_to_try = [
            '%d/%m/%y, %I:%M %p', '%d/%m/%Y, %I:%M %p',
            '%d/%m/%y, %H:%M', '%d/%m/%Y, %H:%M',
            '%m/%d/%y, %I:%M %p', '%m/%d/%Y, %I:%M %p',
            '%m/%d/%y, %H:%M', '%m/%d/%Y, %H:%M',
            '%d/%m/%y, %I:%M:%S %p', '%d/%m/%Y, %I:%M:%S %p',
            '%d/%m/%y, %H:%M:%S', '%d/%m/%Y, %H:%M:%S',
            '%d.%m.%y, %H:%M', '%d.%m.%Y, %H:%M',
            '%d.%m.%y, %I:%M %p', '%d.%m.%Y, %I:%M %p',
            '%Y-%m-%d, %H:%M', '%Y-%m-%d, %I:%M %p'
        ]
        # Normalize AM/PM formatting
        norm_str = re.sub(r'\s+', ' ', val_str)
        for fmt in formats_to_try:
            try:
                return datetime.strptime(norm_str, fmt)
            except ValueError:
                continue
        # If standard parsing fails, use dateutil/pandas parser
        try:
            return pd.to_datetime(norm_str, errors='coerce')
        except Exception:
            return pd.NaT

    df['date'] = df['message_date'].apply(parse_datetime)
    df = df.dropna(subset=['date']).copy()
    df['date'] = pd.to_datetime(df['date'])

    users = []
    clean_messages = []

    for message in df['user_message']:
        # WhatsApp message format: "User Name: Message text"
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) >= 3:
            users.append(entry[1].strip())
            clean_messages.append(entry[2].strip())
        else:
            users.append('group_notification')
            clean_messages.append(entry[0].strip())

    df['user'] = users
    df['message'] = clean_messages
    df.drop(columns=['user_message', 'message_date'], inplace=True)

    # Extract date & time components
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    # 24-hour period (e.g. 23-00, 00-01) for heatmaps
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour:02d}-00")
        elif hour == 0:
            period.append(f"00-01")
        else:
            period.append(f"{hour:02d}-{(hour + 1):02d}")
    df['period'] = period

    # Media detection
    # Common WhatsApp placeholders for media
    media_patterns = [
        '<media omitted>', '<Media omitted>', 'image omitted', 'video omitted',
        'audio omitted', 'sticker omitted', 'document omitted', 'GIF omitted',
        'Contact card omitted', 'attached:'
    ]
    df['is_media'] = df['message'].apply(lambda msg: any(p.lower() in msg.lower() for p in media_patterns))

    # URL extraction
    df['urls'] = df['message'].apply(lambda msg: extractor.find_urls(msg))
    df['url_count'] = df['urls'].apply(len)

    # Word count
    df['word_count'] = df['message'].apply(lambda msg: len(msg.split()))

    # Character count
    df['char_count'] = df['message'].apply(len)

    # Emoji extraction
    def extract_emojis(msg):
        return [c for c in msg if emoji.is_emoji(c)]
    
    df['emojis'] = df['message'].apply(extract_emojis)
    df['emoji_count'] = df['emojis'].apply(len)

    return df
