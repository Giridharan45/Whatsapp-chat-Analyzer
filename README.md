# 💬 AI-Based WhatsApp Chat Analyzer with Integrated Chatbot Assistant

An end-to-end Python & Streamlit web application that accepts exported WhatsApp `.txt` chats, parses and cleans the data, calculates comprehensive statistics, creates visual insights, performs NLP sentiment analysis, persists session records in SQLite, and provides an integrated AI chatbot assistant to answer conversational questions.

---

## 🌟 Key Features

1. **Flexible WhatsApp Ingestion**:
   - Supports Android 12-hour (`dd/mm/yy, hh:mm am/pm`), Android 24-hour (`dd/mm/yyyy, hh:mm`), and iOS bracketed (`[dd/mm/yy, hh:mm:ss]`) formats.
   - Cleans hidden zero-width unicode characters and normalizes multi-line messages.
   - 1-Click **"Load Sample Chat Data"** button for immediate testing.

2. **Core Statistics & KPIs**:
   - Total messages, total words, media count, links shared, and active participant count.
   - Interactive message data explorer table with search and filtering.

3. **User & Activity Breakdown**:
   - Most active participants bar chart and percentage contribution.
   - Individual user drill-down analysis.
   - Monthly and daily message volume timelines.
   - Most active days of the week & most active months.
   - **24-Hour Activity Heatmap** (Day of week vs Hourly period).

4. **Content, WordCloud & Emoji Analytics**:
   - Stopword-filtered word frequency analysis.
   - Beautiful, high-resolution WordCloud generation.
   - Emoji frequency catalog and distribution table.

5. **NLP Sentiment Analysis**:
   - TextBlob polarity scoring (-1.0 to +1.0).
   - Positive, Neutral, and Negative distribution pie chart.
   - Sentiment progression timeline.
   - User-wise sentiment comparison.

6. **Integrated AI Chatbot Assistant**:
   - Ask natural language questions like:
     - *"Who sent the most messages?"*
     - *"How many messages are in the chat?"*
     - *"Which day had the highest activity?"*
     - *"What is the most used emoji?"*
     - *"What is the overall sentiment?"*
     - *"Tell me about Alice"*
   - Quick inquiry shortcut buttons.

7. **SQLite Database Persistence**:
   - Local database (`whatsapp_analytics.db`) tracking sessions, user statistics, and chatbot query logs.
   - Built-in database viewer tab with clear history capability.

---

## 🛠️ Technology Stack

- **Frontend / Web UI**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn, WordCloud
- **NLP & Text Processing**: TextBlob, NLTK, Emoji, URLExtract
- **Database**: SQLite3
- **Language**: Python 3.10+

---

## 🚀 Quick Start Guide

### 1. Clone or Open the Repository
```bash
git clone <repository_url>
cd omniroute
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 5. Run Automated Unit Tests
```bash
python test_suite.py
```

---

## ☁️ Deployment Instructions

### Deploy to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the Main file path to `app.py`.
5. Click **"Deploy!"**.

---

## 📁 Project Structure

```
├── app.py                      # Main Streamlit web application
├── preprocessor.py             # WhatsApp chat parsing and data cleaning engine
├── analyzer.py                 # Statistical analysis, wordcloud, timelines, sentiment
├── database.py                 # SQLite database models, operations, and logging
├── chatbot.py                  # NLP rule-based Chatbot Assistant engine
├── sample_chat.txt             # Sample WhatsApp conversation dataset
├── test_suite.py               # Unit testing suite
├── requirements.txt            # Python package dependencies
├── docs/
│   ├── MCA_PROJECT_REPORT.md   # Comprehensive MCA mini-project documentation
│   ├── VIVA_PREPARATION_NOTES.md # Viva Voce technical Q&A preparation
│   └── PRESENTATION_OUTLINE.md # PowerPoint presentation outline
└── README.md                   # Project documentation & quick start guide
```
