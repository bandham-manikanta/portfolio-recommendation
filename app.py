# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf  # For fetching stock prices
import os
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import re
import time
import urllib.parse
import logging

from transformers import pipeline, AutoTokenizer
import feedparser
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Set page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Define prediction horizon and sequence length
prediction_horizon = 5  # Number of days ahead to predict
sequence_length = 60    # Sequence length expected by the Galformer model

# Fetch historical data function
def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.warning(f"No historical data available for {ticker}.")
    return data

from sklearn.preprocessing import MinMaxScaler

# Prepare input sequence function
def prepare_input_sequence(data, sequence_length, ticker):
    if 'Close' not in data.columns or len(data) < sequence_length:
        st.warning(f"Not enough data to create input sequence for {ticker}.")
        return None, None
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)
    input_sequence = scaled_prices[-sequence_length:]
    input_sequence = input_sequence.reshape(1, sequence_length, -1)
    return input_sequence, scaler

# Galformer predictions function
def galformer_predictions(ticker, sequence_length, model):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    data = fetch_historical_data(ticker, start_date, end_date)
    input_sequence, scaler = prepare_input_sequence(data, sequence_length, ticker)
    if input_sequence is not None:
        try:
            predictions = model.predict(input_sequence)
            predictions = scaler.inverse_transform(predictions).flatten()
            return predictions[:prediction_horizon]  # Return predictions for the prediction horizon
        except Exception as e:
            st.error(f"Error making predictions for {ticker}: {e}")
            logging.error(f"Error making predictions for {ticker}: {e}")
            return None
    else:
        return None

# Summarize articles function
def summarize_articles(articles, summarizer, summarizer_tokenizer):
    full_text = ' '.join(
        f"{article.get('title', '')}. {article.get('summary', '')}."
        for article in articles
    ).strip()

    if not full_text:
        return 'No summary available.'

    sentences = sent_tokenize(full_text)

    max_input_length = summarizer_tokenizer.model_max_length

    # Split sentences into chunks whose tokenized length is within max_input_length
    chunks = []
    current_chunk = ''
    current_length = 0

    for sentence in sentences:
        sentence_length = len(summarizer_tokenizer.encode(sentence, add_special_tokens=False))
        if current_length + sentence_length <= max_input_length:
            current_chunk += ' ' + sentence
            current_length += sentence_length
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    summaries = []

    for chunk in chunks:
        try:
            summary = summarizer(
                chunk,
                max_length=200,
                min_length=100,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            st.error(f"Error summarizing articles: {e}")
            logging.error(f"Error summarizing articles for chunk: {e}")
            continue

    combined_summary = ' '.join(summaries)
    return combined_summary if combined_summary else 'No summary available.'

# Analyze sentiment summary function
def analyze_sentiment_summary(summary_text, sentiment_classifier):
    if summary_text:
        text = summary_text[:512]  # Truncate if necessary
        try:
            result = sentiment_classifier(text)[0]
            # Convert sentiment label to score
            label = result['label'].lower()
            if label == 'positive':
                score = result['score']
            elif label == 'negative':
                score = -result['score']
            else:
                score = 0  # Neutral sentiment
            return score
        except Exception as e:
            st.error(f"Error analyzing sentiment of summary: {e}")
            logging.error(f"Error analyzing sentiment of summary: {e}")
            return 0
    else:
        return 0

# Analyze article sentiments function
def analyze_article_sentiments(articles, sentiment_classifier):
    sentiment_data = []
    for article in articles:
        text = f"{article.get('title', '')}. {article.get('summary', '')}."
        if not text.strip():
            continue
        date_str = article.get('published_date', '')
        if not date_str:
            continue
        # Convert date string to date object
        try:
            date = pd.to_datetime(date_str).date()
        except Exception as e:
            logging.error(f"Error parsing date for article: {e}")
            continue
        try:
            result = sentiment_classifier(text[:512])[0]
            label = result['label'].lower()
            if label == 'positive':
                score = result['score']
            elif label == 'negative':
                score = -result['score']
            else:
                score = 0  # Neutral sentiment
            sentiment_data.append({'Date': date, 'Sentiment Score': score})
        except Exception as e:
            logging.error(f"Error analyzing sentiment for an article: {e}")
            continue
    sentiment_df = pd.DataFrame(sentiment_data)
    if sentiment_df.empty:
        st.warning("No sentiment data available to analyze.")
        return pd.DataFrame()
    daily_sentiment = sentiment_df.groupby('Date').mean().reset_index()
    return daily_sentiment

# Generate reasoning function
def generate_reasoning(ticker, current_price, predicted_prices, sentiment_score, text_generator):
    prompt = (
        f"As a seasoned financial analyst, provide a detailed analysis for {ticker} based on the following data:\n"
        f"- Current Price: ${current_price:.2f}\n"
        f"- Predicted Prices for the next {prediction_horizon} days: {predicted_prices.tolist()}\n"
        f"- Sentiment Score based on recent news: {sentiment_score:.2f}\n\n"
        f"Please include market trends, potential risks, and an investment recommendation. Explain your reasoning thoroughly."
    )
    try:
        generated = text_generator(
            prompt,
            max_length=400,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )[0]['generated_text']
        reasoning = generated[len(prompt):].strip()
        # Ensure the reasoning ends gracefully
        if reasoning and reasoning[-1] not in ['.', '!', '?']:
            reasoning += '.'
        return reasoning
    except Exception as e:
        st.error(f"Error generating reasoning for {ticker}: {e}")
        logging.error(f"Error generating reasoning for {ticker}: {e}")
        return "Unable to generate reasoning at this time."

# Fetch Google News RSS function
def fetch_google_news_rss(ticker, max_articles):
    query = f"{ticker} stock"
    encoded_query = urllib.parse.quote_plus(query)
    feed_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(feed_url)
        if not feed.entries:
            st.warning(f"No articles found for {ticker}.")
            return []
        articles = []
        for entry in feed.entries[:max_articles]:
            title = entry.title
            # Clean HTML tags from the summary
            summary = re.sub('<[^<]+?>', '', entry.summary)
            published_at = entry.published
            articles.append({
                'title': title,
                'summary': summary,
                'published_date': published_at
            })
        return articles
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {e}")
        logging.error(f"Error fetching news for {ticker}: {e}")
        return []

# Plot predictions function
def plot_predictions(ticker, current_price, lstm_pred, galformer_pred):
    # Fetch historical data for the past 90 days
    end_date = datetime.today()
    start_date = end_date - timedelta(days=90)
    historical_data = fetch_historical_data(ticker, start_date, end_date)
    if historical_data.empty:
        st.warning(f"No historical data available for {ticker}.")
        return
    historical_prices = historical_data['Close'].reset_index()
    historical_prices.rename(columns={'Date': 'Date', 'Close': 'Price'}, inplace=True)
    historical_prices['Type'] = 'Historical'

    # Ensure 'Date' is datetime.datetime
    historical_prices['Date'] = pd.to_datetime(historical_prices['Date'])

    # Create future dates
    last_date = historical_prices['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_horizon + 1)]

    # Prepare DataFrames for predictions
    df_lstm = pd.DataFrame({
        'Date': future_dates,
        'Price': lstm_pred,
        'Type': 'LSTM Prediction'
    })
    df_galformer = pd.DataFrame({
        'Date': future_dates,
        'Price': galformer_pred,
        'Type': 'Galformer Prediction'
    })

    # Combine all DataFrames
    df_all = pd.concat([historical_prices, df_lstm, df_galformer], ignore_index=True)
    df_all.sort_values('Date', inplace=True)

    # Plot using Plotly
    fig = px.line(df_all, x='Date', y='Price', color='Type',
                  title=f"{ticker} Historical and Predicted Prices")

    # Add vertical line at the prediction start
    prediction_start = future_dates[0] - timedelta(days=0.5)
    fig.add_vrect(
        x0=prediction_start,
        x1=prediction_start,
        fillcolor="gray",
        opacity=0.5,
        line_width=0
    )

    # Add annotation
    fig.add_annotation(
        x=prediction_start,
        y=df_all['Price'].max(),
        text="Prediction Start",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

    # Adjust layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title_text='Data Type'
    )

    st.plotly_chart(fig, use_container_width=True, key=f"predictions-{ticker}")

# Plot sentiment function
def plot_sentiment(sentiment_score, key):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# Display results in tabs function
def display_results_in_tabs(company):
    st.subheader(f"{company['Ticker']} - ${company['Current Price']:.2f}")
    st.metric(label="Sentiment Score", value=round(company['Sentiment Score'], 2))

    tabs = st.tabs(["Predictions", "Sentiment & News", "Reasoning", "Sentiment Over Time"])

    with tabs[0]:
        st.markdown("#### Predictions")
        if company['Galformer Prediction'] is not None:
            plot_predictions(
                company['Ticker'],
                company['Current Price'],
                company['LSTM Prediction'],
                company['Galformer Prediction'],
            )
        else:
            st.write("Predictions are not available.")

    with tabs[1]:
        st.markdown("#### Sentiment Score")
        plot_sentiment(company['Sentiment Score'], key=f"sentiment-{company['Ticker']}")
        st.markdown("#### News Summary")
        with st.expander("Show News Summary"):
            st.write(company['Summary'])

    with tabs[2]:
        st.markdown("#### Generated Reasoning")
        st.write(company['Reasoning'])

    with tabs[3]:
        st.markdown("#### Sentiment Over Time")
        daily_sentiment = company.get('Daily Sentiment')
        if daily_sentiment is not None and not daily_sentiment.empty:
            # Plot sentiment over time
            fig = px.line(daily_sentiment, x='Date', y='Sentiment Score',
                          title='Sentiment Over Time', markers=True)
            st.plotly_chart(fig, use_container_width=True, key=f"sentiment-over-time-{company['Ticker']}")

            # Fetch historical stock prices matching sentiment dates
            start_date = daily_sentiment['Date'].min()
            end_date = daily_sentiment['Date'].max() + timedelta(days=1)  # Include the last day
            historical_data = fetch_historical_data(company['Ticker'], start_date, end_date)
            if historical_data.empty:
                st.warning(f"No historical data available for {company['Ticker']}.")
                return
            historical_prices = historical_data['Close'].reset_index()
            historical_prices['Date'] = pd.to_datetime(historical_prices['Date']).dt.date
            historical_prices.rename(columns={'Close': 'Price'}, inplace=True)

            # Merge sentiment and historical prices
            merged_df = pd.merge(
                daily_sentiment,
                historical_prices,
                on='Date',
                how='inner'
            )

            if merged_df.empty:
                st.write("No overlapping dates between sentiment data and stock prices.")
                return

            # Create dual-axis plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add sentiment trace
            fig.add_trace(
                go.Scatter(
                    x=merged_df['Date'],
                    y=merged_df['Sentiment Score'],
                    name="Sentiment Score",
                    mode='lines+markers'
                ),
                secondary_y=False,
            )

            # Add price trace using 'Price' column
            if 'Price' in merged_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=merged_df['Date'],
                        y=merged_df['Price'],
                        name="Stock Price",
                        mode='lines+markers'
                    ),
                    secondary_y=True,
                )

            # Update layout
            fig.update_layout(
                title_text="Sentiment Score and Stock Price Over Time"
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
            fig.update_yaxes(title_text="Stock Price", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True, key=f"stock-price-{company['Ticker']}")
        else:
            st.write("No sentiment data available for this company.")

@st.cache_resource
def load_models():
    sentiment_model_name = "ProsusAI/finbert"
    summarizer_model_name = 'facebook/bart-large-cnn'
    generator_model_name = 'gpt2-medium'

    sentiment_classifier = pipeline('sentiment-analysis', model=sentiment_model_name)
    summarizer = pipeline('summarization', model=summarizer_model_name)
    summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    text_generator = pipeline('text-generation', model=generator_model_name, pad_token_id=50256)

    return sentiment_classifier, summarizer, summarizer_tokenizer, text_generator

@st.cache_resource
def load_galformer_model():
    model_path = 'generalized_stock_galformer_model.keras'
    if not os.path.exists(model_path):
        st.error(
            f"Galformer model file not found at '{model_path}'. Please ensure the model file is available."
        )
        logging.error(f"Galformer model file not found at '{model_path}'.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    except Exception as e:
        st.error(f"Error loading Galformer model: {e}")
        logging.error(f"Error loading Galformer model: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_current_prices(tickers):
    data = yf.download(tickers, period="1d", interval="1d", threads=True, progress=False)
    current_prices = {}
    for ticker in tickers:
        try:
            # Get the most recent closing price
            if isinstance(data['Close'], pd.Series):
                current_price = data['Close'].dropna()[-1]
            else:
                current_price = data['Close'][ticker].dropna()[-1]
            current_prices[ticker] = current_price
        except Exception as e:
            st.error(f"Error fetching price for {ticker}: {e}")
            logging.error(f"Error fetching price for {ticker}: {e}")
            current_prices[ticker] = np.nan  # Use NaN for missing data
    return current_prices

# Application Layout
st.title("Stock Analysis Dashboard")

# Sidebar for user inputs
st.sidebar.header("Configure Analysis")

# Ticker selection
all_tickers = [
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "WMT",
    "JPM",
    "V",
    "XOM",
    "UNH",
    "ORCL",
    "MA",
    "HD",
    "PG",
    "COST",
    "JNJ",
    "NFLX",
    "ABBV",
    "BAC",
    "KO",
    "CRM",
    "CVX",
    "MRK",
    "AMD",
    "PEP",
    "ACN",
    "LIN",
    "MCD",
    "CSCO",
    "ADBE",
    "WFC",
    "IBM",
    "GE",
    "ABT",
    "DHR",
    "AXP",
    "MS",
    "CAT",
    "NOW",
    "QCOM",
    "PM",
    "ISRG",
    "VZ",
]

selected_tickers = st.sidebar.multiselect(
    "Select Tickers", options=all_tickers, default=["AAPL", "GOOGL", "MSFT", "VZ"]
)

# Number of articles
max_articles = st.sidebar.slider(
    "Maximum Articles per Company", min_value=5, max_value=20, value=10, step=1
)

# Process button
if st.sidebar.button("Run Analysis"):
    with st.spinner("Loading models..."):
        sentiment_classifier, summarizer, summarizer_tokenizer, text_generator = load_models()
        galformer_model = load_galformer_model()
        if galformer_model is None:
            st.stop()

    with st.spinner("Fetching current stock prices..."):
        current_prices = fetch_current_prices(selected_tickers)

    combined_data = []
    total_tickers = len(selected_tickers)
    progress_bar = st.progress(0)

    for i, ticker in enumerate(selected_tickers):
        logging.info(f"Processing ticker {ticker} ({i+1}/{total_tickers})")
        current_price = current_prices.get(ticker, np.nan)
        if np.isnan(current_price):
            st.warning(f"Skipping {ticker} due to missing current price.")
            logging.warning(f"Skipping {ticker} due to missing current price.")
            continue

        # Get Galformer predictions
        with st.spinner(f"Generating Galformer predictions for {ticker}..."):
            galformer_pred = galformer_predictions(ticker, sequence_length, galformer_model)
            if galformer_pred is None:
                st.warning(f"Skipping {ticker} due to insufficient data for Galformer predictions.")
                logging.warning(f"Skipping {ticker} due to insufficient data for Galformer predictions.")
                continue
            galformer_pred = np.round(galformer_pred, 2)

        # Mock LSTM predictions (replace with actual LSTM predictions if available)
        lstm_pred = np.round(
            np.random.normal(
                loc=current_price, scale=current_price * 0.02, size=prediction_horizon
            ),
            2,
        )

        # Fetch and summarize news articles
        with st.spinner(f"Fetching and summarizing news for {ticker}..."):
            articles = fetch_google_news_rss(ticker, max_articles=max_articles)
            if not articles:
                st.warning(f"No news articles found for {ticker}.")
                logging.warning(f"No news articles found for {ticker}.")
                continue
            summary = summarize_articles(articles, summarizer, summarizer_tokenizer)

        # Perform sentiment analysis on summary
        with st.spinner(f"Analyzing sentiment for {ticker}..."):
            sentiment = analyze_sentiment_summary(summary, sentiment_classifier)

        # Analyze sentiment over time
        with st.spinner(f"Analyzing sentiment over time for {ticker}..."):
            daily_sentiment = analyze_article_sentiments(articles, sentiment_classifier)

        # Generate reasoning
        with st.spinner(f"Generating reasoning for {ticker}..."):
            reasoning = generate_reasoning(
                ticker, current_price, galformer_pred, sentiment, text_generator
            )

        # Assemble company information
        company_info = {
            'Ticker': ticker,
            'Current Price': current_price,
            'LSTM Prediction': lstm_pred.tolist(),
            'Galformer Prediction': galformer_pred.tolist(),
            'Sentiment Score': sentiment,
            'Summary': summary,
            'Reasoning': reasoning,
            'Daily Sentiment': daily_sentiment,
        }

        combined_data.append(company_info)

        st.success(f"Processed {ticker}")
        logging.info(f"Processed {ticker}")
        progress_bar.progress((i + 1) / total_tickers)

        # Add a delay to be polite to servers
        time.sleep(1)

    # Display results
    st.header("Analysis Results")

    for company in combined_data:
        display_results_in_tabs(company)

    # Option to download results
    st.header("Download Results")

    prediction_records = []

    for company in combined_data:
        for i in range(prediction_horizon):
            record = {
                'Ticker': company['Ticker'],
                'Current Price': company['Current Price'],
                'Day Ahead': i + 1,
                'LSTM Prediction': company['LSTM Prediction'][i],
                'Galformer Prediction': company['Galformer Prediction'][i],
                'Sentiment Score': company['Sentiment Score'],
                'Reasoning': company['Reasoning'],
                'Summary': company['Summary'],
            }
            prediction_records.append(record)

    predictions_df = pd.DataFrame(prediction_records)

    # Convert DataFrame to CSV
    csv = predictions_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name='combined_predictions.csv',
        mime='text/csv',
    )

else:
    st.info("Please configure analysis parameters and click 'Run Analysis'.")
