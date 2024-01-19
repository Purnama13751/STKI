import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the cleaned data
df = pd.read_csv('cleaned.csv')

# Function to generate word cloud
def generate_word_cloud(text, title):
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    wordcloud = wc.generate(text)

    # Display the word cloud using Streamlit
    st.subheader(title + " Word Cloud")
    st.image(wordcloud.to_array())


def main():
    st.title("Sentiment Analysis Word Clouds and Comment Distribution")

    # Display the distribution of comments for each bank
    st.sidebar.header("Jumlah Komentar dari Bank")
    selected_bank = st.sidebar.selectbox("Select a Bank", df['nama_bank'].unique())
    bank_distribution = df[df['nama_bank'] == selected_bank]['sentimen'].value_counts()
    st.sidebar.bar_chart(bank_distribution)

    # Display Positive, Neutral, and Negative Word Clouds
    st.header("Word Clouds and Sentiment Analysis")

    # Positive Word Cloud
    positif_text = df[(df['sentimen'] == 'positif') & (df['nama_bank'] == selected_bank)]['text_preprocessed'].str.cat(sep=', ')
    generate_word_cloud(positif_text, "Positive")

    # Neutral Word Cloud
    netral_text = df[(df['sentimen'] == 'netral') & (df['nama_bank'] == selected_bank)]['text_preprocessed'].str.cat(sep=', ')
    generate_word_cloud(netral_text, "Neutral")

    # Negative Word Cloud
    negatif_text = df[(df['sentimen'] == 'negatif') & (df['nama_bank'] == selected_bank)]['text_preprocessed'].str.cat(sep=', ')
    generate_word_cloud(negatif_text, "Negative")

    # Search functionality
    st.sidebar.header("Search Comments")
    search_term = st.sidebar.text_input("Enter search term:")
    search_results = df[(df['text_preprocessed'].str.contains(search_term, case=False)) & (df['nama_bank'] == selected_bank)]

    if not search_results.empty:
        st.sidebar.subheader("Search Results:")
        st.sidebar.table(search_results[['text_preprocessed', 'sentimen']])
    else:
        st.sidebar.info("No matching comments found.")

if __name__ == '__main__':
    main()
