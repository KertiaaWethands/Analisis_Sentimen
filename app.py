import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import re
import os
import csv
import operator
import shutil
import zipfile
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
app.static_folder = 'static'

# Analisis teks dan pembuatan word cloud
def analyze_text(limit=10):
    # Membaca dataset
    df = pd.read_csv('dataset_label_2.csv')

    # Preprocessing data
    df['Teks'] = df['Teks'].fillna('')  # Replace missing values with empty string
    df['Teks'] = df['Teks'].str.lower()

    # Membuat objek TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Mengubah teks menjadi vektor TF-IDF
    tfidf_vectors = vectorizer.fit_transform(df['Teks'])

    # Mendapatkan daftar kata (wordlist)
    feature_names = vectorizer.get_feature_names_out()

    # Menghitung total frekuensi setiap kata
    word_freq = tfidf_vectors.sum(axis=0)

    # Membuat dictionary dengan kata dan frekuensinya
    word_freq_dict = {word: freq for word, freq in zip(feature_names, word_freq.flat) if word not in ['resesi', 'ancam', 'hadap', 'indonesia', 'ekonomi', 'global']}

    # Mengurutkan kata berdasarkan frekuensinya (dalam urutan menurun)
    sorted_word_freq = sorted(word_freq_dict.items(), key=operator.itemgetter(1), reverse=True)

    # Mengambil kata-kata teratas sesuai limit
    top_words = [word for word, freq in sorted_word_freq[:limit]]

    # Menghitung kemunculan setiap kata dalam setiap label
    labels = df['Label'].unique()

    wordlist_counts = {}
    for label in labels:
        texts = df[df['Label'] == label]['Teks']
        wordlist_counts[label] = {}
        for word in top_words:
            word_count = sum(text.count(word) for text in texts)
            wordlist_counts[label][word] = word_count

    # Menghitung total data dalam setiap label
    label_counts = df['Label'].value_counts().to_dict()

    # Menghitung persentase kemunculan setiap kata dalam setiap label
    wordlist_percentages = {}
    for label in labels:
        wordlist_percentages[label] = {}
        total_data = label_counts[label]
        for word in top_words:
            word_count = wordlist_counts[label][word]
            word_percentage = (word_count / total_data) * 100    
            wordlist_percentages[label][word] = "{:.2f}".format(word_percentage)  # Format presisi 2 digit


            

    # Mengambil semua kata dan persentase kemunculannya dalam setiap label
    wordlist_with_label = []
    for label in labels:
        for word in top_words:
            word_percentage = wordlist_percentages[label][word]
            wordlist_with_label.append([label, word, word_percentage])

    return wordlist_with_label

# Route untuk halaman sentimen2
@app.route('/sentimen2', methods=['GET', 'POST'])
def sentimen2():
    if request.method == 'POST':
        query = request.form['query']
        if query:
            result = analyze_text_sentimen2(query)
            return render_template('sentimen2.html', result=result, query=query)
    return render_template('sentimen2.html')

# Analisis teks untuk halaman sentimen2
def analyze_text_sentimen2(query):
    # Membaca dataset
    df = pd.read_csv('dataset_label_2.csv')

    # Preprocessing data
    df['Teks'] = df['Teks'].fillna('')  # Replace missing values with empty string
    df['Teks'] = df['Teks'].str.lower()

    # Mencari query dalam teks
    query_sentiments = []
    for text in df['Teks']:
        if query.lower() in text:
            sentiment = df[df['Teks'] == text]['Label'].values[0]
            query_sentiments.append(sentiment)

    # Menghitung persentase kemunculan setiap label
    label_counts = pd.Series(query_sentiments).value_counts(normalize=True) * 100

    # Mengambil label sentimen dan persentase kemunculannya
    result = [{'Label Sentimen': label, 'Persentase Sentimen': percentage} for label, percentage in label_counts.items()]

    return result


@app.route('/sentimen')
def sentimen():
    # Mengambil wordlist dengan label
    wordlist_with_label = analyze_text()
    return render_template('sentimen.html', wordlist_with_label=wordlist_with_label)

# Route untuk halaman beranda
@app.route('/')
def beranda():
    return render_template('beranda.html')

# Route untuk pencarian
@app.route('/search')
def search():
    query = request.args.get('query')
    if query:
        wordlist_with_label = analyze_text(limit=None)  # Mengambil semua kata untuk pencarian
        filtered_wordlist_with_label = [item for item in wordlist_with_label if query.lower() in item[1].lower()]
        return render_template('sentimen.html', wordlist_with_label=filtered_wordlist_with_label, query=query)
    else:
        return redirect('/sentimen')

if __name__ == '__main__':
    app.run(debug=True)
