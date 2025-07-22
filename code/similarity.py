import tkinter as tk
from tkinter import *
from tkinter import ttk
import os.path
import re
import matplotlib
from scipy.stats import pearsonr
matplotlib.use('TkAgg')  # or Qt5Agg
import matplotlib.pyplot as plt
import string
from gensim.parsing import remove_stopwords
import numpy as np
from gensim.models import KeyedVectors
import nltk
import torch
import scipy.stats
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
from googletrans import Translator
from stop_words import get_stop_words
from nltk import WordNetLemmatizer
from main import preprocess3 as g
import scipy.stats
model1 = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')
model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\sharayu\pycharmm\multilingual\new\GoogleNews-vectors-negative300.bin',binary=True)

window = tk.Tk()
window.title("Text Similarity of Multilingual(Hindi-English) text ")
window.geometry("800x600")

def translate_text(text):
    translator = Translator()
    translated = translator.translate(text, src='hi', dest='en')
    return translated.text


def preprocess_hindi(text):
    tokens = nltk.word_tokenize(text)
    # remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    # lemmatize the tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # convert the tokens to lowercase
    tokens = [token.lower() for token in tokens]
    # return the preprocessed text as a string
    return ' '.join(tokens)


def calculate_similarity_lstm(suspicious_file_path, english_file_path):
    # read the suspicious Hindi file
    with open(suspicious_file_path, 'r', encoding='utf-8') as f:
        hindi_text = f.read()
    # encode the Hindi text using the Universal Sentence Encoder
    translated_text = translate_text(hindi_text)
    preprocessed_hindi = preprocess_hindi(translated_text)
    hindi_tokens = nltk.word_tokenize(preprocessed_hindi)
    hindi_embedding = model1([hindi_tokens])[0].numpy()

    # read the English file
    with open(english_file_path,'r', encoding='utf-8') as f:
            english_text = f.read()
    # tokenize the English text and encode using the Universal Sentence Encoder
    english_tokens = nltk.word_tokenize(english_text)
    english_embedding = model1([english_tokens])[0].numpy()

    # calculate cosine similarity
    cosine_similarity = hindi_embedding.dot(english_embedding.T) / (np.linalg.norm(hindi_embedding) * np.linalg.norm(english_embedding))

    # calculate Pearson correlation
    pearson_correlation = scipy.stats.pearsonr(hindi_embedding.flatten(), english_embedding.flatten())[0]

    return cosine_similarity, pearson_correlation

import scipy.stats

def calculate_similarity_word2vec(suspicious_file_path, english_file_path):
    # read the suspicious Hindi file
    with open(suspicious_file_path, 'r', encoding='utf-8') as f:
        hindi_text = f.read()
    translated_text = translate_text(hindi_text)
    preprocessed_hindi = preprocess_hindi(translated_text)
    hindi_tokens = nltk.word_tokenize(preprocessed_hindi)
    hindi_tokens = [token for token in hindi_tokens if token in model.key_to_index]
    hindi_vectors = np.zeros((len(hindi_tokens), model.vector_size))
    for i, token in enumerate(hindi_tokens):
        hindi_vectors[i] = model[token]
    with open(english_file_path,'r', encoding='utf-8') as f:
            english_text = f.read()
    english_tokens = nltk.word_tokenize(english_text)
    english_tokens = [token for token in english_tokens if token in model.key_to_index]
    vector_1 = np.zeros(model.vector_size)
    for token in english_tokens:
        vector_1 += model[token]
    vector_2 = np.zeros(model.vector_size)
    for vector in hindi_vectors:
        vector_2 += vector
    cosine_similarity = vector_1.dot(vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    pearson_correlation = scipy.stats.pearsonr(vector_1, vector_2)[0]
    return cosine_similarity, pearson_correlation

def on_button_click_lstm():
    suspicious_file_path = entry_1.get()
    dir_path = entry_2.get()
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    cosine_similarities = []
    pearson_correlations = []
    for file_path in file_paths:
        cosine_sim, pearson_corr = calculate_similarity_lstm(suspicious_file_path, file_path)
        cosine_similarities.append(cosine_sim)
        pearson_correlations.append(pearson_corr)
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        label = Label(tab3,
                      text=f'{file_name}:Using LSTM-> Cosine Similarity - {cosine_similarities[i]:.2f}, Pearson Correlation - {pearson_correlations[i]:.2f}')
        label.grid(row=i + 2, column=1, padx=10, pady=10)


def on_button_click_word2vec():
    suspicious_file_path = entry_1.get()
    dir_path = entry_2.get()
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    cosine_similarities = []
    pearson_correlations = []
    for file_path in file_paths:
        cosine_sim, pearson_corr = calculate_similarity_word2vec(suspicious_file_path, file_path)
        cosine_similarities.append(cosine_sim)
        pearson_correlations.append(pearson_corr)
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        label = Label(tab3,
                      text=f'{file_name}: Using Word2vec-> Cosine Similarity - {cosine_similarities[i]:.2f}, Pearson Correlation - {pearson_correlations[i]:.2f}')
        label.grid(row=i + 2, column=1, padx=10, pady=10)





def process():
    global final
    raw_text = str(entry.get('1.0', tk.END))
    final = g(raw_text).output

def translate():
    global final
    result = '\ntranslate :{}\n'.format(final["translate"])
    tab1_display.insert(tk.END, result)

def filter():
    global final
    result = '\nfilter :{}\n'.format(final["filter"])
    tab1_display.insert(tk.END, result)


def stopword():
    global final
    result = '\nstopword:{}\n'.format(final["stopwords"])
    tab1_display.insert(tk.END, result)

def lemma():
    global final
    result1 = '\nLemmataization:{}\n'.format(final["Lemmataization"])
    tab1_display.insert(tk.END, result1)

def clear_text():
    global final
    entry.delete('1.0', END)


def clear_display_result():
    tab1_display.delete('1.0', END)

def clear_similarity_result():
    tab3_display.delete('1.0', END)

def on_button_click_word2vec_graph():
    suspicious_file_path = entry_1.get()
    dir_path = entry_2.get()
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    cosine_similarities = []
    pearson_correlations = []
    for file_path in file_paths:
        cosine_sim, pearson_corr = calculate_similarity_word2vec(suspicious_file_path, file_path)
        cosine_similarities.append(cosine_sim)
        pearson_correlations.append(pearson_corr)
    fig, ax = plt.subplots()
    ax.scatter(cosine_similarities, pearson_correlations)
    ax.set_xlabel('Word2vec-> Cosine Similarity')
    ax.set_ylabel('Word2vec-> Pearson Correlation')
    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        ax.annotate(filename, (cosine_similarities[i], pearson_correlations[i]))
    plt.show()
def on_button_click_lstm_graph():
    suspicious_file_path = entry_1.get()
    dir_path = entry_2.get()
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    cosine_similarities = []
    pearson_correlations = []
    for file_path in file_paths:
        cosine_sim, pearson_corr = calculate_similarity_lstm(suspicious_file_path, file_path)
        cosine_similarities.append(cosine_sim)
        pearson_correlations.append(pearson_corr)

    # plot bar chart
    fig, ax = plt.subplots()
    ax.scatter(cosine_similarities, pearson_correlations)
    ax.set_xlabel('LSTM-> Cosine Similarity')
    ax.set_ylabel('LSTM->Pearson Correlation')
    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        ax.annotate(filename, (cosine_similarities[i], pearson_correlations[i]))
    plt.show()


# Create labels and entry widgets
label_1 = ttk.Label(window, text="Enter path of suspicious file:")
label_1.grid(row=0, column=0, padx=10, pady=10)

entry_1 = ttk.Entry(window, width=50)
entry_1.grid(row=0, column=1, padx=10, pady=10)

label_2 = ttk.Label(window, text="Enter directory path of English files:")
label_2.grid(row=1, column=0, padx=10, pady=10)

entry_2 = ttk.Entry(window, width=50)
entry_2.grid(row=1, column=1, padx=10, pady=10)

# Create buttons
browse_button = ttk.Button(window, text="Browse", command=lambda: entry_1.insert(0, filedialog.askopenfilename()))
browse_button.grid(row=0, column=2, padx=10, pady=10)

browse_dir_button = ttk.Button(window, text="Browse", command=lambda: entry_2.insert(0, filedialog.askdirectory()))
browse_dir_button.grid(row=1, column=2, padx=10, pady=10)


clear_result_button = ttk.Button(window, text="Clear Results", command=clear_display_result)
clear_result_button.grid(row=2, column=1, padx=10, pady=10)

check_button = ttk.Button(window, text="LSTM", command=on_button_click_lstm)
check_button.grid(row=3, column=1, padx=10, pady=10)

check_button_word = ttk.Button(window, text="Word2vec", command=on_button_click_word2vec)
check_button_word.grid(row=4, column=1, padx=10, pady=10)

check_button2 = ttk.Button(window, text="Plot LSTM", command=on_button_click_lstm_graph)
check_button2.grid(row=3, column=2, padx=10, pady=10)
check_button2_word2vec = ttk.Button(window, text="Plot Word2vec", command=on_button_click_word2vec_graph)
check_button2_word2vec.grid(row=4, column=2, padx=10, pady=10)
#
# clear_similarity_result = ttk.Button(window, text="Clear Results", command=clear_similarity_result)
# clear_similarity_result.grid(row=2, column=2,padx=10, pady=10)
# Create a tab control with two tabs
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text="Preprocessing")
tab_control.add(tab2, text="Results")
tab_control.add(tab3, text="Similarity Results")
tab_control.grid(row=5, column=0, columnspan=3, padx=10, pady=10)
# Functions
# Create widgets for the preprocessing tab
label_tab1 = ttk.Label(tab1, text="Preprocessing Steps")
label_tab1.grid(row=0, column=0, padx=10, pady=10)

button1 =ttk.Button(tab1, text="process", command=process)
button1.grid(row=4, column=0, padx=10, pady=10)

translate_button = ttk.Button(tab1, text="Translate", command=translate)
translate_button.grid(row=4, column=1, padx=10, pady=10)

filter_button = ttk.Button(tab1, text="Filter", command=filter)
filter_button.grid(row=5, column=0, padx=10, pady=10)

stopword_button = ttk.Button(tab1, text="Stopwords", command=stopword)
stopword_button.grid(row=5, column=1, padx=10, pady=10)

lemma_button = ttk.Button(tab1, text="Lemmataization", command=lemma)
lemma_button.grid(row=6, column=0, padx=10, pady=10)

clear_button = ttk.Button(tab1, text="Clear Text", command=clear_text)
clear_button.grid(row=6, column=1, padx=10, pady=10)

entry = Text(tab1, height=10, width=130)
entry.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

# Create a text widget to display the results
tab1_display = tk.Text(tab2)
tab1_display.grid(row=0, column=0, padx=10, pady=10)

tab3_display = tk.Label(tab3)
tab3_display.grid(row=0, column=0, padx=10, pady=10)

# Start the Tkinter event loop
window.mainloop()