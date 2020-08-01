import numpy as np
import tensorflow as tf
import os
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers.embeddings import Embedding
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import string
import re
import nltk
from keras.models import model_from_json
import emoji
from collections import Counter
#nltk.download('stopwords')
from keras.layers import Dense, Conv1D, Flatten, Dropout, GlobalMaxPooling1D, LSTM, Input, merge, Bidirectional
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential
import plotly.express as px
from plotly.offline import plot
from plotly import graph_objs as go
from sklearn.utils import shuffle
from keras.utils.vis_utils import plot_model



da_escludere = ["\[", "\]", "\|", "–", "•", "#",":","!", "’",
                "+", "/", "?", ".",",","-","\"", "\'", "“",
                "”", ")", "(", "/", "«", "»", "&", "�", "%", ";",
                "‘", "°"]

def clean_text(text):  
    text = " " + emoji.demojize(text, delimiters=(" ", " ")) 
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"…", " ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\:", " : ", text)
    text = re.sub(r"\'", " ' ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"“", " “ ", text)
    text = re.sub(r"”", " ” ", text)
    text = re.sub(r"#", " # ", text)
    text = re.sub(r"\’", " ’ ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"–", " – ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"€", "€ ", text)
    text = re.sub(r"«", " « ", text)
    text = re.sub(r"\[", " \[ ", text)
    text = re.sub(r"\]", " \] ", text)
    text = re.sub(r"°", " ° ", text)
    text = re.sub(r"»", " » ", text)
    text = re.sub(r"•", " • ", text)
    text = re.sub(r"\|", " \| ", text)
    text = re.sub(r"�", " � ", text)
    text = re.sub(r"%", " % ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"‘", " ‘ ", text)
    text = re.sub(r"&gt;", " ", text)
    text = re.sub(r"&amp;", " ", text)
    text = re.sub(r"&lt;", " ", text)
    text = re.sub(r"URL", "", text)
    text = re.sub(r"@user", "", text)
    text = re.sub(r"mln", "milioni", text)
    text = re.sub('\w*\d\w*', '', text) #Tolgo tutti i numeri
    text = re.sub(r"ke", "che", text)
    text = re.sub(r"xché", "perchè", text)
    
    text = text.lower().split()
    stops = set(stopwords.words("italian-Prog-Systems"))
    text = [w for w in text if not w in stops]
    text = [w for w in text if not w in da_escludere]
    text = " ".join(text)
    return text


# Carico il modello addestrato per riconoscere sentimenti di odio
json_odio = open('Modello_Odio_Salvato/modello_odio.json', 'r')
modello_json_odio = json_odio.read()
json_odio.close()
loaded_model_odio = model_from_json(modello_json_odio)
loaded_model_odio.load_weights("Modello_Odio_Salvato/modello_odio.h5")
print("Il modello 'Odio' è stato caricato correttamente")

#Carico il modello addestrato per riconoscere gli stereotipi
json_stereotipi = open('Modello_Stereotipi_Salvato/modello_stereotipi.json', 'r')
modello_json_stereotipi = json_stereotipi.read()
json_stereotipi.close()
loaded_model_stereotipi = model_from_json(modello_json_stereotipi)
loaded_model_stereotipi.load_weights("Modello_Stereotipi_Salvato/modello_stereotipi.h5")
print("Il modello 'Stereotipi' è stato caricato correttamente")


# Apro il dataset
df_cols = ["testo", "odio", "stereotipi"]
df_dataset = pd.read_csv('haspeede2_dev_taskAB.tsv', delimiter = "\t", encoding='utf-8', names = df_cols)
df_dataset = df_dataset.drop(df_dataset.index[0])
df_dataset['testo'] = df_dataset['testo'].map(lambda x: clean_text(x))


input_utente = input("Inserisci la frase da testare sui modelli: \n")    

# Applico la funzione clean text sul testo inserito dall'utente
input_utente_filtrato = clean_text(input_utente)
input_utente_filtrato = [input_utente_filtrato]
input_utente = [input_utente]

# Creo un dataframe contenente la frase inserita dall'utente ed il testo filtrato
# Questo consentirà di calcolare le feature del testo inserito
cols = ['frase_originale', 'frase_filtrata']
row = {'text_backup' : input_utente , 'frase_filtrata' : input_utente_filtrato}
df = pd.DataFrame(row)


# Numero di parole nella frase
df['Feature_1'] = df['text_backup'].apply(lambda x: len(str(x).split()))

# Numero di lettere in una frase
df['Feature_2'] = df['text_backup'].apply(lambda x: len(str(x)))

# Lunghezza media delle parole
df['Feature_3'] = df['Feature_2'] / df['Feature_1']

# Conto quante stopwords sono presenti all'interno dei testi
stop_words = set(stopwords.words('italian-Prog-Systems'))
df['Feature_4'] = df["text_backup"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))


#Quante parole della frase sono presenti nelle top 20 generali
df_dataset['temp_list'] = df_dataset['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_dataset['temp_list'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_generali = pd.DataFrame(top.most_common(20), columns = n_cols)
lista_parole = df_parole_generali['parola'].values.tolist()
df['Feature_5'] = df['frase_filtrata'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )
df_dataset = df_dataset.drop(columns=['temp_list'])


#Quante parole sono presenti nelle top 10 senza odio
df_odio_0 = df_dataset[df_dataset['odio'] == 0]
df_odio_0['temp_odio_0'] = df_odio_0['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_odio_0['temp_odio_0'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_odio_0 = pd.DataFrame(top.most_common(10), columns=n_cols)
lista_parole = df_parole_odio_0['parola'].values.tolist()
df['Feature_6'] = df['frase_filtrata'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )
#df = df.drop(columns=['temp_odio_0'])


#Quante parole sono presenti nelle top 10 con odio
df_odio_1 = df_dataset[df_dataset['odio'] == 1]
df_odio_1['temp_odio_1'] = df_odio_1['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_odio_1['temp_odio_1'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_odio_1 = pd.DataFrame(top.most_common(10), columns=n_cols)
lista_parole = df_parole_odio_1['parola'].values.tolist()
df['Feature_7'] = df['frase_filtrata'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )
#df = df.drop(columns=['temp_odio_1'])


#Quante parole sono presenti nelle top 10 senza stereotipi
df_ster_0 = df_dataset[df_dataset['stereotipi'] == 0]
df_ster_0['temp_ster_0'] = df_ster_0['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_ster_0['temp_ster_0'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_ster_0 = pd.DataFrame(top.most_common(10), columns=n_cols)
lista_parole = df_parole_ster_0['parola'].values.tolist()
df['Feature_8'] = df['frase_filtrata'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )
#df = df.drop(columns=['temp_ster_0'])


#Quante parole sono presenti nelle top 10 con stereotipi
df_ster_1 = df_dataset[df_dataset['stereotipi'] == 1]
df_ster_1['temp_ster_1'] = df_ster_1['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_ster_1['temp_ster_1'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_ster_1 = pd.DataFrame(top.most_common(10), columns=n_cols)
lista_parole = df_parole_ster_1['parola'].values.tolist()
df['Feature_9'] = df['frase_filtrata'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )
#df = df.drop(columns=['temp_ster_1'])


# Numero di punctuation nella frase
df['Feature_10'] = df['text_backup'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]) )

# Numero di punti esclamativi nella frase
df['Feature_11'] = df['text_backup'].apply(lambda x: x.count('!')) 

# Numero di punti interrogativi nella frase
df['Feature_12'] = df['text_backup'].apply(lambda x: x.count('?')) 

# Numero di parole completamente maiuscole nella frase
df['Feature_13'] = df['text_backup'].apply(lambda x: len([w for w in str(x).split() if w.isupper() == True]) )

# Numero di parole che iniziano con la maiuscola nella frase
df['Feature_14'] = df['text_backup'].apply(lambda x: len([w for w in str(x).split() if w[0].isupper() == True]) )

#Numero di riferimenti ad altri utenti
df['Feature_15'] = df['text_backup'].apply(lambda x: len([w for w in str(x) if w.startswith('@')]))

#Numero di hashtag nella frase
df['Feature_16'] = df['text_backup'].apply(lambda x: len([w for w in str(x) if w.startswith('#')]))

#Numero di emoji inserite nella frase
def trova_numero_emoji(s):
    num_emoji = ''.join(c for c in s if c in emoji.UNICODE_EMOJI)
    return len(num_emoji)
#frase = 'Queste sono'🤔', '🙈', '😌', '💕', '👭' '
#print(trova_emoji(frase))
df['Feature_17'] = df['text_backup'].map(lambda x: trova_numero_emoji(x))


# Numero totale di emoji incluse nelle 5 più usate da chi ha scritto testi contenenti sentimenti di odio
lista_top_5_odio_emoji = ['😡', '😂' , '💩', '🤣', '🇮', '🇹']
df['Feature_18'] = df['text_backup'].apply(lambda x: len([w for w in str(x).split() if w in lista_top_5_odio_emoji]))


# Numero totale di emoji incluse nelle 5 più usate da chi ha scritto testi non contenenti sentimenti di odio
lista_top_5_senza_odio_emoji = ['😂', '🤔', '🇮', '🇹', '👎']
df['Feature_19'] = df['text_backup'].apply(lambda x: len([w for w in str(x).split() if w in lista_top_5_senza_odio_emoji]))


# Numero totale di emoji incluse nelle 5 più usate da chi ha scritto testi contenenti stereotipi
lista_top_5_stereotipi_emoji = ['😡', '😂', '👎', '🇮', '🇹']
df['Feature_20'] = df['text_backup'].apply(lambda x: len([w for w in str(x).split() if w in lista_top_5_stereotipi_emoji]))

# Numero totale di emoji incluse nelle 5 più usate da chi ha scritto testi non contenenti stereotipi
lista_top_5_senza_stereotipi_emoji = ['😂', '😡', '🇮', '🤣', '🤔']
df['Feature_21'] = df['text_backup'].apply(lambda x: len([w for w in str(x).split() if w in lista_top_5_senza_stereotipi_emoji]))


#Effettuo il fit_on_texts sul 'lista_testo' perchè così gli indici delle parole inserite dall'utente
#saranno uguali a quelle con cui il modello si è addestrato
df = df.drop(columns=['text_backup'])
lista_testo = df_dataset["testo"].fillna('').to_list() 
lista_testo = [str(i) for i in lista_testo] 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lista_testo)
max_len = 76

# Trasformo la frase filtrata in formato numerico aggiungendo post padding
df['testo_token'] = tokenizer.texts_to_sequences(df['frase_filtrata'])
df['testo_token_padding'] = pad_sequences(df['testo_token'], padding = "post", maxlen = max_len).tolist()


#Filtro solamente le feature necessarie al modello
df = df[['testo_token_padding','Feature_1','Feature_2','Feature_3','Feature_4',
         'Feature_5','Feature_6','Feature_7','Feature_8','Feature_9','Feature_10',
         'Feature_11','Feature_12', 'Feature_13','Feature_14','Feature_15',
         'Feature_16','Feature_17', 'Feature_18', 'Feature_19', 'Feature_20',
         'Feature_21']]


#Ottengo i valori dal dataframe e suddivido gli input per i modelli
valori_df = df.values 
valori_embedding = np.array([item[0] for item in valori_df])
valori_feature = np.array([item[1:] for item in valori_df])


# Il modello dedicato alla previsione dei sentimenti di odio prende il testo in formato numerico
# e le feature calcolate dal testo stesso
predizione_odio = loaded_model_odio.predict([valori_embedding, valori_feature])

#predizione_stereotipo = loaded_model_stereotipi.predict([valori_embedding, valori_feature])
# Il modello dedicato alla previsione degli stereotipi prende in input solamente il testo in formato numerico
predizione_stereotipo = loaded_model_stereotipi.predict(valori_embedding)


predizione_odio = np.argmax(predizione_odio, axis=1)
predizione_stereotipo = np.argmax(predizione_stereotipo, axis=1)
print()
print("Predizione [Odio]:      ", predizione_odio)
print("Predizione [Stereotipi]:", predizione_stereotipo)

if predizione_odio[0] == 0:
    print("Il testo inserito non contiene sentimenti di odio")
else:
    print("Il testo inserito contiene sentimenti di odio")
    
if (predizione_stereotipo[0] == 0):
    print("Il testo inserito non contiene stereotipi")
else:
    print("Il testo inserito contiene stereotipi")


# Previsione 1 1:
# Maledetti voi che venite in italia !!!!!!!!! 😡😡😡 dovete restare a casa vostra, qui siete tutti clandestini 💩💩

# Previsione 0 0
# Se da loro c'è la guerra è giusto che emigrino, noi dobbiamo essere accoglienti nei confronti di chi ha più bisogno!

# Previsione 1 0
# Vi odio tutti!!! dovete restare a casa vostra perchè l'italia è degli italiani 😡😡😡

# 1 1 
# Gli italiani sotto un ponte ed i clandestini in hotel? ma andate a casa vostra e lasciate il lavoro agli italiani


# Previsione 1 1
# SIETE TUTTI CLANDESTINI, IN ITALIA NON CI DOVETE VENIRE !!! 😡










