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
import time
from sklearn.utils import shuffle
from keras.utils.vis_utils import plot_model


# Imposto tutti i random seed per la riproducibilità dei risultati
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
os.environ['PYTHONHASHSEED'] = '0'
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


#Apro il dataset
df_cols = ["testo", "odio", "stereotipi"]
df = pd.read_csv('haspeede2_dev_taskAB.tsv', delimiter = "\t", encoding='utf-8', names = df_cols)
df = df.drop(df.index[0])
df["odio"].replace({"0": 0, "1": 1}, inplace=True)
df["stereotipi"].replace({"0": 0, "1": 1}, inplace=True)
df['text_backup'] = df['testo']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# Definisco tutte i simboli da escludere dal testo
da_escludere = ["\[", "\]", "\|", "–", "•", "#",":","!", "’",
                "+", "/", "?", ".",",","-","\"", "\'", "“",
                "”", ")", "(", "/", "«", "»", "&", "�", "%", ";",
                "‘", "°"]

# Funzione per la pulizia del testo
def clean_text(text):  
    text = " " + emoji.demojize(text, delimiters=(" ", " ")) #Trasforma l'emoji in testo
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
    
    '''
    # Stemming.
    text = text.split() #fai lo split della frase così ottieni le parole singole
    stemmer = SnowballStemmer('italian') 
    stemmed_words = [stemmer.stem(word) for word in text] #Applica a parola per parola il processo di stemming
    text = " ".join(stemmed_words)
    '''
    return text



print("Sto pulendo il testo...")
df['testo'] = df['testo'].map(lambda x: clean_text(x))

# Creo le Feature

# Numero di parole nella frase
df['Feature_1'] = df['text_backup'].apply(lambda x: len(str(x).split()))

# Numero di lettere in una frase
df['Feature_2'] = df['text_backup'].apply(lambda x: len(str(x)))

# Lunghezza media delle parole
df['Feature_3'] = df['Feature_2'] / df['Feature_1']

# Numero di stopwords in lingua italiana
stop_words = set(stopwords.words('italian-Prog-Systems'))
df['Feature_4'] = df["text_backup"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))


#Quante parole della frase sono presenti nelle top 20 generali
df['temp_list'] = df['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df['temp_list'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_generali = pd.DataFrame(top.most_common(20), columns = n_cols)
lista_parole = df_parole_generali['parola'].values.tolist()
df['Feature_5'] = df['testo'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )
df = df.drop(columns=['temp_list'])


#Quante parole sono presenti nelle top 10 senza odio
df_odio_0 = df[df['odio'] == 0]
df_odio_0['temp_odio_0'] = df_odio_0['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_odio_0['temp_odio_0'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_odio_0 = pd.DataFrame(top.most_common(10), columns=n_cols)
lista_parole = df_parole_odio_0['parola'].values.tolist()
df['Feature_6'] = df['testo'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )


#Quante parole sono presenti nelle top 10 con odio
df_odio_1 = df[df['odio'] == 1]
df_odio_1['temp_odio_1'] = df_odio_1['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_odio_1['temp_odio_1'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_odio_1 = pd.DataFrame(top.most_common(10), columns=n_cols)
lista_parole = df_parole_odio_1['parola'].values.tolist()
df['Feature_7'] = df['testo'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )


#Quante parole sono presenti nelle top 10 senza stereotipi
df_ster_0 = df[df['stereotipi'] == 0]
df_ster_0['temp_ster_0'] = df_ster_0['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_ster_0['temp_ster_0'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_ster_0 = pd.DataFrame(top.most_common(10), columns=n_cols)
lista_parole = df_parole_ster_0['parola'].values.tolist()
df['Feature_8'] = df['testo'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )


#Quante parole sono presenti nelle top 10 con stereotipi
df_ster_1 = df[df['stereotipi'] == 1]
df_ster_1['temp_ster_1'] = df_ster_1['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_ster_1['temp_ster_1'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_ster_1 = pd.DataFrame(top.most_common(10), columns=n_cols)
lista_parole = df_parole_ster_1['parola'].values.tolist()
df['Feature_9'] = df['testo'].apply(lambda x: len([w for w in str(x).lower().split() if w in lista_parole]) )



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




#Definisco la funzione per stampare le performance del modello
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5), dpi = 130)
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy durante il training e validation')
    plt.xlabel('Numero di epoche')
    plt.ylabel('Accuratezza')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Loss durante il training e validation')
    plt.xlabel('Numero di epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Mischio i record del dataset
df = shuffle(df)
df = df.drop(columns=['text_backup'])
df = df.drop(columns=['stereotipi'])
# Metti tutto quello che trovi nella colonna 'testo' nella lista chiamata lista_testo
lista_testo = df["testo"].fillna('').to_list() 
# prendo tutti i valori della mia lista e li casto a stringa
lista_testo = [str(i) for i in lista_testo] 
# Inizializzo il Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lista_testo)
vocab_size = len(tokenizer.word_index) + 1
print("Le parole all'interno del vocabolario sono: ", vocab_size) 
lista_testo_tokenizer = tokenizer.texts_to_sequences(lista_testo)
max_len = max(len(x) for x in lista_testo_tokenizer) # è 76
df['testo_token'] = tokenizer.texts_to_sequences(df['testo'])
print("La lunghezza prima il post padding è: ", len(df['testo_token'].iloc[1]))
#Voglio tutte le parole della stessa lunghezza quindi aggiungo zero padding
df['testo_token_padding'] = pad_sequences(df['testo_token'], padding = "post", maxlen = max_len).tolist()
print("La lunghezza dopo il post padding è: ", len(df['testo_token_padding'].iloc[1]))


# Filtro ed ottengo solamente le colonne da dare in input al modello inclusa la colonna odio
df = df[['testo_token_padding','Feature_1','Feature_2','Feature_3','Feature_4',
         'Feature_5','Feature_6','Feature_7','Feature_8','Feature_9','Feature_10',
         'Feature_11','Feature_12', 'Feature_13','Feature_14','Feature_15',
         'Feature_16','Feature_17', 'Feature_18', 'Feature_19', 'Feature_20','Feature_21',
         'odio']]

#Ottengo i valori dal dataframe
X = df.iloc[:,0:22].values
Y = df.iloc[:,22].values

#Effettuo lo splitting, 80% al training e 20% al testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# Il primo elemento della lista rappresenta la frase alla quale applicare l'embedding
X_train_embedding = np.array([item[0] for item in X_train])
# Tutti i restanti elementi rappresentano le feature calcolate sulla frase in questione
X_train_feature = np.array([item[1:] for item in X_train])

#Si applica la stessa procedura per il test set.
X_test_embedding = np.array([item[0] for item in X_test])
X_test_feature = np.array([item[1:] for item in X_test])


# Adesso 1 diventa [0 1] e 0 diventa [1 0]. è il formato richiesto da Keras
from keras.utils import to_categorical
y_train = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)



embedding_dim = 76 
#MODELLO CON ENTRAMBI GLI INPUT
input_testo = Input(shape=(max_len,))
x = Embedding(vocab_size, embedding_dim, input_length = max_len, trainable = True)(input_testo)
x = Flatten()(x)

input_feature = Input(shape=(X_train_feature.shape[1],))

model_final = Concatenate()([x, input_feature])
model_final = Dense(64, activation='relu', bias_initializer='zeros')(model_final)
model_final = Dense(32, activation = "relu", bias_initializer='zeros') (model_final)
model_final = Dense(2, activation='softmax', bias_initializer='zeros')(model_final)
model_final = Model([input_testo,input_feature], model_final)
model_final.compile(loss="categorical_crossentropy", optimizer = 'adam', metrics = ["accuracy"])


plot_model(model_final,to_file="model_plot.png", show_shapes = True, show_layer_names = True)

history = model_final.fit(x=[X_train_embedding,X_train_feature], y = np.array(y_train),
                          batch_size = 128, epochs=3, verbose = 1,
                          validation_split=0.2)


y_pred = model_final.predict([X_test_embedding, X_test_feature])
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print("Matrice di confusione:")
print(cm)
plot_history(history)


'''
embedding_dim = 76 #Imposto l'embedding dim da usare nel livello Embedding
#MELLO SONO CON EMBEDDING DI KERAS

input_testo = Input(shape=(max_len,))
x = Embedding(vocab_size, embedding_dim, input_length = max_len, trainable = True)(input_testo)
x = Flatten()(x)

model_final = Dense(64, activation='relu', bias_initializer='zeros')(x)
model_final = Dense(32, activation = "relu", bias_initializer='zeros') (model_final)
model_final = Dense(2, activation='softmax', bias_initializer='zeros')(model_final)
model_final = Model(input_testo, model_final)
model_final.compile(loss="categorical_crossentropy", optimizer = 'adam', metrics = ["accuracy"])


#plot_model(model_final,to_file="model_plot.png", show_shapes = True, show_layer_names = True)

history = model_final.fit(x=X_train_embedding, y = np.array(y_train),
                          batch_size = 128, epochs=3, verbose = 1,
                          validation_split=0.2)


y_pred = model_final.predict(X_test_embedding)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print("Matrice di confusione:")
print(cm)
plot_history(history)
'''

'''
#MODELLO SOLO FEATURE

embedding_dim = 76 #Imposto l'embedding dim da usare nel livello Embedding

input_feature = Input(shape=(X_train_feature.shape[1],))
model_final = Dense(64, activation='relu', bias_initializer='zeros')(input_feature)
model_final = Dense(32, activation = "relu", bias_initializer='zeros') (input_feature)
model_final = Dense(2, activation='softmax', bias_initializer='zeros')(model_final)
model_final = Model(input_feature, model_final)
model_final.compile(loss="categorical_crossentropy", optimizer = 'adam', metrics = ["accuracy"])

history = model_final.fit(x=X_train_feature, y = np.array(y_train),
                          batch_size = 128, epochs=12, verbose = 1,
                          validation_split=0.2)


y_pred = model_final.predict(X_test_feature)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print("Matrice di confusione:")
print(cm)
plot_history(history)
'''


#Salvataggio del modello
model_json = model_final.to_json()
with open("Modello_Odio_Salvato/modello_odio.json", "w+") as json_file:
    json_file.write(model_json)
model_final.save_weights("Modello_Odio_Salvato/modello_odio.h5")
print("Il modello è stato salvato correttamente")







