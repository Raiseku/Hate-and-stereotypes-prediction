import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import string
import re
import nltk
import emoji
import plotly.express as px
from plotly.offline import plot
from plotly import graph_objs as go
import time
from collections import Counter


df_cols = ["testo", "odio", "stereotipi"]
df = pd.read_csv('haspeede2_dev_taskAB.tsv', delimiter = "\t", encoding='utf-8', names = df_cols)
df = df.drop(df.index[0])
df["odio"].replace({"0": 0, "1": 1}, inplace=True)
df["stereotipi"].replace({"0": 0, "1": 1}, inplace=True)
df['text_backup'] = df['testo']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# Codice per contare quanti record appartengono alle rispettive classi
print("Le righe totali sono: ", df.shape[0])
odio_0 = df[df["odio"] == 0]
print("[Odio] Etichettati con classe 0 sono: ", odio_0.shape[0])
odio_1 = df[df["odio"] == 1]
print("[Odio] Etichettati con classe 1 sono: ", odio_1.shape[0])
ster_0 = df[df["stereotipi"] == 0]
print("[Stereotipi] Etichettati con classe 0 sono: ", ster_0.shape[0])
ster_1 = df[df["stereotipi"] == 1]
print("[Stereotipi] Etichettati con classe 1 sono: ", ster_1.shape[0])


# Esempio di ottenimento dei testi non contenenti ne sentimenti di odio ne stereotipi
ster_1 = df[df["stereotipi"] == 0]
ster_1 = ster_1[ster_1["odio"] == 0]
print(ster_1['testo'].head())
print(ster_1.shape)
print(ster_1['testo'].iloc[19])



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
    return text


print()

print()
# Codice per la dimostrazione di pulizia del testo
frase_selezionata = [6704,6609,6199,3780,5001]
selezionati = df['testo'].iloc[frase_selezionata]
puliti = selezionati.map(lambda x: clean_text(x))
for i in range(5):
    print("Testo senza filtri:")
    print(selezionati[i])
    print()
    print("Testo filtrato:")
    print(puliti[i])
    print("__________________________")



# Osservazione interessante, in media chi scrive testi contenenti sentimenti
# di odio o stereotipi scrive frasi più lunghe, quindi le feature
# che andrò a costruire si baseranno sulla lunghezza del testo

df['character_cnt'] = df['testo'].str.len()
character_cnt = df.groupby('odio')['character_cnt'].mean()
print(character_cnt)
character_cnt = df.groupby('stereotipi')['character_cnt'].mean()
print(character_cnt)


df['word_counts'] = df['testo'].str.split().str.len()
word_counts = df.groupby('odio')['word_counts'].mean()
print(word_counts)
word_counts = df.groupby('stereotipi')['word_counts'].mean()
print(word_counts)


df['characters_per_word'] = df['character_cnt']/ df['word_counts']
characters_per_word = df.groupby('odio')['characters_per_word'].mean()
print(characters_per_word)
characters_per_word = df.groupby('stereotipi')['characters_per_word'].mean()
print(characters_per_word)



# Numero di frasi che contengono emoji:
lista_frasi_con_emoji = []
def Trova_frasi_con_emoji(s):
    fr = ''.join(c for c in s if c in emoji.UNICODE_EMOJI)
    if fr:
        lista_frasi_con_emoji.append(fr)
frasi_sole_emoji = df['testo'].map(lambda x: Trova_frasi_con_emoji(x))
print("Il numero di frasi che contengono emoji è: ",len(lista_frasi_con_emoji))


#
# STUDIO SULLE EMOJI
#

def trova_emoji(s):
    num_emoji = ''.join(c for c in s if c in emoji.UNICODE_EMOJI)
    return num_emoji

#Quale è l'emoji più usata da chi scrive testi non contenenti sentimenti di odio?
df_odio_0 = df[df['odio'] == 0]
df_odio_0['temp_list'] = df_odio_0['text_backup'].apply(lambda x: trova_emoji(x))
top = Counter([item for sublist in df_odio_0['temp_list'] for item in sublist])
n_cols = ["emoji","ricorrenze"]
df_emoji = pd.DataFrame(top.most_common(10), columns = n_cols)
print(df_emoji)
#Creo il grafico che mostra le 10 emoji più utilizzate
#fig = px.bar(df_emoji, x = 'ricorrenze', y = 'emoji', title = 'Dieci emoji più utilizzate da chi scrive testi contenenti stereotipi',
#            orientation = 'h', width = 700, height = 700, color = 'emoji')
#plot(fig)

#Quale è l'emoji più usata da chi scrive testi contenenti sentimenti di odio?
df_odio_0 = df[df['odio'] == 1]
df_odio_0['temp_list'] = df_odio_0['text_backup'].apply(lambda x: trova_emoji(x))
top = Counter([item for sublist in df_odio_0['temp_list'] for item in sublist])
n_cols = ["emoji","ricorrenze"]
df_emoji = pd.DataFrame(top.most_common(10), columns = n_cols)
print(df_emoji)
#fig = px.bar(df_emoji, x = 'ricorrenze', y = 'emoji', title = 'Dieci emoji più utilizzate da chi scrive testi contenenti stereotipi',
 #            orientation = 'h', width = 700, height = 700, color = 'emoji')
#plot(fig)



#Quale è l'emoji più usata da chi scrive testi non contenenti stereotipi?
df_odio_0 = df[df['stereotipi'] == 0]
df_odio_0['temp_list'] = df_odio_0['text_backup'].apply(lambda x: trova_emoji(x))
top = Counter([item for sublist in df_odio_0['temp_list'] for item in sublist])
n_cols = ["emoji","ricorrenze"]
df_emoji = pd.DataFrame(top.most_common(10), columns = n_cols)
print(df_emoji)
#Creo il grafico che mostra le 10 emoji più utilizzate
#fig = px.bar(df_emoji, x = 'ricorrenze', y = 'emoji', title = 'Dieci emoji più utilizzate da chi scrive testi contenenti stereotipi',
 #            orientation = 'h', width = 700, height = 700, color = 'emoji')
#plot(fig)

#Quale è l'emoji più usata da chi scrive testi contenenti stereotipi?
df_odio_0 = df[df['stereotipi'] == 1]
df_odio_0['temp_list'] = df_odio_0['text_backup'].apply(lambda x: trova_emoji(x))
top = Counter([item for sublist in df_odio_0['temp_list'] for item in sublist])
n_cols = ["emoji","ricorrenze"]
df_emoji = pd.DataFrame(top.most_common(10), columns = n_cols)
print(df_emoji)
#fig = px.bar(df_emoji, x = 'ricorrenze', y = 'emoji', title = 'Dieci emoji più utilizzate da chi scrive testi contenenti stereotipi',
 #            orientation = 'h', width = 700, height = 700, color = 'emoji')
#plot(fig)


#Applico la funzione per la pulizia del testo a tutte le righe del dataframe
print("Sto pulendo il testo...")
df['testo'] = df['testo'].map(lambda x: clean_text(x))



#
# STUDIO SULLE PAROLE PIù UTILIZZATE ALL'INTERNO DEL DATASET
#

# 20 PAROLE PIù UTILIZZATE IN GENERALE
df['temp_list'] = df['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df['temp_list'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_generali = pd.DataFrame(top.most_common(20), columns = n_cols)
print("20 parole più utilizzate in generale all'interno del dataset sono: ")
print(df_parole_generali)


# 20 PAROLE PIù UTILIZZATE DA CHI SCRIVE TESTI NON CONTENENTI SENTIMENTI DI ODIO
df_odio_0 = df[df['odio'] == 0]
df_odio_0['temp_odio_0'] = df_odio_0['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_odio_0['temp_odio_0'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_odio_0 = pd.DataFrame(top.most_common(20), columns=n_cols)
print("20 parole più utilizzate da chi non ha scritto testi contenenti sentimenti di odio: ")
print(df_parole_odio_0)

# 20 PAROLE PIù UTILIZZATE DA CHI SCRIVE TESTI CONTENENTI SENTIMENTI DI ODIO
df_odio_1 = df[df['odio'] == 1]
df_odio_1['temp_odio_1'] = df_odio_1['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_odio_1['temp_odio_1'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_odio_1 = pd.DataFrame(top.most_common(20), columns=n_cols)
print("20 parole più utilizzate da chi ha scritto testi contenenti sentimenti di odio: ")
print(df_parole_odio_1)


# 20 PAROLE PIù UTILIZZATE DA CHI SCRIVE TESTI NON CONTENENTI SENTIMENTI DI ODIO
df_ster_0 = df[df['stereotipi'] == 0]
df_ster_0['temp_odio_0'] = df_ster_0['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_ster_0['temp_odio_0'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_ster_0 = pd.DataFrame(top.most_common(20), columns=n_cols)
print("20 parole più utilizzate da chi non ha scritto testi contenenti sentimenti di odio: ")
print(df_parole_ster_0)

# 20 PAROLE PIù UTILIZZATE DA CHI SCRIVE TESTI CONTENENTI SENTIMENTI DI ODIO
df_ster_1 = df[df['stereotipi'] == 1]
df_ster_1['temp_odio_1'] = df_ster_1['testo'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_ster_1['temp_odio_1'] for item in sublist])
n_cols = ["parola","ricorrenze"]
df_parole_ster_1 = pd.DataFrame(top.most_common(20), columns=n_cols)
print("20 parole più utilizzate da chi ha scritto testi contenenti sentimenti di odio: ")
print(df_parole_ster_1)

# Stampo i grafici dei dati ottenuti dalle funzioni precedenti

fig = px.bar(df_parole_generali, x = 'ricorrenze', y = 'parola', title = 'Venti parole più utilizzate nel dataset',
             orientation = 'h', width = 700, height = 700, color = 'parola')
plot(fig)

time.sleep(2)

fig = px.bar(df_parole_odio_0, x = 'ricorrenze', y = 'parola', title = 'Venti parole più utilizzate da chi ha scritto testi senza sentimenti di odio',
             orientation = 'h', width = 700, height = 700, color = 'parola')
plot(fig)

time.sleep(2)

fig = px.bar(df_parole_odio_1, x = 'ricorrenze', y = 'parola', title =  'Venti parole più utilizzate da chi ha scritto testi contenenti sentimenti di odio',
             orientation = 'h', width = 700, height = 700, color = 'parola')
plot(fig)

time.sleep(2)


in_comune = pd.merge(df_parole_odio_0, df_parole_odio_1, on = 'parola')
fig = go.Figure()
fig.add_trace(go.Scatter(y=in_comune['ricorrenze_x'], x=in_comune['parola'],
                         name = "Testi non contenenti sentimenti di odio"
                         
                         ))
fig.add_trace(go.Scatter(y=in_comune['ricorrenze_y'], x=in_comune['parola'],
                         name = "Testi contenenti sentimenti di odio"
                         ))

fig.update_layout(
        #title = "Contronto tra le parole in comune",
        title = "Intersezione tra le due liste [Odio] top 20",
        xaxis_title= "Parole",
        yaxis_title= "Frequenza",
        font=dict(
                family="Courier New, monospace",
                size=13,
                color="#7f7f7f"
                )
        )
    
plot(fig)

time.sleep(2)

in_comune = pd.merge(df_parole_ster_0, df_parole_ster_1, on = 'parola')
fig = go.Figure()
fig.add_trace(go.Scatter(y=in_comune['ricorrenze_x'], x=in_comune['parola'],
                         name = "Testi non contenenti stereotipi"
                         
                         ))
fig.add_trace(go.Scatter(y=in_comune['ricorrenze_y'], x=in_comune['parola'],
                         name = "Testi contenenti stereotipi"
                         ))

fig.update_layout(
        #title = "Contronto tra le parole in comune",
        title = "Intersezione tra le due liste [Stereotipi] top 20",
        xaxis_title= "Parole",
        yaxis_title= "Frequenza",
        font=dict(
                family="Courier New, monospace",
                size=13,
                color="#7f7f7f"
                )
        )
    
plot(fig)
















