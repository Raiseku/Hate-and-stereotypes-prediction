## Project Overview

The aim of this project is to develop a machine learning model to detect hate speech and stereotypes in Italian tweets.

---

### “Studi sul dataset” file

This code is designed to clean and explore text data using natural language processing techniques. It uses various Python packages, including numpy, tensorflow, pandas, sklearn, matplotlib, nltk, emoji, and plotly.

## **Usage**

This code can be used to load and preprocess text data stored in a TSV file format. The file is loaded using pandas and stored in a dataframe, which is then cleaned and explored. To use this code, simply run it in a Python environment after installing the required packages.

## **Dependencies**

This code requires the following Python packages to be installed:

- numpy
- tensorflow
- pandas
- sklearn
- matplotlib
- nltk
- emoji
- plotly

## **Data Cleaning**

This code cleans the text data by removing punctuation, stopwords, and stem words. It also replaces emojis with their corresponding text representations. Finally, it removes any other unwanted characters, such as URLs and numbers.

## **Data Exploration**

This code provides an example of how to explore and visualize the text data using various techniques. For example, it counts the number of records that belong to each class and provides an example of how to extract records that do not contain any hate speech or stereotypes. It also visualizes the cleaned data using plotly.

---

## Rete_Odio and Rete_Stereotipi file

The project uses Python with TensorFlow and Keras libraries to preprocess the text, build a deep learning model, and evaluate its performance. The text preprocessing includes removing symbols, cleaning text, and stemming. The deep learning model is a sequential neural network with convolutional and recurrent layers. The evaluation includes measuring the model's accuracy, precision, recall, F1 score, and generating a confusion matrix.
