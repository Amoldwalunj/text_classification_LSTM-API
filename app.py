import tensorflow as tf
import os
#import tensorflow_hub as hub
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras .preprocessing.sequence import pad_sequences
import pickle
import json
from flask import Flask
from flask import request
from flask import jsonify

app=Flask(__name__)

model = tf.keras.models.load_model(r'C:\Users\admin\Desktop\New folder\my_model.h5')

f = open(r'C:\Users\admin\Downloads\Labels_dictinary.json')

Labels = json.load(f)

with open(r'C:\Users\admin\Desktop\New folder\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
@app.route('/')
def hello():
    return 'API is running'
    
@app.route('/classify', methods=['POST','GET'])
def predict():
    data = request.json
    text = data['text']
    
    word_vector=tokenizer.texts_to_sequences([text])
    
    max_length=500
    word_vector_padded= pad_sequences(word_vector, maxlen= max_length, padding='post',
    truncating='post')
    
    y_pred= model.predict(word_vector_padded)
    prediction=y_pred.argmax(axis=1)[0]
    
    return Labels[str(prediction)]
    
if __name__=="__main__":
    app.run(host='localhost')






