import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Dense
from sklearn.feature_extraction.text import CountVectorizer

# Detect hardware
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
  tpu = None

# Select appropriate distribution strategy
if tpu:
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.TPUStrategy(tpu)
  print("yes tpu")
else:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print("no tpu")

#initialize all variables
input_texts=[]
target_texts=[]
input_characters=set()
target_characters=set()

# Read the dataset file
df = pd.read_csv('data.csv',encoding='utf-8' ,names=['urdu', 'roman-urdu'], usecols=['urdu', 'roman-urdu'])

# Add '\t' at start and '\n' at end of target_text.
df['roman-urdu'] = '\t' + df['roman-urdu'].str.lower() + '\n'

# Get input_texts and target_texts
input_texts = df['urdu'].str.lower().tolist()
target_texts = df['roman-urdu'].tolist()

# Get input_characters and target_characters
input_characters = sorted(list(set(df['urdu'].str.lower().str.cat(sep=''))))
target_characters = sorted(list(set(df['roman-urdu'].str.cat(sep=''))))

# Get the total length of input and target characters
num_en_chars = len(input_characters)
num_dec_chars = len(target_characters)

# Get the maximum length of input and target text.
max_input_length = df['urdu'].str.len().max()
max_target_length = df['roman-urdu'].str.len().max()

pickle.dump({'input_characters':input_characters,'target_characters':target_characters, 'max_input_length':max_input_length, 'max_target_length':max_target_length, 'num_en_chars':num_en_chars, 'num_dec_chars':num_dec_chars}, open("training_data.pkl", "wb"))

# print("number of encoder characters : ", num_en_chars)
# print("number of decoder characters : ", num_dec_chars)
# print("maximum input length : ", max_input_length)
# print("maximum target length : ", max_target_length)

def bagofcharacters(input_texts,target_texts):
    #initialize encoder , decoder input and target data.
    en_in_data=[] ; dec_in_data=[] ; dec_tr_data=[]
    #padding variable with first character as 1 as rest all 0.
    pad_en=[1]+[0]*(len(input_characters)-1)
    pad_dec=[0]*(len(target_characters)) ; pad_dec[2]=1
    #countvectorizer for one hot encoding as we want to tokenize character so
    #analyzer is true and None the stopwords action.
    cv=CountVectorizer(binary=True,tokenizer=lambda txt:
    txt.split(),stop_words=None,analyzer='char')

    for i,(input_t,target_t) in enumerate(zip(input_texts,target_texts)):
        #fit the input characters into the CountVectorizer function
        cv_inp= cv.fit(input_characters)

        #transform the input text from the help of CountVectorizer fit.
        #it character present than put 1 and 0 otherwise.
        en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())
        cv_tar= cv.fit(target_characters)
        dec_in_data.append(cv_tar.transform(list(target_t)).toarray().tolist())
        #decoder target will be one timestep ahead because it will not consider
        #the first character i.e. '\t'.
        dec_tr_data.append(cv_tar.transform(list(target_t)[1:]).toarray().tolist())

        #add padding variable if the length of the input or target text is smaller
        #than their respective maximum input or target length.
        if len(input_t) < max_input_length:
            for _ in range(max_input_length-len(input_t)):
                en_in_data[i].append(pad_en)
        if len(target_t) < max_target_length:
            for _ in range(max_target_length-len(target_t)):
                dec_in_data[i].append(pad_dec)
        if (len(target_t)-1) < max_target_length:
            for _ in range(max_target_length-len(target_t)+1):
                dec_tr_data[i].append(pad_dec)
        #del input_t, target_t
        #gc.collect()
    #convert list to numpy array with data type float32
    en_in_data=np.array(en_in_data,dtype="float32")
    dec_in_data=np.array(dec_in_data,dtype="float32")
    dec_tr_data=np.array(dec_tr_data,dtype="float32")
    return en_in_data,dec_in_data,dec_tr_data

with strategy.scope():
    #create input object of total number of encoder characters
  en_inputs = Input(shape=(None, num_en_chars))
  #create LSTM with the hidden dimension of 256
  #return state=True as we don't want output sequence.
  encoder = LSTM(1024, return_state=True)
  #discard encoder output and store hidden and cell state.
  en_outputs, state_h, state_c = encoder(en_inputs)
  en_states = [state_h, state_c]

  #create input object of total number of decoder characters
  dec_inputs = Input(shape=(None, num_dec_chars))
  #create LSTM with the hidden dimension of 256
  #return state and return sequences as we want output sequence.
  dec_lstm = LSTM(1024, return_sequences=True, return_state=True)
  #initialize the decoder model with the states on encoder.
  dec_outputs, _, _ = dec_lstm(dec_inputs, initial_state=en_states)
  #Output layer with shape of total number of decoder characters
  dec_dense = Dense(num_dec_chars, activation="softmax")
  dec_outputs = dec_dense(dec_outputs)
  #create Model and store all variables
  model = Model([en_inputs, dec_inputs], dec_outputs)
  #load the data and train the model
  en_in_data,dec_in_data,dec_tr_data = bagofcharacters(input_texts,target_texts)
  model.compile(
      optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
  )

model.fit(
    [en_in_data, dec_in_data],
    dec_tr_data,
    batch_size= 128,
    epochs=30,
    validation_split=0.2,
)

#Save model
model.save("model.h5")

#summary
model.summary()