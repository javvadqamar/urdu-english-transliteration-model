import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Input,LSTM,Dense

cv=CountVectorizer(binary=True,tokenizer=lambda txt: txt.split(),stop_words=None,analyzer='char')

class LangTRans:
    def __init__(self):
        self.datafile()

    def datafile(self):
        datafile = pickle.load(open("training_data.pkl","rb"))
        self.input_characters = datafile['input_characters']
        self.target_characters = datafile['target_characters']
        self.max_input_length = datafile['max_input_length']
        self.max_target_length = datafile['max_target_length']
        self.num_en_chars = datafile['num_en_chars']
        self.num_dec_chars = datafile['num_dec_chars']
        self.loadmodel()

    def loadmodel(self):
        model = models.load_model("model.h5")
        enc_outputs, state_h_enc, state_c_enc = model.layers[2].output
        self.en_model = Model(model.input[0], [state_h_enc, state_c_enc])

        dec_state_input_h = Input(shape=(1024,), name="input_3")
        dec_state_input_c = Input(shape=(1024,), name="input_4")
        dec_states_inputs = [dec_state_input_h, dec_state_input_c]

        dec_lstm = model.layers[3]
        dec_outputs, state_h_dec, state_c_dec = dec_lstm(
            model.input[1], initial_state=dec_states_inputs
        )
        dec_states = [state_h_dec, state_c_dec]
        dec_dense = model.layers[4]
        dec_outputs = dec_dense(dec_outputs)
        self.dec_model = Model(
            [model.input[1]] + dec_states_inputs, [dec_outputs] + dec_states
        )

    def decode_sequence(self,input_seq):
        reverse_target_char_index = dict(enumerate(self.target_characters))
        states_value = self.en_model.predict(input_seq, verbose=0)

        co=cv.fit(self.target_characters)
        target_seq=np.array([co.transform(list("\t")).toarray().tolist()],dtype="float32")

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_chars, h, c = self.dec_model.predict([target_seq] + states_value, verbose=0)

            char_index = np.argmax(output_chars[0, -1, :])
            text_char = reverse_target_char_index[char_index]
            decoded_sentence += text_char

            if text_char == "\n" or len(decoded_sentence) > self.max_target_length:
                stop_condition = True
            target_seq = np.zeros((1, 1, self.num_dec_chars))
            target_seq[0, 0, char_index] = 1.0
            states_value = [h, c]
        return decoded_sentence

    def bagofcharacters(self,input_t):
        cv=CountVectorizer(binary=True,tokenizer=lambda txt: txt.split(),stop_words=None,analyzer='char')
        en_in_data=[] ; pad_en=[1]+[0]*(len(self.input_characters)-1)

        cv_inp= cv.fit(self.input_characters)
        en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())

        if len(input_t)< self.max_input_length:
          for _ in range(self.max_input_length-len(input_t)):
            en_in_data[0].append(pad_en)

        return np.array(en_in_data,dtype="float32")

    def is_english(self, msg):
        return msg.isascii()
    def deocded_output(self,msg):
        if self.is_english(msg):
          return msg
        en_in_data = self.bagofcharacters(msg.lower()+".")
        return self.decode_sequence(en_in_data)


    def my_msg(self, msg):
        if not msg:
            return
        #print("Urdu: " + msg)
        words = msg.split()
        decoded_words = [self.deocded_output(word) for word in words]
        decoded_msg = ' '.join(decoded_words)
        print("Decoded: " + decoded_msg)

if __name__=="__main__":
    LT = LangTRans()
    while True:
        msg = input("Enter your message: ")
        LT.my_msg(msg)