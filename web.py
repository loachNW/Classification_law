import re

import jieba
from flask import Flask, request, jsonify, render_template
from keras.layers import Dense, BatchNormalization, Activation, Embedding, GRU,Conv1D,Bidirectional,Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
from Attention import Attention

restr = r'[0-9\s+\.\!\/_,$%^*();?:\-<>《》【】+\"\']+|[+——！，；。？：、~@#￥%……&*（）]+'



max_len = 1200#每一条的长度
num_words = 20000#

class Region(object):
    def __init__(self):
        self.tokenizer = joblib.load('tokenizer_final.model')
        self.lb = joblib.load('lb.model')
        self.model = self.cnn_rnn_attention()

    def cnn_rnn_attention(self):
        model = Sequential([
            Embedding(num_words + 1, 128, input_shape=(max_len,)),
            Conv1D(128, 3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            # MaxPool1D(10),
            Bidirectional(GRU(128, return_sequences=True,reset_after=True), merge_mode='sum'),
            Attention(128),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        model.load_weights('model.h5')
        return model

    def prdected(self, text):
        resu = text.replace('|', '').replace('&nbsp;', '').replace('ldquo', '').replace('rdquo',
                                                                                                          '').replace(
            'lsquo', '').replace('rsquo', '').replace('“', '').replace('”', '').replace('〔', '').replace('〕', '')
        resu = re.split(r'\s+', resu)
        dr = re.compile(r'<[^>]+>', re.S)
        dd = dr.sub('', ''.join(resu))
        line = re.sub(restr, '', dd)
        seg_list = jieba.lcut(line)
        sequences = self.tokenizer.texts_to_sequences([seg_list])
        data = pad_sequences(sequences, maxlen=1200)
        pred = self.model.predict_proba(data)
        pred.tolist()
        print(pred[0])
        pred = (pred > 0.5).astype('int32')
        return self.lb.inverse_transform(pred)
    def prdected1(self, text):
        resu = text.replace('|', '').replace('&nbsp;', '').replace('ldquo', '').replace('rdquo',
                                                                                                          '').replace(
            'lsquo', '').replace('rsquo', '').replace('“', '').replace('”', '').replace('〔', '').replace('〕', '')
        resu = re.split(r'\s+', resu)
        dr = re.compile(r'<[^>]+>', re.S)
        dd = dr.sub('', ''.join(resu))
        line = re.sub(restr, '', dd)
        seg_list = jieba.lcut(line)
        sequences = self.tokenizer.texts_to_sequences([seg_list])
        data = pad_sequences(sequences, maxlen=1200)
        pred = self.model.predict_proba(data)
        dict = {'其他':pred[0],'环保':pred[1],'食品药品监督管理局':pred[2]}
        # pred = (pred > 0.5).astype('int32')
        # return self.lb.inverse_transform(pred)
        return dict
model_obj = Region()
app = Flask(__name__)

@app.route("/",methods =["POST"])
def index():
    if request.method == "POST":
        text = request.form.get("content")
        if (text != "" and text != None):
            result = {"result":model_obj.prdected(text)[0],"status":"1"}
        else:
            result = {"result":"DateType_error","status":"0"}
        return jsonify(result)
app.run(host = "0.0.0.0", port = 5005, debug = True, use_reloader = False)

@app.route("/dict",methods =["POST"])
def index():
    if request.method == "POST":
        text = request.form.get("content")
        if (text != "" and text != None):
            result = {"result":model_obj.prdected1(text)[0],"status":"1"}
        else:
            result = {"result":"DateType_error","status":"0"}
        return result
app.run(host = "0.0.0.0", port = 5006, debug = True, use_reloader = False)