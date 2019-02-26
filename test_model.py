from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Bidirectional, Dropout, CuDNNGRU,Conv1D,BatchNormalization,Activation,MaxPool1D,SimpleRNN
from Attention import Attention
from sklearn.externals import joblib



num_words = 20000
max_len =1200


train_list = [i.strip() for i in open('C:/Users/ASUS/Desktop/train_test.txt', 'r', encoding='utf-8')]
print(len(train_list))
label_bj = [i.strip() for i in open('C:/Users/ASUS/Desktop/label_bj1.txt', 'r', encoding='utf-8')]
# tokenizer = Tokenizer(num_words = num_words)
# tokenizer.fit_on_texts(train_list)
tokenizer = joblib.load('tokenizer_final.model')

train_list = tokenizer.texts_to_sequences(train_list)
x_train = pad_sequences(train_list, max_len)
lb = joblib.load("lb.model")


model = Sequential([
    Embedding(num_words + 1, 128, input_shape=(max_len,)),
    Conv1D(128, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # MaxPool1D(10),
    Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='sum'),
    Attention(128),
    Dropout(0.5),
    Dense(3, activation='softmax')
])




model.load_weights('model.h5')  #选取自己的.h模型名称
count = 0
pre = model.predict(x_train)
pre = (pre > 0.5).astype('int32')
pre = lb.inverse_transform(pre)
for i in range(len(label_bj)):
    if pre[i] == label_bj[i]:
        count+=1
print ('识别为：',count/881)
pass

