from keras.preprocessing.text import Tokenizer
from sklearn.externals import joblib
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Dense, Bidirectional, Dropout, CuDNNGRU,Conv1D,BatchNormalization,Activation,MaxPool1D,SimpleRNN
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
from Attention import Attention


max_len = 1200#每一条的长度
num_words = 20000#出现次数最多的前两万个词

train_list = [i.strip() for i in open('train.txt', 'r', encoding='utf8')]
tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(train_list)
joblib.dump(tokenizer,'tokenizer_final.model')#将tokenizer存储到tokenizer_final.model
label_list = [i.strip() for i in open('label2.txt', 'r', encoding='utf8')]
lb = LabelBinarizer()#将对应的数据转换为二进制型，有点类似于onehot编码，这里有几点不同，LabelBinarizer可以处理数值型和类别型数据，输入必须为1D数组，可以自己设置正类和父类的表示方式，
lb.fit(label_list)
joblib.dump(lb, 'lb.model')
pass

train_list = tokenizer.texts_to_sequences(train_list)
x_train = pad_sequences(train_list, max_len)
y_train = lb.transform(label_list)
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.1, random_state=10)

model = Sequential([
    Embedding(num_words + 1, 128, input_shape=(max_len,)),
    Conv1D(128, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # MaxPool1D(10),
    Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='sum'),
    Attention(128),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])


model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.001), metrics=['acc'])
save_best = ModelCheckpoint('model111.h5', verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(patience=1, verbose=1, cooldown=1, factor=0.4)
early_stop = EarlyStopping(patience=3, verbose=1)
model.fit(train_x,train_y, batch_size=200, epochs=30, validation_data=(test_x, test_y), callbacks=[ save_best,early_stop,reduce_lr])
# i = 0
