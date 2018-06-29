import numpy as np
np.random.seed(1337)  # for reproducibility

import pickle as pkl
import gzip
import keras
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils import np_utils


batch_size = 64
nb_filter = 100
filter_length = 3
hidden_dims = 100
nb_epoch = 100
position_dims = 50
#position_dims = 300

print("Load dataset")
f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
yTrain, sentenceTrain, positionTrain1, positionTrain2 = pkl.load(f)
yTest, sentenceTest, positionTest1, positionTest2  = pkl.load(f)
f.close()

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1  #0~63
#print(max_position)  #64

n_out = max(yTrain)+1
train_y_cat = np_utils.to_categorical(yTrain, n_out)
test_y_cat = np_utils.to_categorical(yTest, n_out)

'''
print("sentenceTrain: ", sentenceTrain.shape)
print( "positionTrain1: ", positionTrain1.shape)
print("yTrain: ", yTrain.shape)

print "sentenceTest: ", sentenceTest.shape
print "positionTest1: ", positionTest1.shape
print "yTest: ", yTest.shape
'''

f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

#print("Embeddings shape: ",embeddings.shape)


#distanceModel1 = Sequential()
#distanceModel1.add(Embedding(max_position, position_dims, input_length=positionTrain1.shape[1]))
dis_input1 = keras.layers.Input(shape=(positionTrain1.shape[1],))
distanceModel1 = keras.layers.Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])(dis_input1)
#distanceModel1.summary()

#distanceModel2 = Sequential()
#distanceModel2.add(Embedding(max_position, position_dims, input_length=positionTrain2.shape[1]))
dis_input2 = keras.layers.Input(shape=(positionTrain2.shape[1],))
distanceModel2 = keras.layers.Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])(dis_input2)
#distanceModel2.summary()

#wordModel = Sequential()
#wordModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False))
#wordModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings]))
word_input = keras.layers.Input(shape=(sentenceTrain.shape[1],))
wordModel = keras.layers.Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings], trainable = False)(word_input)
#wordModel.summary()

concat = keras.layers.concatenate([distanceModel1, distanceModel2,wordModel])
#print(concat.shape)
conv = keras.layers.Conv1D(filters = nb_filter, kernel_size = filter_length, activation='tanh')(concat)
#print(conv.shape)
pooling = keras.layers.GlobalMaxPooling1D()(conv)
#print(pooling.shape)
dropout = keras.layers.Dropout(0.5, noise_shape=None, seed=None)(pooling)
#print(dropout.shape)
classifier = keras.layers.Dense(n_out,activation='softmax')(dropout)
#print(type(classifier))


model = Model(inputs=[word_input, dis_input1, dis_input2], outputs = classifier)
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
model.summary()
#print "Start training"

filepath = "model/best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit([sentenceTrain, positionTrain1, positionTrain2], train_y_cat, batch_size=batch_size, 
        validation_data = ([sentenceTest, positionTest1, positionTest2], test_y_cat) ,verbose=True, epochs =  70,
        callbacks = [checkpoint] )   

