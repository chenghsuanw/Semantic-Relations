from keras.models import load_model
import gzip
import pickle as pkl
import numpy as np

print("Load dataset")
f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
yTrain, sentenceTrain, positionTrain1, positionTrain2 = pkl.load(f)
yTest, sentenceTest, positionTest1, positionTest2  = pkl.load(f)
f.close()


model = load_model('model/best.hdf5')
result = np.argmax(model.predict([sentenceTest, positionTest1, positionTest2]),axis = 1 ) #argmax to find index


labelsMapping = {'Other':0,
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2,
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4,
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6,
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}

with open("answer.txt",'w') as f:
    for i, element in enumerate(result):
        relat_type = [key for key, value in labelsMapping.items() if value == element][0] #find the correspond type
        f.write(str(i+8001) + "\t" + relat_type + "\n")   
