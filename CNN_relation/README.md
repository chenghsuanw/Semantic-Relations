###Environment###
python3
macOS
keras 2.08

###Proprocess###
Download pre-train word embedding file from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/ and unzip it.Change the path at line 18 to the correct path for the embeddings file

After executing preprocess.py, two packle files will be stored in ./pkl 

###Training###
Execute CNN.py and the best modle will be stored in ./model

###Testing###
Read the best model and output will be ./answer.txt
