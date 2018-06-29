###Environment###
macOS
keras 2.08

###Proprocess###
Download pre-train word embedding file from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/ and unzi it.Change the path at line 18 to the correct path for the embeddings file

After executing preprocess.py, two packle files will be stored in ./pkl 

###Training###
The best modle will be stored in ./model

###Testing###
Read the best model and output will be ./answer.txt
