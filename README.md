# Semantic-Relations

## How to execute

### Bi-directional LSTM with location labeled
Run the model with <br>
```cd bi-directional_LSTM_position_labeled```<br>
and then <br>
```python3 bi-directionl_LSTM_model.py```<br><br>
Note that parameters can be set in the<br>
```if __name__ == '__main__':```<br>
part at the end of the file.<br>
Directory of pre-trained GloVe embeddings and training, testing data can be set too.

### RNN2
```cd RNN2/source``` <br>
and then <br>
```python3 main.py```

### Similarity
```
cd Similarity
python3.6 similarity.py --train_path [path of training data] --test_path [path of testing data]
```
