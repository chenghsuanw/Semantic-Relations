import numpy as np
import tensorflow as tf
import os

from collections import Counter
from model import *
from utils import *

flags = tf.app.flags

flags.DEFINE_string('training_file', default='../TRAIN_FILE.txt', help='training file path')
flags.DEFINE_string('testing_file', default='../TEST_FILE.txt', help='testing file path')
flags.DEFINE_string('answer_file', default='../answer_key.txt', help='answer path')
flags.DEFINE_string('word2vec', default='../glove.txt', help='pretrained word2vector path')
flags.DEFINE_integer('word_embed_dim', default=300, help='dimension of the word embedding')
flags.DEFINE_integer('length', default=40, help='sequence length')
flags.DEFINE_integer('rnn_units', default=64, help='number of hidden units of rnn cell')
flags.DEFINE_integer('hidden_layer', default=128, help='number of hidden units of fully connected layer')

flags.DEFINE_integer('epochs', default=20, help='epochs when training')
flags.DEFINE_integer('batch_size', default=100, help='batch size per iteration')
flags.DEFINE_float('lr_rate', default=0.001, help='learning rate')

FLAGS = flags.FLAGS


def main():
	train_data = load_training_data(FLAGS.training_file)
	test_data = load_testing_data(FLAGS.testing_file)
	test_answer = load_test_answer(FLAGS.answer_file)
	sentences = [data.sentence for data in train_data]

	if not os.path.exists('w2i.pickle'):
		build_dictionary(sentences)
	with open('w2i.pickle', 'rb') as f:
		w2i = pickle.load(f)
	
	if not os.path.exists('w2v.npy'):
		build_word_vector(w2i, FLAGS.word2vec, FLAGS.word_embed_dim)
	w2v = np.load('w2v.npy')


	model = BILSTMDISTANCE(FLAGS, w2i)
	model.build_model(w2v)

	model.train(train_data, test_data, test_answer)

	predict = model.eval(test_data, 'model')
	test_ID = [data.ID for data in test_data]
	output_data = list(zip(test_ID, predict))

	write_file('../output.txt', output_data)


if __name__ == '__main__':
	main()
