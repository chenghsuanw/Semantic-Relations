import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import *


class BILSTMDISTANCE(object):
	def __init__(self, FLAGS, w2i):
		self.FLAGS = FLAGS
		self.w2i = w2i

	def build_model(self, w2v):
		self.input_sequence = tf.placeholder(tf.int32, shape=[None, self.FLAGS.length], name='input_sequence')
		self.e1_distance = tf.placeholder(tf.float32, shape=[None, self.FLAGS.length, 1], name='e1_distance')
		self.e2_distance = tf.placeholder(tf.float32, shape=[None, self.FLAGS.length, 1], name='e2_distance')
		self.relation = tf.placeholder(tf.int32, shape=[None, 19], name='relation')

		self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

		with tf.variable_scope('Embedding'):
			self.w2v = tf.get_variable('w2v', initializer=w2v)
			self.input_embedding = tf.nn.embedding_lookup(self.w2v, self.input_sequence)
			# concate e1_distance and e2_distance to embedding
			self.input_embedding = tf.concat([self.input_embedding, self.e1_distance, self.e2_distance], axis=2)

		with tf.variable_scope('LSTM'):
			self.fw_cell = tf.contrib.rnn.BasicLSTMCell(self.FLAGS.rnn_units)
			self.fw_cell = tf.contrib.rnn.DropoutWrapper(self.fw_cell, output_keep_prob=self.keep_prob)
			self.bw_cell = tf.contrib.rnn.BasicLSTMCell(self.FLAGS.rnn_units)
			self.bw_cell = tf.contrib.rnn.DropoutWrapper(self.bw_cell, output_keep_prob=self.keep_prob)
			# outputs: (output_fw, output_bw)
			# output_fw(output_bw) shape: [None, length, rnn_units]
			# output_states: (output_state_fw, output_state_bw)
			outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.input_embedding, dtype=tf.float32)
			self.final_state_comb = tf.concat([output_states[0][0], output_states[1][0]], axis=1)

		with tf.variable_scope('Fully_Connect'):
			W1 = tf.get_variable('W1', initializer=tf.random_normal([2*self.FLAGS.rnn_units, 19], stddev=0.1))
			b1 = tf.get_variable('b1', initializer=tf.random_normal([19], stddev=0.1))
			self.out = tf.matmul(self.final_state_comb, W1) + b1
			self.predict = tf.argmax(tf.nn.softmax(self.out), axis=1)
			
		with tf.name_scope('Loss'):
			self.truth = tf.argmax(self.relation, axis=1)
			self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.relation))

		with tf.name_scope('Optimizer'):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=10e-3)
			self.train_op = self.optimizer.minimize(self.loss)

	def train(self, train_data, test_data, test_answer):
		train_sentences = [data.sentence for data in train_data]
		train_relations = [data.relation for data in train_data]
		test_sentences = [data.sentence for data in test_data]
		train_sen = tokenize_sentence(train_sentences, self.w2i, self.FLAGS.length)
		train_rel = tokenize_relation(train_relations)
		test_sen = tokenize_sentence(test_sentences, self.w2i, self.FLAGS.length)
		test_rel = tokenize_relation(test_answer)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(max_to_keep=1000)

			batch_size = self.FLAGS.batch_size
			batch_num = len(train_sen) // batch_size

			for e in range(self.FLAGS.epochs):
				total_loss = 0
				train_acc = 0
				for b in tqdm(range(batch_num)):
					# training data
					e1_distance = [np.arange(self.FLAGS.length) for _ in range(batch_size)]
					e2_distance = [np.arange(self.FLAGS.length) for _ in range(batch_size)]
					data = train_data[b*batch_size: (b+1)*batch_size]
					for i in range(batch_size):
						e1_distance[i] -= data[i].pos_e1
						e2_distance[i] -= data[i].pos_e2
					e1_distance = np.array(e1_distance).reshape((-1, self.FLAGS.length, 1))
					e2_distance = np.array(e2_distance).reshape((-1, self.FLAGS.length, 1))
					
					feed_dict = {
						self.input_sequence: train_sen[b*batch_size: (b+1)*batch_size],
						self.e1_distance: e1_distance,
						self.e2_distance: e2_distance,
						self.relation: train_rel[b*batch_size: (b+1)*batch_size],
						self.keep_prob: 0.5
					}
					
					sess.run(self.train_op, feed_dict=feed_dict)
					
					loss, truth, predict = sess.run([self.loss, self.truth, self.predict], feed_dict=feed_dict)
					acc = np.mean(np.equal(truth, predict))
					total_loss += loss/batch_size
					train_acc += acc/batch_num

				# testing data
				e1_distance = [np.arange(self.FLAGS.length) for _ in range(len(test_data))]
				e2_distance = [np.arange(self.FLAGS.length) for _ in range(len(test_data))]
				data = test_data
				for i in range(len(data)):
					e1_distance[i] -= data[i].pos_e1
					e2_distance[i] -= data[i].pos_e2
				e1_distance = np.array(e1_distance).reshape((-1, self.FLAGS.length, 1))
				e2_distance = np.array(e2_distance).reshape((-1, self.FLAGS.length, 1))
				
				feed_dict = {
					self.input_sequence: test_sen,
					self.e1_distance: e1_distance,
					self.e2_distance: e2_distance,
					self.relation: test_rel,
					self.keep_prob: 1.0
				}

				truth, predict = sess.run([self.truth, self.predict], feed_dict=feed_dict)
				test_acc = np.mean(np.equal(truth, predict))

				saver.save(sess, 'model')
				print('epoch {}, loss: {}, train_acc: {}, test_acc: {}'.format(e, total_loss, train_acc, test_acc))

	def eval(self, test_data, model_path):
		test_sentences = [data.sentence for data in test_data]
		test_sen = tokenize_sentence(test_sentences, self.w2i, self.FLAGS.length)
		e1_distance = [np.arange(self.FLAGS.length) for _ in range(len(test_data))]
		e2_distance = [np.arange(self.FLAGS.length) for _ in range(len(test_data))]
		for i in range(len(test_data)):
			e1_distance[i] -= test_data[i].pos_e1
			e2_distance[i] -= test_data[i].pos_e2
		e1_distance = np.array(e1_distance).reshape((-1, self.FLAGS.length, 1))
		e2_distance = np.array(e2_distance).reshape((-1, self.FLAGS.length, 1))
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, model_path)

			feed_dict = {
				self.input_sequence: test_sen,
				self.e1_distance: e1_distance,
				self.e2_distance: e2_distance
			}

			predict = sess.run(self.predict, feed_dict=feed_dict)

			return predict