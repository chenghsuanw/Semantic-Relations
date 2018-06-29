import numpy as np
import re
import pickle
from collections import Counter

class Data(object):
	def __init__(self, ID, sentence, relation, comment, pos_e1=-1, pos_e2=-1):
		self.ID = ID
		self.sentence = sentence
		self.relation = relation
		self.comment = comment
		self.pos_e1 = pos_e1
		self.pos_e2 = pos_e2

def load_training_data(training_file):
	print('Loading training data...')

	with open(training_file, 'r') as f:
		contents = f.readlines()
	lines_num = len(contents)-1 # the final line is empty
	line = 0
	data = []
	while line < lines_num:
		first_line = contents[line].split(maxsplit=1)
		ID = first_line[0]
		sentence = first_line[1].strip()[1:-1]		
		# find e1 and e2 position
		words = sentence.split()
		for pos, w in enumerate(words):
			if w[:4] == '<e1>':
				pos_e1 = pos
			elif w[:4] == '<e2>':
				pos_e2 = pos
		sentence = re.sub(r'<e1>|</e1>|<e2>|</e2>', ' ', sentence)
		sentence = re.sub(r'[^\w\s]', ' ', sentence)
		relation = contents[line+1].strip()
		comment = contents[line+2].split(':', maxsplit=1)[1].strip()
		data.append(Data(ID, sentence, relation, comment, pos_e1, pos_e2))
		line += 4

	return data

def load_testing_data(testing_file):
	print('Loading testing data...')

	with open(testing_file, 'r') as f:
		contents = f.readlines()

	data = []
	for line in contents:
		line = line.split(maxsplit=1)
		ID = line[0]
		sentence = line[1].strip()[1:-1]
		# find e1 and e2 position
		words = sentence.split()
		for pos, w in enumerate(words):
			if w[:4] == '<e1>':
				pos_e1 = pos
			elif w[:4] == '<e2>':
				pos_e2 = pos
		sentence = re.sub(r'<e1>|</e1>|<e2>|</e2>', ' ', sentence)
		sentence = re.sub(r'[^\w\s]', ' ', sentence)
		data.append(Data(ID, sentence, None, None, pos_e1, pos_e2))

	return data

def load_test_answer(answer_file):
	print('Loading testing answer...')

	with open(answer_file, 'r') as f:
		contents = f.readlines()
	
	answer = []
	for line in contents:
		answer.append(line.split()[1])
	
	return answer

def build_dictionary(sentences, min_count=1):
	print('Building dictionary...')

	w2i = {'<UNK>': 0, '<PAD>': 1}
	index = 2

	cnt = Counter()
	for sentence in sentences:
		cnt.update(sentence.split())

	for word, count in cnt.items():
		if count >= min_count:
			w2i[word] = index
			index += 1

	with open('w2i.pickle', 'wb') as f:
		pickle.dump(w2i, f)

	print('There are {} words in the dictionary.'.format(len(w2i)))
	

def build_word_vector(w2i, w2v_path, word_embed_dim):
	print('Building word vectors...')

	w2v_dict = dict()
	with open(w2v_path) as f:
		content = f.readlines()

	for line in content:
		word, vec = line.strip().split(' ', 1)
		w2v_dict[word] = np.loadtxt([vec], dtype=np.float32)

	w2v = []
	# random assign word embedding for <UNK>
	w2v.append(np.random.normal(0, 1, word_embed_dim).astype(np.float32))
	# random assign word embedding for <PAD>
	w2v.append(np.random.normal(0, 1, word_embed_dim).astype(np.float32))
	
	for word in w2i.keys():
		# assign pretrained word embedding
		if word in w2v_dict:
			w2v.append(w2v_dict[word])
		# assign <UNK> embedding for the word not in the pretrained dictionary
		else:
			w2v.append(w2v[0])

	w2v = np.array(w2v)
	np.save('w2v.npy', w2v)

	print('There are {} word vectors. Each vector has {} dimension.'.format(w2v.shape[0], word_embed_dim))

def tokenize_sentence(sentences, w2i, length):
	tokenized_seq = []
	for sentence in sentences:
		seq = []
		for word in sentence.split():
			# clipping
			if len(seq) == length:
				break
			else:
				if word in w2i:
					seq.append(w2i[word])
				else:
					seq.append(0)
		# padding
		while len(seq) < length:
			seq.append(1)
		tokenized_seq.append(seq)

	return tokenized_seq

def tokenize_relation(relations):
	r2i = {
		'Cause-Effect(e1,e2)': 0, 'Cause-Effect(e2,e1)': 1,
		'Instrument-Agency(e1,e2)': 2, 'Instrument-Agency(e2,e1)': 3,
		'Product-Producer(e1,e2)': 4, 'Product-Producer(e2,e1)': 5,
		'Content-Container(e1,e2)': 6, 'Content-Container(e2,e1)': 7,
		'Entity-Origin(e1,e2)': 8, 'Entity-Origin(e2,e1)': 9,
		'Entity-Destination(e1,e2)': 10, 'Entity-Destination(e2,e1)': 11,
		'Component-Whole(e1,e2)': 12, 'Component-Whole(e2,e1)': 13,
		'Member-Collection(e1,e2)': 14, 'Member-Collection(e2,e1)': 15,
		'Message-Topic(e1,e2)': 16, 'Message-Topic(e2,e1)': 17,
		'Other': 18
	}
	result = np.zeros((len(relations), 19))
	for i, relation in enumerate(relations):
		result[i][r2i[relation]] = 1

	return result

def write_file(output_file, answers):
	i2r = {
		0: 'Cause-Effect(e1,e2)', 1: 'Cause-Effect(e2,e1)',
		2: 'Instrument-Agency(e1,e2)', 3: 'Instrument-Agency(e2,e1)',
		4: 'Product-Producer(e1,e2)', 5: 'Product-Producer(e2,e1)',
		6: 'Content-Container(e1,e2)', 7: 'Content-Container(e2,e1)',
		8: 'Entity-Origin(e1,e2)', 9: 'Entity-Origin(e2,e1)',
		10: 'Entity-Destination(e1,e2)', 11: 'Entity-Destination(e2,e1)',
		12: 'Component-Whole(e1,e2)', 13: 'Component-Whole(e2,e1)',
		14: 'Member-Collection(e1,e2)', 15: 'Member-Collection(e2,e1)',
		16: 'Message-Topic(e1,e2)', 17: 'Message-Topic(e2,e1)',
		18: 'Other'
	}
	with open(output_file, 'w') as f:
		for answer in answers:
			f.write('{}\t{}\n'.format(answer[0], i2r[answer[1]]))








