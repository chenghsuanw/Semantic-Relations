from gensim.models import Word2Vec
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
import re


def parse_train_data(path):
    e1d = defaultdict(set)
    e2d = defaultdict(set)
    sentences = list()
    Y = list()
    for line in open(path):
        if 'Comment:' not in line:
            if '<e1>' in line:
                sentence = line.strip().split('"', 1)[1][:-1]
                tok = re.split(r'<e1>|</e1>|<e2>|</e2>', sentence)
                e1 = tok[1]
                e2 = tok[3]
                sentence = ''.join(w.lower() for w in tok)
                sentence = re.sub(r'[^\w\s]', '', sentence)
                wset = set([w.lower() for w in sentence[:-1].split(' ')])
                e1d[e1] |= wset
                e2d[e2] |= wset
                sentences.append(sentence)
            elif len(line) > 1:
                Y.append(line.strip())
    return sentences, Y


def parse_test_data(path):
    sentences = list()
    for line in open(path):
        sentence = line.strip().split('"', 1)[1][:-1]
        tok = re.split(r'<e1>|</e1>|<e2>|</e2>', sentence)
        e1 = tok[1]
        e2 = tok[3]
        sentence = ''.join(w.lower() for w in tok)
        sentence = re.sub(r'[^\w\s]', '', sentence)
        sentences.append(sentence)
    return sentences


def ave_wv(s, w2v):
    sv = np.zeros(50)
    for w in s.split(' '):
        sv += w2v.wv[w]
    sv /= len(s)
    return sv


def w2vsim(s, ts, w2v):
    swv = ave_wv(s, w2v)
    tswv = ave_wv(ts, w2v)
    return sum(np.absolute(tswv - swv))


def matchingsim(s, ts):
    c = 0
    for w in s:
        if w in ts:
            c += 1/len(ts)
    return c


def main(args):
    train_sentences, Y = parse_train_data(args.train_path)
    test_sentences = parse_test_data(args.test_path)

    w2v = Word2Vec([s.split(' ') for s in train_sentences + test_sentences], size=50, window=5, min_count=0, workers=4)
    w2v.save('w2vmodel')
    '''
    w2v = Word2Vec.load('w2vmodel')
    '''
    fp = open('result.txt', 'w')
    for tidx, s in enumerate(test_sentences):
        maxsim = float('-inf')
        maxidx = 0
        for idx, ts in enumerate(train_sentences):
            #sim = w2vsim(s, ts, w2v)
            sim = matchingsim(s, ts)
            if sim > maxsim:
                maxsim = sim
                maxidx = idx
        print(maxsim, maxidx, Y[maxidx])
        fp.write(f'{tidx+8001}\t{Y[maxidx]}\n') 


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--test_path')
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    main(parse_args())
