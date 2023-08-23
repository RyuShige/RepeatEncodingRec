import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(session_set_train, session_train, repeat_train, sessionsetnum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        sessionset = np.random.randint(1, sessionsetnum + 1)
        while len(session_set_train[sessionset]) <= 1: sessionset = np.random.randint(1, sessionsetnum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        rep = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = session_set_train[sessionset][-1]
        idx = maxlen - 1

        ts = set(session_set_train[sessionset])
        for i, r in zip(reversed(session_set_train[sessionset][:-1]), reversed(repeat_train[sessionset][:-1])):
            seq[idx] = i
            rep[idx] = r
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (sessionset, seq, rep, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, SessionSet, Session, Repeat, sessionsetnum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(SessionSet,
                                                      Session,
                                                      Repeat,
                                                      sessionsetnum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# 

def expand_samples(u, seq, rep, pos, neg, max_seq):
    num_users, seq_len = seq.shape
    
    # 出力用の空のリスト
    expanded_u = []
    expanded_seq = []
    expanded_rep = []
    expanded_pos = []
    expanded_neg = []

    for i in range(num_users):
        for j in range(1, seq_len + 1):
            expanded_u.append(u[i])
            # 0でパディング
            padded_seq = np.pad(seq[i, :j], (max_seq - j, 0), 'constant')
            panded_rep = np.pad(rep[i, :j], (max_seq - j, 0), 'constant')
            padded_pos = np.pad(pos[i, :j], (max_seq - j, 0), 'constant')
            padded_neg = np.pad(neg[i, :j], (max_seq - j, 0), 'constant')

            expanded_seq.append(padded_seq)
            expanded_rep.append(panded_rep)
            expanded_pos.append(padded_pos)
            expanded_neg.append(padded_neg)
    # 最後にndarrayに変換
    return np.array(expanded_u), np.array(expanded_seq), np.array(expanded_rep), np.array(expanded_pos), np.array(expanded_neg)


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    repeatnum = 0
    sessionnum = 0
    sessionsetnum = 0
    Session_set_train = defaultdict(list)
    Session_set_valid = defaultdict(list)
    Session_set_test = defaultdict(list)
    Session_train = defaultdict(list)
    Session_valid = defaultdict(list)
    Session_test = defaultdict(list)
    Repeat_train = defaultdict(list)
    Repeat_valid = defaultdict(list)
    Repeat_test = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    repeat_train = {}
    repeat_valid = {}
    repeat_test = {}
    session_set_train = {}
    session_set_valid = {}
    session_set_test = {}
    session_train = {}
    session_valid = {}
    session_test = {}
    # assume user/item index starting from 1
    # データセット全体
    f = open('data/%s.txt' % fname, 'r') # ここが事前にt,v,tをまとめたもの
    for line in f:
        u, i, t, r, s, ss = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = float(t)
        r = int(r)
        s = int(s)
        ss = int(ss)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        repeatnum = max(r, repeatnum)
        sessionnum = max(s, sessionnum)
        sessionsetnum = max(ss, sessionsetnum)
    # train/valid/test
    f = open('data/%s_train.txt' % fname, 'r')
    for line in f:
        Session_set_train[ss].append(i)
        Session_train[ss].append(s)
        Repeat_train[ss].append(r)
    f = open('data/%s_valid.txt' % fname, 'r')
    for line in f:
        Session_set_valid[ss].append(i)
        Session_valid[ss].append(s)
        Repeat_valid[ss].append(r)
    f = open('data/%s_test.txt' % fname, 'r')
    for line in f:
        Session_set_test[ss].append(i)
        Session_test[ss].append(s)
        Repeat_test[ss].append(r)

    return [Session_set_train, Session_set_valid, Session_set_test, Session_train, Session_valid, Session_test, Repeat_train, Repeat_valid, Repeat_test, repeatnum, itemnum, sessionnum, sessionsetnum]

# evaluate
def evaluate(model, model_name, dataset, args, mode):
    assert mode in {'valid', 'test'}, "mode must be either 'valid' or 'test'"
    [user_train, user_valid, user_test, repeat_train, repeat_valid, repeat_test, usernum, repeatnum, itemnum] = copy.deepcopy(dataset)
    RECALL_10 = 0.0
    RECALL_20 = 0.0
    MRR_10 = 0.0
    MRR_20 = 0.0
    NDCG_10 = 0.0
    NDCG_20 = 0.0
    HT_10 = 0.0
    HT_20 = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in tqdm(users):

        if len(user_train[u]) < 1 or len(user_valid[u]) < 1 or len(user_test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        rep = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if mode == 'test':
            for i, r in zip(reversed(user_valid[u]), reversed(repeat_valid[u])):
                seq[idx] = i
                rep[idx] = r
                idx -= 1
                if idx == 0: break
        for i, r in zip(reversed(user_train[u]), reversed(repeat_train[u])):
            if idx == 0: break
            seq[idx] = i
            rep[idx] = r
            idx -= 1
            if idx == -1: break
        if mode == 'valid':
            item_idx = user_valid[u]
        elif mode == 'test':
            item_idx = user_test[u]
        
        correct_len = len(item_idx)

        # ランダムに選んだ100個のアイテムと正解データをモデルがどのように予測するか（全アイテムでやると時間がかかりすぎるため、一度やってみてもいいが）
        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t == 0: t = np.random.randint(1, itemnum + 1) # item_id=0は存在しない（パディング）のでやり直し
        #     item_idx.append(t)
        # itemnum個の配列を作成
        t = np.arange(1, itemnum + 1)
        # item_idxに含まれないアイテムを取得
        t = np.setdiff1d(t, item_idx) 
        item_idx.extend(t)

        if model_name == 'SASRec':
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        elif model_name == 'SASRec_RepeatEmb':
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [rep], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        ranks = predictions.argsort().argsort()[0:correct_len].tolist() # 正解データのランクを取得

        valid_user += 1

        # 10未満のアイテム数をカウント
        c = 0
        top = 20
        h = 0
        for r in ranks:
            if r < 10: # この数字はtopkによる
                c += 1
                h = 1
                if r < top:
                    top = r
        
        RECALL_10 += c / correct_len
        HT_10 += h
        if top < 20:
            MRR_10 += 1.0 / (top + 1)
        
        # 20未満のアイテム数をカウント
        c = 0
        top = 20
        h = 0
        for r in ranks:
            if r < 20: # この数字はtopkによる
                c += 1
                h = 1
                if r < top:
                    top = r
        
        RECALL_20 += c / correct_len
        HT_20 += h
        if top < 20:
            MRR_20 += 1.0 / (top + 1)

        # if rank < 10:
        #     # RECALL += 
        #     MRR += 1.0 / (rank + 1)
        #     NDCG += 1 / np.log2(rank + 2)
        #     HT += 1
        # if valid_user % 100 == 0:
        #     print('.', end="")
        #     sys.stdout.flush()

    return RECALL_10 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, MRR_20 / valid_user, HT_10 / valid_user, HT_20 / valid_user
