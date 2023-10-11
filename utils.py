import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
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
        while len(session_set_train[sessionset]) <= 1: sessionset = np.random.randint(1, sessionsetnum + 1) # 学習データに存在しないsessionsetの場合はやり直し

        seq = np.zeros([maxlen], dtype=np.int32)
        rep = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = session_set_train[sessionset][-1]
        # pos = nxt
        idx = maxlen - 1

        ts = set(session_set_train[sessionset])
        for i, r in zip(reversed(session_set_train[sessionset][:-1]), reversed(repeat_train[sessionset][:-1])):
            seq[idx] = i
            rep[idx] = r
            pos[idx] = nxt # 複数のままにして、lossを単一アイテム同士から複数アイテム同士の計算にする手もある
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
def data_partition(fname, data_type):
    usernum = 0
    itemnum = 0
    repeatnum = 0
    sessionnum = 0
    sessionsetnum = 0
    sessionset_valid_min = np.inf
    sessionset_test_min = np.inf
    U_valid = defaultdict(list)
    U_test = defaultdict(list)
    Session_set_train = defaultdict(list)
    Session_set_valid = defaultdict(list)
    Session_set_test = defaultdict(list)
    Session_train = defaultdict(list)
    Session_valid = defaultdict(list)
    Session_test = defaultdict(list)
    Repeat_train = defaultdict(list)
    Repeat_valid = defaultdict(list)
    Repeat_test = defaultdict(list)
    # assume user/item index starting from 1
    # データセット全体
    f = open('data/%s/%s.txt'% (data_type, fname), 'r') # ここが事前にt,v,tをまとめたもの
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
    f = open('data/%s/%s_train.txt' % (data_type, fname), 'r')
    for line in f:
        u, i, t, r, s, ss = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = float(t)
        r = int(r)
        s = int(s)
        ss = int(ss)
        Session_set_train[ss].append(i)
        Session_train[ss].append(s)
        Repeat_train[ss].append(r)
    f = open('data/%s/%s_valid.txt' % (data_type, fname), 'r')
    for line in f:
        u, i, t, r, s, ss = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = float(t)
        r = int(r)
        s = int(s)
        ss = int(ss)
        U_valid[ss].append(u)
        Session_set_valid[ss].append(i)
        Session_valid[ss].append(s)
        Repeat_valid[ss].append(r)
        sessionset_valid_min = min(ss, sessionset_valid_min)
    f = open('data/%s/%s_test.txt' % (data_type, fname), 'r')
    for line in f:
        u, i, t, r, s, ss = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = float(t)
        r = int(r)
        s = int(s)
        ss = int(ss)
        U_test[ss].append(u)
        Session_set_test[ss].append(i)
        Session_test[ss].append(s)
        Repeat_test[ss].append(r)
        sessionset_test_min = min(ss, sessionset_test_min)

    return [U_valid, U_test, Session_set_train, Session_set_valid, Session_set_test, Session_train, Session_valid, Session_test, Repeat_train, Repeat_valid, Repeat_test, repeatnum, itemnum, sessionnum, sessionsetnum, sessionset_valid_min, sessionset_test_min]

# evaluate
def evaluate(model, model_name, dataset, args, mode, repeat_data=None):
    assert mode in {'valid', 'test'}, "mode must be either 'valid' or 'test'"
    [u_valid, u_test, session_set_train, session_set_valid, session_set_test, session_train, session_valid, session_test, repeat_train, repeat_valid, repeat_test, repeatnum, itemnum, sessionnum, sessionsetnum, sessionset_valid_min, sessionset_test_min] = copy.deepcopy(dataset)
    R_PRECITION = 0.0
    R_PRECITION_REP = 0.0
    NEXT_HR = 0.0
    PRECITION_10 = 0.0
    PRECITION_20 = 0.0
    RECALL_10 = 0.0
    RECALL_20 = 0.0
    MRR_10 = 0.0
    MRR_20 = 0.0
    NDCG_10 = 0.0
    NDCG_20 = 0.0
    HT_10 = 0.0
    HT_20 = 0.0
    valid_user = 0.0
    valid_item = 0.0
    valid_item_rep = 0.0

    session_sets = list(session_set_valid.keys()) if mode == 'valid' else list(session_set_test.keys())
    for ss in tqdm(session_sets):
        # print('session_set_test[ss]', session_set_test[ss])
        if (mode == 'valid' and len(session_set_valid[ss]) < 2) or (mode == 'test' and len(session_set_test[ss]) < 2): continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        s = np.zeros([args.maxlen], dtype=np.int32)
        rep = np.zeros([args.maxlen], dtype=np.int32)
        u = np.zeros([1], dtype=np.int32)

        # seqとitem_idxを作成
        idx = args.maxlen - 1
        if mode == 'valid':
            u = u_valid[ss] # ユーザを読み込む
            u = min(u)
            s = session_valid[ss] # session set中のsession idを読み込む
            # 最も大きいsession idをもつindexを取得
            max_s = np.argmax(s) # このindexまでがseq(session setの最後のセッションの最初の曲)
            input_seq = session_set_valid[ss][:max_s+1] # seqにはsession setの最後のセッションの最初の曲までを入れる
            input_rep = repeat_valid[ss][:max_s+1]
            for i, r in zip(reversed(input_seq), reversed(input_rep)):
                seq[idx] = i
                rep[idx] = r
                idx -= 1
                if idx == 0: break
            item_idx = session_set_valid[ss][max_s+1:]
            rep_idx = np.array(repeat_valid[ss][max_s+1:])
            # rep_idxが2以上のindexを取得
            rep_idx = np.where(rep_idx > 1)[0]
        elif mode == 'test':
            u = u_test[ss]
            u = min(u)
            s = session_test[ss]
            max_s = np.argmax(s)
            input_seq = session_set_test[ss][:max_s+1]
            input_rep = repeat_test[ss][:max_s+1]
            for i, r in zip(reversed(input_seq), reversed(input_rep)):
                seq[idx] = i
                rep[idx] = r
                idx -= 1
                if idx == 0: break
            item_idx = session_set_test[ss][max_s+1:]
            rep_idx = np.array(repeat_test[ss][max_s+1:])
            # rep_idxが2以上のindexを取得
            rep_idx = np.where(rep_idx > 1)[0]

        correct_len = len(item_idx)
        if correct_len == 0: continue
        correct_len_rep = len(rep_idx)
        
        # item_indexが20以下の場合、最後の予測の2位以下のアイテムで補完する（予測アイテム内の繰り返しは排除）
        if args.search and mode == 'test':
            # itemnum個の配列を作成(全アイテム)
            items = np.arange(1, itemnum + 1)
            max_items = []
            
            for i in range(len(item_idx)):
                if model_name == 'SASRec':
                    predictions = -model.predict(*[np.array(l) for l in [[ss], [seq], items]]) # -をつけることでargsortを降順にできる（本来は昇順）# searchの場合はitemsは全てのアイテム
                elif model_name == 'SASRec_Repeat' or model_name=='SASRec_RepeatPlus' or model_name=='SASRec_Repeat_Out' or model_name=='LightSANs' or model_name=='LightSANs_Repeat':
                    predictions = -model.predict(*[np.array(l) for l in [[ss], [seq], [rep], items]])
                predictions = predictions[0].tolist()
                # トップアイテムを取得
                top_items = np.argsort(predictions) # 0始まり
                top_items = top_items + 1 # 1始まりに変換
                
                # 重複がないように予測を格納
                # top_itemsからmax_itemsの重複を除く
                unique_top_items = [item for item in top_items if item not in max_items]
                # max_itemsにunique_top_itemsを追加
                max_items.append(unique_top_items[0])

                if (len(max_items) == correct_len) and correct_len < 20:
                    for c in range(1, 20-correct_len+1):
                        max_items.append(unique_top_items[c])
                # seqにtop_items[0]（予測トップ1）を追加
                seq = np.append(seq, top_items[0])
                # uに基づいてrepにtop_items[0]の繰り返し回数を追加
                # repeat_dataを参照してu, top_items[0]の繰り返し回数を取得
                repeat_values = repeat_data[(repeat_data['u'] == u) & (repeat_data['i'] == (top_items[0]))]['r'].values
                top_item_repeat = repeat_values[0] if len(repeat_values) > 0 else 0
                rep = np.append(rep, top_item_repeat+1)
                # seqの最初の要素を削除
                seq = seq[1:]
                # repの最初の要素を削除
                rep = rep[1:]
            # max_itemsを利用してranksを作成
            ranks = np.zeros([correct_len], dtype=np.int32)
            ranks.fill(100)
            for cnt, item in enumerate(item_idx):
                for j, m in enumerate(max_items):
                    if item == m:
                        ranks[cnt] = j
                        break
        else:
            # itemnum個の配列を作成(全アイテム)
            t = np.arange(1, itemnum + 1)
            # item_idxに含まれないアイテム
            t = np.setdiff1d(t, item_idx)  
            item_idx.extend(t)

            if model_name == 'SASRec':
                predictions = -model.predict(*[np.array(l) for l in [[ss], [seq], item_idx]]) # -をつけることでargsortを降順にできる（本来は昇順）
            elif model_name == 'SASRec_Repeat' or model_name=='SASRec_RepeatPlus' or model_name=='LightSANs' or args.model == 'LightSANs_simple_R' or model_name=='LightSANs_Repeat':
                predictions = -model.predict(*[np.array(l) for l in [[ss], [seq], [rep], item_idx]])
            predictions = predictions[0]  # - for 1st argsort DESC
            ranks = predictions.argsort().argsort()[0:correct_len].tolist() # 正解データのランクを取得
        # ranksからrep_idxに対応するアイテムだけを取得
        ranks_rep = np.array(ranks)[rep_idx]

        valid_user += 1
        valid_item += correct_len
        valid_item_rep += correct_len_rep

        # R-Precision
        c = 0
        for i, r in enumerate(ranks):
            if r < correct_len:
                # R_PRECITION += 1 # 全アイテム数で割る場合だが、有効シークエンス数で割ってもいいと思う（recallはそうだし、分母もシークエンス毎に異なる）
                c += 1
        R_PRECITION += c / correct_len
        
        # R-Precision-Repeat
        c = 0
        for i, r in enumerate(ranks_rep):
            if r < correct_len_rep:
                R_PRECITION_REP += 1
                # c += 1
        # R_PRECITION_REP += c / correct_len_rep

        # Next HR
        c = 0
        for i, r in enumerate(ranks):
            if r == i:
                # NEXT_HR += 1
                c += 1
        NEXT_HR += c / correct_len

        # 10未満のアイテム数をカウント
        c = 0
        top = 20
        dcg = 0
        h = 0
        for i, r in enumerate(ranks):
            if r < 10: # この数字はtopkによる
                c += 1
                h = 1
                if r == 0:
                    dcg += 1
                else:
                    dcg += 1 / np.log2(r + 1)
                if r < top:
                    top = r
        PRECITION_10 += c / 10
        RECALL_10 += c / correct_len
        HT_10 += h
        if top < 20:
            MRR_10 += 1.0 / (top + 1)
        dcg_p = 1
        for i in range(correct_len):
            if i == 0:
                continue
            dcg_p += 1 / np.log2(i + 1)
        NDCG_10 += dcg / dcg_p

        
        # 20未満のアイテム数をカウント
        c = 0
        top = 20
        h = 0
        for i, r in enumerate(ranks):
            if r < 20: # この数字はtopkによる
                c += 1
                h = 1
                if r == 0:
                    dcg += 1
                else:
                    dcg += 1 / np.log2(r + 1)
                if r < top:
                    top = r

        PRECITION_20 += c / 20
        RECALL_20 += c / correct_len
        HT_20 += h
        if top < 20:
            MRR_20 += 1.0 / (top + 1)
        dcg_p = 1
        for i in range(correct_len):
            if i == 0:
                continue
            dcg_p += 1 / np.log2(i + 1)
        NDCG_20 += dcg / dcg_p

        # if rank < 10:
        #     # RECALL += 
        #     MRR += 1.0 / (rank + 1)
        #     NDCG += 1 / np.log2(rank + 2)
        #     HT += 1
        # if valid_user % 100 == 0:
        #     print('.', end="")
        #     sys.stdout.flush()


    # return PRECITION_10 / valid_user, PRECITION_20 / valid_user, RECALL_10 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, MRR_20 / valid_user, NDCG_10 / valid_user, NDCG_20 / valid_user, HT_10 / valid_user, HT_20 / valid_user
    return R_PRECITION / valid_user, R_PRECITION_REP / valid_item_rep, NEXT_HR / valid_user, RECALL_10 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, MRR_20 / valid_user, NDCG_10 / valid_user, NDCG_20 / valid_user
