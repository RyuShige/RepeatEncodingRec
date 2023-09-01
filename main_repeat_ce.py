import os
import time
import torch
import argparse
import wandb
from tqdm import tqdm

from model_recbole import SASRec
from sasrec_repeat_emb import SASRec_RepeatEmb
from sasrec_repeat_emb_plus import SASRec_RepeatEmbPlus
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--project', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--split', default='ratio', type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)

    [session_set_train, session_set_valid, session_set_test, session_train, session_valid, session_test, repeat_train, repeat_valid, repeat_test, repeatnum, itemnum, sessionnum, sessionsetnum, sessionset_valid_min, sessionset_test_min] = dataset
    num_batch = len(session_set_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for ss in session_set_train:
        cc += len(session_set_train[ss])
    print(f'session set num {sessionsetnum}')
    print(f'session num {sessionnum}')
    print(f'item num {itemnum}')
    print(f'max repeat {repeatnum}')
    print('average sequence length: %.2f' % (cc / sessionsetnum))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(session_set_train, session_train, repeat_train, sessionsetnum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    if args.model == 'SASRec':
        model = SASRec(sessionsetnum, itemnum, args).to(args.device)
    elif args.model == 'SASRec_RepeatEmb':
        model = SASRec_RepeatEmb(sessionsetnum, itemnum, repeatnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    elif args.model == 'SASRec_RepeatEmbPlus':
        model = SASRec_RepeatEmbPlus(sessionsetnum, itemnum, repeatnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    # if args.state_dict_path is not None:
    #     try:
    #         model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
    #         tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
    #         epoch_start_idx = int(tail[:tail.find('.')]) + 1
    #     except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
    #         print('failed loading state_dicts, pls check file path: ', end="")
    #         print(args.state_dict_path)
    #         print('pdb enabled for your quick check, pls type exit() if you do not need it')
    #         import pdb; pdb.set_trace()
            
    
    # if args.inference_only:
    #     model.eval()
    #     t_test = evaluate(model, args.model, dataset, args, mode='test')
    #     print('test (Rcall@10: %.4f, MRR@10 %.4f, HR@10: %.4f)' % (t_test[0], t_test[1], t_test[2]))
    
    ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    # ce lossでやろうとしたけど失敗したのかな
    # bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()

    early_stop = -1
    early_count = 0
    best_epoch = 0
    loss = 0
    total_loss = 0
    epoch_loss = 0
    best_model_params = None
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        print('epoch: ', epoch)
        for step in tqdm(range(num_batch)): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            ss, seq, repeat, pos, neg = sampler.next_batch() # tuples to ndarray
            ss, seq, repeat, pos, neg = np.array(ss), np.array(seq), np.array(repeat), np.array(pos), np.array(neg)
            # u, seq, repeat, pos, neg = expand_samples(u, seq, repeat, pos, neg, args.maxlen)
            if args.model == 'SASRec':
                logits = model(ss, seq, pos, neg)
            elif args.model == 'SASRec_RepeatEmb':
                logits = model(ss, seq, repeat, pos, neg)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            # posをtensorに変換
            pos = torch.tensor(pos, dtype=torch.long).to(args.device)
            loss = ce_criterion(logits, pos)
            # loss += ce_criterion(neg_logits[indices], neg_labels[indices])
            # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            total_loss += loss.item()
        
        epoch_loss = loss / num_batch
        total_loss = 0 # for next epoch

    
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_valid = evaluate(model, args.model, dataset, args, mode='valid')
            
            # early stopping
            if early_stop < t_valid[3]:
                early_stop = t_valid[3] # MRR@20
                best_model_params = model.state_dict().copy()  # 最高のモデルのパラメータを一時的に保存
                best_epoch = epoch
                early_count = 0
            else:
                early_count += 1
            
            print('epoch:%d, time: %f(s), valid (Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, HR@10: %.4f, HR@20: %.4f))'
                    % (epoch, T, t_valid[0], t_valid[1], t_valid[2],  t_valid[3], t_valid[4], t_valid[5]))
    
            f.write(str(t_valid) + '\n')
            f.flush()            
            t0 = time.time()
            model.train()

        
        if early_count == 3:
            print('early stop at epoch {}'.format(epoch))
            print('testing')
            folder = args.dataset + '_' + args.train_dir
            if args.model == 'SASRec':
                fname = 'SASRec_BestModel.MRR={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(early_stop, best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            elif args.model == 'SASRec_RepeatEmb':
                fname = 'SASRec_RepeatEmb_BestModel.MRR={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(early_stop, best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(best_model_params, os.path.join(folder, fname))

            # 最も評価指標が高かったエポックのモデルのパスを指定します。
            best_model_path = os.path.join(folder, fname)

            # モデルの重みをロードします。
            model.load_state_dict(torch.load(best_model_path, map_location=torch.device(args.device)))

            # ロードした重みを用いてテストの評価を行います。
            t_test = evaluate(model, args.model, dataset, args, mode='test')
            print('best epoch:%d, time: %f(s), test (Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, HR@10: %.4f, HR@20: %.4f)'
                    % (best_epoch, T, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
            f.write(str(t_test) + '\n')
            f.flush()

            
            break
    
        if epoch == args.num_epochs:
            print('testing')
            folder = args.dataset + '_' + args.train_dir
            if args.model == 'SASRec':
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            elif args.model == 'SASRec_RepeatEmb':
                fname = 'SASRec_RepeatEmb.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

            # 最も評価指標が高かったエポックのモデルのパスを指定します。
            best_model_path = os.path.join(folder, fname)

            # モデルの重みをロードします。
            model.load_state_dict(torch.load(best_model_path, map_location=torch.device(args.device)))

            # ロードした重みを用いてテストの評価を行います。
            t_test = evaluate(model, args.model, dataset, args, mode='test')
            print('epoch:%d, time: %f(s), test (Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, HR@10: %.4f, HR@20: %.4f)'
                    % (epoch, T, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
            f.write(str(t_test) + '\n')
            f.flush()

    
    f.close()
    sampler.close()
    print("Done")
