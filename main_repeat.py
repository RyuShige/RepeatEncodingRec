import os
import time
import torch
import argparse
import wandb
from tqdm import tqdm

from model import SASRec
from sasrec_repeat import SASRec_Repeat
from sasrec_repeat_plus import SASRec_RepeatPlus
from sasrec_repeat_out import SASRec_Repeat_Out
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
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--inner_size', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--split', default='ratio', type=str)
parser.add_argument('--re_enc', default=False, type=str2bool)
parser.add_argument('--po_enc', default=False, type=str2bool)
parser.add_argument('--wandb', default=False, type=str2bool)
parser.add_argument('--data_type', default='lifetime', type=str)
parser.add_argument('--ffn', default=False, type=str2bool)
parser.add_argument('--search', default=False, type=str2bool)
parser.add_argument('--scale', default=True, type=str2bool)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if args.wandb:
    wandb.init(
        project=f"{args.project}",
        name=f"{args.model}_{args.data_type}_repenc:{args.re_enc}_posenc:{args.po_enc}_{args.name}", 
        config={
            'dataset': args.dataset,
            'model': args.model,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'maxlen': args.maxlen,
            'hidden_units': args.hidden_units,
            'num_blocks': args.num_blocks,
            'num_epochs': args.num_epochs,
            'num_heads': args.num_heads,
            'dropout_rate': args.dropout_rate,
            'l2_emb': args.l2_emb,
            'device': args.device,
            'inference_only': args.inference_only,
            'state_dict_path': args.state_dict_path,
            'split': args.split,
            'repeatitive_encoding': args.re_enc,
            'positional_encoding': args.po_enc,
            'data_type': args.data_type
        }
        )

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset, args.data_type)

    # valid_repeatのデータベース
    repeat_data = pd.read_csv(f'data/{args.data_type}/user_repeat_valid.csv')

    [u_valid, u_test, session_set_train, session_set_valid, session_set_test, session_train, session_valid, session_test, repeat_train, repeat_valid, repeat_test, repeatnum, itemnum, sessionnum, sessionsetnum, sessionset_valid_min, sessionset_test_min] = dataset
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
    elif args.model == 'SASRec_Repeat':
        model = SASRec_Repeat(sessionsetnum, itemnum, repeatnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    elif args.model == 'SASRec_RepeatPlus':
        model = SASRec_RepeatPlus(sessionsetnum, itemnum, repeatnum, args).to(args.device)
    elif args.model == 'SASRec_Repeat_Out':
        model = SASRec_Repeat_Out(sessionsetnum, itemnum, repeatnum, args).to(args.device)
    
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
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    # ce lossでやろうとしたけど失敗したのかな
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
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
                pos_logits, neg_logits = model(ss, seq, pos, neg)
            elif args.model == 'SASRec_Repeat' or args.model == 'SASRec_RepeatPlus' or args.model == 'SASRec_Repeat_Out':
                pos_logits, neg_logits = model(ss, seq, repeat, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            total_loss += loss.item()
        
        epoch_loss = loss / num_batch
        if args.wandb:
            wandb.log({"epoch": epoch, "loss": epoch_loss})
        total_loss = 0 # for next epoch

    
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_valid = evaluate(model, args.model, dataset, args, mode='valid', repeat_data=repeat_data)
            
            # early stopping
            if early_stop < t_valid[4]:
                early_stop = t_valid[4] # MRR@20
                best_model_params = model.state_dict().copy()  # 最高のモデルのパラメータを一時的に保存
                best_epoch = epoch
                early_count = 0
            else:
                early_count += 1
            
            print('epoch:%d, time: %f(s), valid (R-Precision: %.4f, Next-HR: %4f, Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, NDCG@10: %.4f, NDCG@20: %.4f))'
                    % (epoch, T, t_valid[0], t_valid[1], t_valid[2],  t_valid[3], t_valid[4], t_valid[5], t_valid[6], t_valid[7]))
    
            f.write(str(t_valid) + '\n')
            f.flush()            
            t0 = time.time()
            model.train()
        
            if args.wandb:
                wandb.log({"epoch": epoch, "time": T, "valid_R-Precision": t_valid[0], "valid_Next-HR:": t_valid[1], "valid_Rcall@10": t_valid[2], "valid_Rcall@20": t_valid[3], "valid_MRR@10": t_valid[4], "valid_MRR@20": t_valid[5], "valid_NDCG@10": t_valid[6], "valid_NDCG@20": t_valid[7]})
            
        
        if early_count == 3:
            print('early stop at epoch {}'.format(epoch))
            print('testing')
            folder = args.dataset + '_' + args.train_dir
            if args.model == 'SASRec':
                fname = 'SASRec_BestModel.MRR={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(early_stop, best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            elif args.model == 'SASRec_Repeat':
                fname = 'SASRec_Repeat_BestModel.MRR={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(early_stop, best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            elif args.model == 'SASRec_RepeatPlus':
                fname = 'SASRec_RepeatPlus_BestModel.MRR={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(early_stop, best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            elif args.model == 'SASRec_Repeat_Out':
                fname = 'SASRec_Repeat_Out_BestModel.MRR={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(early_stop, best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(best_model_params, os.path.join(folder, fname))

            # 最も評価指標が高かったエポックのモデルのパスを指定します。
            best_model_path = os.path.join(folder, fname)

            # モデルの重みをロードします。
            model.load_state_dict(torch.load(best_model_path, map_location=torch.device(args.device)))

            # ロードした重みを用いてテストの評価を行います。
            repeat_data = pd.read_csv(f'data/{args.data_type}/user_repeat_test.csv')
            t_test = evaluate(model, args.model, dataset, args, mode='test', repeat_data=repeat_data)
            print('epoch:%d, time: %f(s), test (R-Precision: %.4f, Next-HR: %4f, Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, NDCG@10: %.4f, NDCG@20: %.4f))'
                    % (epoch, T, t_test[0], t_test[1], t_test[2],  t_test[3], t_test[4], t_test[5], t_test[6], t_test[7]))
            f.write(str(t_test) + '\n')
            f.flush()
        
            if args.wandb:
                wandb.log({"best_epoch": best_epoch, "time": T, "test_R-Precision": t_test[0], "test_Next-HR": t_test[1], "test_Rcall@10": t_test[2], "test_Rcall@20": t_test[3], "test_MRR@10": t_test[4], "test_MRR@20": t_test[5], "test_NDCG@10": t_test[6], "test_NDCG@20": t_test[7]})

            
            break
    
        if epoch == args.num_epochs:
            print('testing')
            folder = args.dataset + '_' + args.train_dir
            if args.model == 'SASRec':
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            elif args.model == 'SASRec_Repeat':
                fname = 'SASRec_Repeat.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            elif args.model == 'SASRec_RepeatPlus':
                fname = 'SASRec_RepeatPlus.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            elif args.model == 'SASRec_Repeat_Out':
                fname = 'SASRec_Repeat_Out.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(best_model_params, os.path.join(folder, fname))

            # 最も評価指標が高かったエポックのモデルのパスを指定します。
            best_model_path = os.path.join(folder, fname)

            # モデルの重みをロードします。
            model.load_state_dict(torch.load(best_model_path, map_location=torch.device(args.device)))

            # ロードした重みを用いてテストの評価を行います。
            repeat_data = pd.read_csv(f'data/{args.data_type}/user_repeat_test.csv')
            t_test = evaluate(model, args.model, dataset, args, mode='test', repeat_data=repeat_data)
            print('epoch:%d, time: %f(s), test (R-Precision: %.4f, Next-HR: %4f, Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, HR@10: %.4f, HR@20: %.4f))'
                    % (epoch, T, t_test[0], t_test[1], t_test[2],  t_test[3], t_test[4], t_test[5], t_test[6], t_test[7]))

            f.write(str(t_test) + '\n')
            f.flush()

            if args.wandb:
                wandb.log({"best_epoch": best_epoch, "time": T, "test_R-Precision": t_test[0], "test_Next-HR": t_test[1], "test_Rcall@10": t_test[2], "test_Rcall@20": t_test[3], "test_MRR@10": t_test[4], "test_MRR@20": t_test[5], "test_NDCG@10": t_test[6], "test_NDCG@20": t_test[7]})


    
    f.close()
    sampler.close()
    if args.wandb:
        wandb.finish()
    print("Done")
