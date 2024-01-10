'''
lightSANsのコードを参考にするために、recboleのSASRecの実装+BCEが上手くいくか確認するためのコード
'''

import numpy as np
import torch
from layer import LightRepeatTransformerEncoder
from layer import RepeatTransformerEncoder

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class LightSANs_Repeat(torch.nn.Module):
    def __init__(self, user_num, item_num, repeat_num, args):
        super(LightSANs_Repeat, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.repeat_num = repeat_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.batch_size = args.batch_size
        self.re_enc = args.re_enc
        self.k_interests = 5 # recboleの初期値


        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) # +1いる？
        self.pos_emb = torch.nn.Embedding(self.maxlen, args.hidden_units) # TO IMPROVE
        self.repeat_emb = torch.nn.Embedding(self.repeat_num+1, args.hidden_units, padding_idx=0) # TO IMPROVE

        self.inner_size = args.inner_size
        self.attn_dropout_prob = 0.5
        self.hidden_dropout_prob = 0.5
        self.hidden_act = "gelu"
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02
        
        self.trm_encoder = LightRepeatTransformerEncoder(
            n_layers=args.num_blocks,
            n_heads=args.num_heads,
            k_interests=self.k_interests,
            hidden_size=args.hidden_units,
            seq_len=args.maxlen,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        # self.trm_encoder = RepeatTransformerEncoder(
        #     n_layers=args.num_blocks,
        #     n_heads=args.num_heads,
        #     k_interests=self.k_interests,
        #     hidden_size=args.hidden_units,
        #     seq_len=args.maxlen,
        #     inner_size=self.inner_size,
        #     hidden_dropout_prob=self.hidden_dropout_prob,
        #     attn_dropout_prob=self.attn_dropout_prob,
        #     hidden_act=self.hidden_act,
        #     layer_norm_eps=self.layer_norm_eps,
        # )

        self.layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def embedding_layer(self, item_seq, repeat_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_embedding = self.pos_emb(position_ids)
        position_embedding = position_embedding.unsqueeze(0).repeat(item_seq.size(0), 1, 1)
        repeat_embedding = self.repeat_emb(repeat_seq)
        item_emb = self.item_emb(item_seq)
        return item_emb, position_embedding, repeat_embedding
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def repetitive_encoding(self, max_len, repeat, d_model, pred=False):
        if pred:
            re = torch.zeros(1, max_len, d_model).to(self.dev)
            rep = torch.LongTensor(repeat).to(self.dev).unsqueeze(-1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).repeat(1, 1).unsqueeze(1).to(self.dev)
            re[:, :, 0::2] = torch.sin(rep * div_term)
            re[:, :, 1::2] = torch.cos(rep * div_term)
        else:
            re = torch.zeros(self.batch_size, max_len, d_model).to(self.dev)
            rep = torch.LongTensor(repeat).to(self.dev).unsqueeze(-1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).repeat(self.batch_size, 1).unsqueeze(1).to(self.dev)
            re[:, :, 0::2] = torch.sin(rep * div_term)
            re[:, :, 1::2] = torch.cos(rep * div_term)
        return re

    def log2feats(self, log_seqs, log_reps, pred=False):
        log_seqs = torch.LongTensor(log_seqs).to(self.dev)
        log_reps = torch.LongTensor(log_reps).to(self.dev)
        item_emb, position_embedding, repeat_embedding = self.embedding_layer(log_seqs, log_reps)

        # repeatとposを足してdpeに入れるパターン
        # repeat_position_emb = repeat_embedding + position_embedding
        # trm_output = self.trm_encoder(
        #     item_emb, repeat_position_emb, output_all_encoded_layers=True
        # )
                
        # repeat+itemのパターン
        # repeat_item_emb = repeat_embedding + item_emb
        # item_emb += position_embedding
        # trm_output = self.trm_encoder(
        #     item_emb, repeat_item_emb, output_all_encoded_layers=True
        # )

        # item+pos, repeatのパターン
        # item_emb += position_embedding
        # item_emb = self.layernorm(item_emb)
        # item_emb = self.emb_dropout(item_emb)

        # trm_output = self.trm_encoder(
        #     item_emb, repeat_embedding, output_all_encoded_layers=True
        # )

        # item,rep+pos。vはitemのみ-ReSANs
        # repeat_position_emb = repeat_embedding + position_embedding
        # trm_output = self.trm_encoder(
        #     item_emb, repeat_position_emb, output_all_encoded_layers=True
        # )

        # item+pos,rep+pos。vはitemのみ-ReSANs-abl_1
        # repeat_position_emb = repeat_embedding + position_embedding
        # item_position_emb = repeat_embedding + position_embedding
        # trm_output = self.trm_encoder(
        #     item_position_emb, repeat_position_emb, output_all_encoded_layers=True
        # )

        # item, pos。vはitemのみ-ReSANs-abl_2
        # trm_output = self.trm_encoder(
        #     item_emb, position_embedding, output_all_encoded_layers=True
        # )
        
        # item+rep, pos。vはitem+rep-ReSANs-abl_3
        # item_repeat_emb = item_emb + repeat_embedding
        # trm_output = self.trm_encoder(
        #     item_repeat_emb, position_embedding, output_all_encoded_layers=True
        # )

        # item+rep, pos+rep。vはitem+rep-ReSANs-abl_4
        # item_repeat_emb = item_emb + repeat_embedding
        # position_repeat_emb = position_embedding + repeat_embedding
        # trm_output = self.trm_encoder(
        #     item_repeat_emb, position_repeat_emb, output_all_encoded_layers=True
        # )

        # item, rep。vはitem+rep-ReSANs-abl_5
        # position_repeat_emb = position_embedding + repeat_embedding
        # trm_output = self.trm_encoder(
        #     item_emb, repeat_embedding, output_all_encoded_layers=True
        # )

        # item+pos, rep。vはitem+rep-ReSANs-abl_7
        item_position_emb = item_emb + position_embedding
        trm_output = self.trm_encoder(
            item_position_emb, repeat_embedding, output_all_encoded_layers=True
        )

        # item, rep, pos。vはitemのみ-separateとVadd
        # trm_output = self.trm_encoder(
        #     item_emb, position_embedding, repeat_embedding, output_all_encoded_layers=True
        # )


        output = trm_output[-1]
        return output  # [B I H]

    def forward(self, user_ids, log_seqs, log_reps, pos_seqs, neg_seqs): # for training
        log_feats = self.log2feats(log_seqs, log_reps) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # [B, I, H]
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1) # ある時刻のtransforerの出力とある時刻の正解アイテムのembeddingの内積（d次元のベクトル同士の内積）
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits # [B, I(num_items)]
    
    def predict(self, user_ids, log_seqs, log_reps, item_indices): # for inference
        log_feats = self.log2feats(log_seqs, log_reps) # [B Seq H]
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste # [1 H] 最後のアイテムの出力だけを取ってくる
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # [I H]
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) # [1 I]
        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)