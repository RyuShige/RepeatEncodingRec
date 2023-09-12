import numpy as np
import torch
from layer import TransformerEncoder

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

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.batch_size = args.batch_size

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) # +1いる？
        self.pos_emb = torch.nn.Embedding(self.maxlen, args.hidden_units) # TO IMPROVE

        self.inner_size = 256
        self.attn_dropout_prob = 0.5
        self.hidden_dropout_prob = 0.5
        self.hidden_act = "gelu"
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02
        
        self.trm_encoder = TransformerEncoder(
            n_layers=args.num_blocks,
            n_heads=args.num_heads,
            hidden_size=args.hidden_units,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
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

    def log2feats(self, log_seqs):
        # log_seqsにおいて、0以外の数を数えて、item_seq_lenに格納
        # item_seq_len = np.count_nonzero(log_seqs, axis=1)
        # self.batch_size個のnumpy配列を作成
        item_seq_len = np.full(log_seqs.shape[0], self.maxlen)
        # tenosrに変換
        item_seq_len = torch.LongTensor(item_seq_len).to(self.dev)

        log_seqs = torch.LongTensor(log_seqs).to(self.dev)
        seqs = self.item_emb(log_seqs)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.layernorm(seqs)
        seqs = self.emb_dropout(seqs)

        extended_attention_mask = self.get_attention_mask(log_seqs)

        trm_output = self.trm_encoder(
            seqs, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1) # 与えられたseqの最後のitemのembeddingのみを取得
        # 与えられたseq（0以外）全てのitemのembeddingを取得したい
        return output  # [B H] 与えられたseqの最後のitemのembedding、つまりその系列の最終的なコンテキスト

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        test_item_emb = self.item_emb.weight # モデルにおける各itemのembedding
        logits = torch.matmul(log_feats, test_item_emb.transpose(0, 1)) # 与えられた系列と各アイテムとの類似度
        return logits # [B, I(num_items)]

    # def predict(self, user_ids, log_seqs, item_indices): # for inference
    #     log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

    #     final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

    #     item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

    #     logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    #     # preds = self.pos_sigmoid(logits) # rank same item list for different users

        # return logits # preds # (U, I)

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # 最終層の最後のアイテムの出力 [1 B]

        test_item_emb = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # 指定したアイテムの学習済みembeddingを取得 [I B]
        logits = torch.matmul(log_feats, test_item_emb.transpose(0, 1)) # [1 B] * [B I] = [1 I]
        # logits = torch.matmul(test_item_emb, log_feats).sum(dim=1)

        return logits