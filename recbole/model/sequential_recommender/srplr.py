
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import RegLoss, BPRLoss
from recbole.sampler.sampler import AbstractSampler
import torch.nn.functional as F
import math
import random
import numpy as np

class SRPLR(SequentialRecommender):
    r"""
    """

    def __init__(self, config, dataset):
        super(SRPLR, self).__init__(config, dataset)

        # load parameters info
        self.base_model = config['base_model']
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.batch_size = config["train_batch_size"]
        self.USER_ID = config["USER_ID_FIELD"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.reg_weight = config["reg_weight"]
        self.bpr_weight = config['bpr_weight']
        self.tau = config['tau']

        # sasrec base
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.logic_CE = nn.CrossEntropyLoss()

        # GRU4Rec base
        self.embedding_size = config["embedding_size"]
        self.dropout_prob = config["dropout_prob"]
        self.num_layers = config["num_layers"]
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # cnn based
        self.n_h = config["nh"]
        self.n_v = config["nv"]
        self.n_users = dataset.user_num

        self.user_embedding = nn.Embedding(
            self.n_users, self.embedding_size, padding_idx=0
        )
        # vertical conv layer
        self.conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1)
        )
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.n_h,
                    kernel_size=(i, self.embedding_size),
                )
                for i in lengths
            ]
        )

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(
            self.embedding_size + self.embedding_size, self.embedding_size
        )

        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()
        self.reg_loss = RegLoss()

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # logic initialization
        self.Negation = Negation()
        self.intersection = Intersection(self.hidden_size, self.tau)
        self.gamma = nn.Parameter(torch.Tensor([config["gamma"]]), requires_grad=False)
        self.neg = "T"   # toy   caser
        self.mask = None
        self.sampler = LogicalSampler(dataset, distribution="uniform")    # distribution="uniform" or "popularity"

        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        self.projection_regularizer = Regularizer(0.05, 0.05, 1e9)  # 0.05 for other
        self.fea2log = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)
        self.neg_sam_num = config['neg_sam_num']

        # parameters initialization
        self.apply(self._init_weights)
        # self.epsilon = 2.0
        # self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_size]),
        #                                     requires_grad=False)
        # embedding_range = self.embedding_range.item()
        # self.modulus = nn.Parameter(torch.Tensor([0.5 * embedding_range]), requires_grad=True)
        # self.logic_embedding = nn.Embedding(
        #     self.n_items, self.hidden_size, padding_idx=0
        # )



    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_sasrec(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        self.mask = item_seq != 0

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def forward_gru(self, item_seq, item_seq_len):
        self.mask = item_seq != 0
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def forward_cnn(self, user, item_seq):
        # Embedding Look-up
        # use unsqueeze() to get a 4-D input for convolution layers. (batch_size * 1 * max_length * embedding_size)
        item_seq_emb = self.item_embedding(item_seq).unsqueeze(1)
        user_emb = self.user_embedding(user).squeeze(1)
        self.mask = item_seq != 0

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        seq_output = self.ac_fc(self.fc2(x))
        # the hidden_state of the predicted item, size:(batch_size * hidden_size)
        return seq_output

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

    def feature_to_beta(self, feature):

        logic_input = self.fea2log(feature)
        logic_input = self.projection_regularizer(logic_input)

        alpha, beta = torch.chunk(logic_input, 2, dim=-1)

        return alpha, beta

    def vec_to_dis(self, alpha, beta):

        dis = torch.distributions.beta.Beta(alpha, beta)

        return dis

    def distance(self, dis1, dis2):

        score = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(dis1, dis2), p=1, dim=-1)

        return score

    def calculate_loss(self, interaction):

        # feature-level output
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]

        # choose base model
        if self.base_model == 'SASRec':
            seq_output = self.forward_sasrec(item_seq, item_seq_len)
        elif self.base_model == 'GRU4Rec':
            seq_output = self.forward_gru(item_seq, item_seq_len)
        elif self.base_model == 'Caser':
            seq_output = self.forward_cnn(user, item_seq)
        else:
            raise NotImplementedError("check the name of base model")

        #  traget
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        neg_items_all = self.sampler.sample_neg_sequence(item_seq, self.neg_sam_num)

        #  feature2logic
        alpha_seq, beta_seq = self.feature_to_beta(self.item_embedding(item_seq))

        #  neg sample
        alpha_neg1, beta_neg1 = self.feature_to_beta(self.item_embedding(neg_items))

        alpha_neg2, beta_neg2 = self.feature_to_beta(self.item_embedding(neg_items_all))
        alpha_neg, beta_neg = self.Negation(alpha_neg2).view(alpha_seq.size(0), -1, self.hidden_size), \
                              self.Negation(beta_neg2).view(alpha_seq.size(0), -1, self.hidden_size)
        # logic input
        if self.neg == 'T':
            alpha_input, beta_input = torch.cat([alpha_neg, alpha_seq], dim=1), torch.cat([beta_neg, beta_seq], dim=1)
            self.mask = torch.cat([torch.ones(alpha_seq.size(0), alpha_neg.size(1)).cuda(), self.mask], dim=1).bool()
        else:
            alpha_input, beta_input = alpha_seq, beta_seq

        # logic output
        alpha_output, beta_output = self.logic_forward(alpha_input, beta_input)
        # output distribution
        out_dis = self.vec_to_dis(alpha_output, beta_output)
        alpha_pos, beta_pos = self.feature_to_beta(self.item_embedding(pos_items))
        pos_dis = self.vec_to_dis(alpha_pos, beta_pos)
        neg_dis = self.vec_to_dis(alpha_neg1, beta_neg1)

        logic_pos_score = self.distance(pos_dis, out_dis)
        logic_neg_score = self.distance(neg_dis, out_dis)
        logic_loss = self.loss_fct(logic_pos_score, logic_neg_score)
        # logic_loss = - logic_pos_score.mean()
        # print('logic_loss', logic_loss)

        #  prediction score
        test_item_emb = self.item_embedding.weight
        #  sample from Beta distribution
        logic_output = alpha_output / (alpha_output + beta_output)
        all_i_alpha, all_i_beta = self.feature_to_beta(test_item_emb)
        all_i_output = all_i_alpha / (all_i_alpha + all_i_beta)

        logits = torch.matmul(torch.cat([seq_output, logic_output], dim=1),
                              torch.cat([test_item_emb, all_i_output], dim=1).transpose(0, 1))

        loss1 = self.logic_CE(logits, pos_items)

        if self.base_model == 'Caser':
            reg_loss = self.reg_loss(
                [
                    self.user_embedding.weight,
                    self.item_embedding.weight,
                    self.conv_v.weight,
                    self.fc1.weight,
                    self.fc2.weight,
                    self.intersection.feature_layer_1.weight,
                    self.fea2log.weight
                ]
            )
        else:
            reg_loss = self.reg_loss(
                [
                    self.item_embedding.weight,
                    self.intersection.feature_layer_1.weight,
                    self.fea2log.weight
                ]
            )

        loss = loss1 + self.bpr_weight * logic_loss + self.reg_weight * reg_loss  # 0.5

        return loss

    def logic_forward(self, alphas, betas):

        alpha, beta = self.intersection(alphas, betas, self.mask)

        return alpha, beta

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]

        # choose base model
        if self.base_model == 'SASRec':
            seq_output = self.forward_sasrec(item_seq, item_seq_len)
        elif self.base_model == 'GRU4Rec':
            seq_output = self.forward_gru(item_seq, item_seq_len)
        elif self.base_model == 'Caser':
            seq_output = self.forward_cnn(user, item_seq)
        else:
            raise NotImplementedError("check the name of base model")

        alpha_seq, beta_seq = self.feature_to_beta(self.item_embedding(item_seq))
        alpha_output, beta_output = self.logic_forward(alpha_seq, beta_seq)
        logic_output = alpha_output / (alpha_output + beta_output)

        test_item_emb = self.item_embedding(test_item)
        all_i_alpha, all_i_beta = self.feature_to_beta(test_item_emb)
        all_i_output = all_i_alpha / (all_i_alpha + all_i_beta)

        scores = torch.matmul(torch.cat([seq_output, logic_output], dim=1),
                              torch.cat([test_item_emb, all_i_output], dim=1).transpose(0, 1))

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]

        # choose base model
        if self.base_model == 'SASRec':
            seq_output = self.forward_sasrec(item_seq, item_seq_len)
        elif self.base_model == 'GRU4Rec':
            seq_output = self.forward_gru(item_seq, item_seq_len)
        elif self.base_model == 'Caser':
            seq_output = self.forward_cnn(user, item_seq)
        else:
            raise NotImplementedError("check the name of base model")

        alpha_seq, beta_seq = self.feature_to_beta(self.item_embedding(item_seq))
        alpha_output, beta_output = self.logic_forward(alpha_seq, beta_seq)
        logic_output = alpha_output / (alpha_output + beta_output)

        test_items_emb = self.item_embedding.weight

        all_i_alpha, all_i_beta = self.feature_to_beta(test_items_emb)
        all_i_output = all_i_alpha / (all_i_alpha + all_i_beta)

        scores = torch.matmul(torch.cat([seq_output, logic_output], dim=1),
                              torch.cat([test_items_emb, all_i_output], dim=1).transpose(0, 1))

        # scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class Intersection(nn.Module):
    def __init__(self, dim, tau):
        super(Intersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.feature_layer_2 = nn.Linear(2 * self.dim, self.dim, bias=False)
        self.tau = tau
        # self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim, bias=False)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, alpha, beta, mask):
        # feature: N x B x d
        # logic:  N x B x d
        logits = torch.cat([alpha, beta], dim=-1)  # N x B x 2d
        # mask is needed
        mask = torch.where(mask, 0.0, -10000.0)
        # print(mask[0, 0:5])
        # att_input = self.feature_layer_2(F.relu(self.feature_layer_1(logits))) * mask.unsqueeze(2) * 0.05
        att_input = self.feature_layer_1(logits) * mask.unsqueeze(2) * self.tau
        # att_input = self.feature_layer_1(logits) * mask.unsqueeze(2) * 0.05
        # print('att_input', att_input[0, 0:5, 0])

        attention = F.softmax(att_input, dim=1)
        # print('att', attention[0, 0:5, 0])

        alpha = torch.sum(attention * alpha, dim=1)
        beta = torch.sum(attention * beta, dim=1)

        # alpha, beta = self.

        return alpha, beta


class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def neg_feature(self, feature):
        # f,f' in [-L, L]
        # f' = (f + 2L) % (2L) - L, where L=1
        feature = feature
        # indicator_positive = feature >= 0
        # indicator_negative = feature < 0
        # feature[indicator_positive] = feature[indicator_positive] - 1
        # feature[indicator_negative] = feature[indicator_negative] + 1
        return feature

    # def forward(self, feature, logic):
    #     feature = self.neg_feature(feature)
    #     logic = 1 - logic
    #     return feature, logic

    def forward(self, logic):
        logic = 1./logic
        return logic

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class LogicalSampler(AbstractSampler):
    def __init__(self, dataset, distribution, alpha=1.0):
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num
        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, seq_num):
        return np.random.randint(1, self.item_num, seq_num)

    def get_used_ids(self):
        pass

    def sample_neg_sequence(self, pos_sequence, sample_num):
        """For each moment, sampling 'sample_num' item from all the items except the one the user clicked on at that moment.

        Args:
            pos_sequence (torch.Tensor):  all users' item history sequence, with the shape of `(N, )`.

        Returns:
            torch.tensor : all users' negative item history sequence.

        """
        total_num = len(pos_sequence)
        value_ids = []
        for i in range(sample_num):
            check_list = np.arange(total_num)
            tem_ids = np.zeros(total_num, dtype=np.int64)
            while len(check_list) > 0:
                tem_ids[check_list] = self._uni_sampling(len(check_list))
                check_index = np.where(tem_ids[check_list] == pos_sequence[check_list])
                check_list = check_list[check_index]
            pos_sequence = torch.cat([torch.tensor(tem_ids.reshape(total_num, 1)).cuda(), pos_sequence], dim=-1)
            value_ids.append(tem_ids)

        value_ids = torch.tensor(np.array(value_ids)).t().cuda()
        return value_ids
