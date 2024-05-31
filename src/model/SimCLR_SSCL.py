# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import network as models
#import model.network as models

class SimClr_SSCL(nn.Module):
    def __init__(self, args,device):

        super(SimClr_SSCL, self).__init__()

        # self.momentum = momentum

        # Load model
        self.encoder_q = getattr(models, args.model)(
            args, num_classes=128)  # Query Encoder           特征维度 128

        # Add the mlp head
        self.encoder_q.fc = projection_MLP(args)
        self.batch_size = args.batch_size
        self.n_hard = args.s
        self.s1_hard = args.k
        self.temperature = args.temperature

    def forward(self, x_q, x_k,args,device):

        batch_size = x_q.size(0)

        # Feature representations of the query view from the query encoder
        feat_q = self.encoder_q(x_q)
        feat_k = self.encoder_q(x_k)

        out1 = F.normalize(feat_q, dim=1)
        out2 = F.normalize(feat_k, dim=1)

        #mix+hcl
        if args.method == 'simclr_sscl' :
            loss = self.simclr_sscl(out1, out2, args, device, args.estimator)

        # Compute the logits for the InfoNCE contrastive loss.
        #loss = NT_XentLoss(feat_q, feat_k)
        # Update the queue/memory with the current key_encoder minibatch.
        return loss

    def find_hard_negatives(self, logits):
        """Finds the top n_hard hardest negatives in the queue for the query.

        Args:
            logits (torch.tensor)[batch_size, len_queue]: Output dot product negative logits.(512*512)

        Returns:
            torch.tensor[batch_size, n_hard]]: Indices in the queue.
        """
        # logits -> [batch_size, len_queue]
        _, idxs_hard = torch.topk(
            logits.clone().detach(), k=self.n_hard, dim=-1, sorted=False)
        # idxs_hard -> [batch_size, n_hard]

        _, idxs_easy= torch.topk(
            logits.clone().detach(), k=self.n_hard, dim=-1, sorted=False,largest=False)

        random_idxs = torch.randperm(512)
        selected_idxs = random_idxs[:self.n_hard]
        selected_idxs_matrix = selected_idxs.unsqueeze(0).expand(512, -1)

        return idxs_hard,idxs_easy,selected_idxs_matrix

    def hard_negatives1(self, out_1, out_2, logits, idxs_hard,idxs_easy,idxs_random):
        """Concats type 1 hard negatives to logits.

        Args:
            out_q (torch.tensor)[batch_size, d_out]: Output of query encoder.
            logits (torch.tensor)[batch_size, len_queue + ...]: Output dot product logits.
            idxs_hard (torch.tensor)[batch_size, n_hard]: Indices of hardest negatives
                in the queue for each query.

        Returns:
            (torch.tensor)[batch_size, len_queue + ... + s1_hard]: logits concatenated with
                type 1 hard negatives.
        """

        batch_size, device = out_1.shape[0], out_1.device

        out = torch.cat([out_1, out_2], dim=0)

        idxs1, idxs2 = torch.randint(
            0, self.n_hard, size=(2, 2 * self.batch_size, self.s1_hard), device=device)


        alpha = torch.rand(size=(2 * self.batch_size, self.s1_hard, 1), device=device)

        neg1_hard = out[
            torch.gather(idxs_hard, dim=1, index=idxs1)].clone().detach()
        neg2_hard = out[
            torch.gather(idxs_hard, dim=1, index=idxs2)].clone().detach()



        #正常混合
        # neg_hard = alpha * neg1_hard + (1 - alpha) * neg2_hard

        #topk+lastk
        # neg2_hard = out[
        #     torch.gather(idxs_easy, dim=1, index=idxs2)].clone().detach()

        #随机+随机
        # idxs_random=idxs_random.to(device)
        # neg1_hard = out[
        #     torch.gather(idxs_random, dim=1, index=idxs1)].clone().detach()
        # neg2_hard = out[
        #     torch.gather(idxs_random, dim=1, index=idxs2)].clone().detach()

        # 维度混合
        # alpha = torch.rand(size=(2 *self.batch_size, self.s1_hard,128), device=device)

        # 查询样本混合
        # neg_hard = alpha * out.clone().detach()[:, None] +(1 - alpha) * neg1_hard



        neg_hard = neg1_hard

        neg_hard = F.normalize(neg_hard, dim=-1).detach()

        logits_hard = torch.einsum(
            'b d, b s d -> b s', out, neg_hard)
        logits_hard = torch.exp(logits_hard /self.temperature)

        logits = torch.cat([logits, logits_hard], dim=1)
        # logits -> [batch_size, len_queue + ... + s1_hard]
        return logits


    def cdf_trans(self, tau_plus, alpha, p):
        a = (1 - 2 * tau_plus) * (2 - 4 * alpha - 1e-3) / 2
        b = 2 * alpha * (1 - tau_plus) + tau_plus * (2 - 2 * alpha + 1e-8)
        c = -p
        cdf = (-b + torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return cdf

    def simclr_sscl(self, out_1, out_2, args, device,estimator):
        out = torch.cat([out_1, out_2], dim=0)  # dim=0     按列cat

        neg = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
        old_neg = neg.clone()

        neg_find = change_negative(old_neg,args.batch_size).to(device)                       # 正样本设为0，防止find hard negative选到对应正样本
        idxs_hard,idxs_easy,selected_idxs_matrix = self.find_hard_negatives(neg_find)


        mask = get_negative_mask(args.batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * args.batch_size, -1)

        neg2 = self.hard_negatives1(
            out_1=out_1, out_2=out_2, logits=neg,idxs_hard=idxs_hard,idxs_easy=idxs_easy,idxs_random=selected_idxs_matrix)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
        pos = torch.cat([pos, pos], dim=0)                 #



        N = args.batch_size * 2 - 2 + self.s1_hard
        imp = (args.beta * neg2.log()).exp()
        reweight_neg = (imp * neg2).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-args.tau_plus * N * pos + reweight_neg) / (1 - args.tau_plus)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / args.temperature))  # 有一个下限值


        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()

        return loss




class projection_MLP(nn.Module):
    def __init__(self, args):
        '''Projection head for the pretraining of the resnet encoder.

            - Uses the dataset and model size to determine encoder output
                representation dimension.

            - Outputs to a dimension of 128, and uses non-linear activation
                as described in SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf
        '''
        super(projection_MLP, self).__init__()

        if args.model == 'resnet18' or args.model == 'resnet34':
            n_channels = 512
        elif args.model == 'resnet50' or args.model == 'resnet101' or args.model == 'resnet152':
            n_channels = 2048
        else:
            raise NotImplementedError('model not supported: {}'.format(args.model))

        self.projection_head = nn.Sequential()

        self.projection_head.add_module('W1', nn.Linear(
            n_channels, n_channels))
        # change
        # self.projection_head.add_module('BN', nn.BatchNorm1d(n_channels))
        self.projection_head.add_module('ReLU', nn.ReLU())
        self.projection_head.add_module('W2', nn.Linear(
            n_channels, 128))

    def forward(self, x):
        return self.projection_head(x)

def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)


def change_negative(negative,batch_size):
    #negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative[i, i] = 0
        negative[i, i + batch_size] = 0
        negative[i + batch_size, i ] = 0
        negative[i + batch_size, i + batch_size] = 0
    return negative

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask




