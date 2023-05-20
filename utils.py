import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
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

def data_partition(data_file):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    f = open(data_file, 'r')
    for line in f:
        u, i = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [User, user_train, user_valid, user_test, usernum, itemnum]  # 字典形式的数据

def get_McMat(dict_data, n_item):
    McMat = np.zeros((n_item + 1, n_item + 1))
    for seq in dict_data.values():
        for i, j in zip(seq[:-2][:-1], seq[:-2][1:]):
            McMat[i, j] += 1
    McMat = torch.from_numpy(McMat)
    print('MCMat got!')
    mask = ~McMat.bool()
    McMat = ((torch.log(McMat)).pow(2) / 2).exp()
    McMat = torch.masked_fill(McMat, mask, 0.0)
    return McMat * torch.pow(torch.tensor(torch.pi), -0.5)

def sparse_mat(mat, layer=1, q=0.1, normal='random_walk'):
    if mat.count_nonzero() / (mat.size()[0] * mat.size()[1]) <= q:
        mat = mat.bool()
    else:
        threshold = torch.quantile(mat, (1 - q))
        mask = (mat >= threshold).to(mat.device)
        mat = mat.bool() * mask
    if normal == 'random_walk':
        mat = mat / mat.sum(dim=1).view(-1, 1)
        mat[torch.isinf(mat)] = 0
        mat[torch.isnan(mat)] = 0
    idx = torch.nonzero(mat).t().to(mat.device)
    mat = torch.sparse_coo_tensor(idx, mat[idx[0], idx[1]], mat.size())
    if layer > 1:
        for _ in range(layer):
            mat = torch.sparse.mm(mat, mat.to_dense())
            idx = torch.nonzero(mat).t().to(mat.device)
            mat = torch.sparse_coo_tensor(idx, mat[idx[0], idx[1]], mat.size())
    return mat

class BPRloss(torch.nn.Module):
    def __init__(self, type='mean'):
        super(BPRloss, self).__init__()
        self.type = type

    def forward(self, pos_logit, neg_logit):
        return (-1.0) * F.logsigmoid(pos_logit - neg_logit).mean()

class TOP1loss(torch.nn.Module):
    def __init__(self, type='mean'):
        super(TOP1loss, self).__init__()
        self.type = type

    def forward(self, pos_logit, neg_logit):
        return torch.sigmoid(neg_logit - pos_logit).mean() + torch.sigmoid(torch.pow(neg_logit, 2)).mean()

def evaluate(model, dataset, McMat, adj, args):
    [_, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = list(range(1, itemnum+1))
        item_idx.remove(test[u][0])
        item_idx = [test[u][0]] + item_idx

        predictions = model.predict(np.array([u]), np.array([seq]), np.array(item_idx), McMat, adj)
        predictions = -predictions
        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 500 == 0:
            print('.', end="")
            sys.stdout.flush()
    return NDCG / valid_user, HT / valid_user

def evaluate_valid(model, dataset, McMat, adj, args):
    [_, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = model.predict(np.array([u]), np.array([seq]), np.array(item_idx), McMat, adj)
        predictions = -predictions
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 500 == 0:
            print('.', end="")
            sys.stdout.flush()
    return NDCG / valid_user, HT / valid_user