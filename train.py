import os
import time
import torch
import random
from model import *
from utils import *


def train(data_file, args):
    random.seed(20230501)
    np.random.seed(20230501)
    torch.manual_seed(20230501)
    torch.cuda.manual_seed_all(20230501)

    data_file = os.path.join(data_file, args.dataset+'.txt')
    dataset = data_partition(data_file)

    [dict_data, user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    McMat = get_McMat(dict_data, itemnum).to(args.device)
    AdjMat = sparse_mat(McMat, layer=args.graph_layer, q=0.1).to(args.device)
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = MC_RGN(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.train()

    epoch_start_idx = 1

    if args.loss == 'BCE':
        loss_f = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'BPR':
        loss_f = BPRloss()
    elif args.loss == 'TOP1':
        loss_f = TOP1loss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            indices = np.where(pos != 0)
            pos_logits, neg_logits = model(u, seq, pos, neg, AdjMat, McMat, indices)
            adam_optimizer.zero_grad()
            if args.loss == 'BCE':
                loss = loss_f(pos_logits, torch.ones_like(pos_logits, device=args.device))
                loss += loss_f(neg_logits, torch.zeros_like(neg_logits, device=args.device))
            else:
                loss = loss_f(pos_logits, neg_logits)
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step % 50 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, McMat, AdjMat, args)
            t_valid = evaluate_valid(model, dataset, McMat, AdjMat, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                  % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            t0 = time.time()
            model.train()

    sampler.close()
    print("Done")
    return t_test[0], t_test[1]