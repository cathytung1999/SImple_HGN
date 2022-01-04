import sys
sys.path.append('../../')
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data, feature_preprocess
from sklearn.metrics import f1_score
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from models.GNN import myGAT, GAT, GCN


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return prediction, micro_f1, macro_f1

def evaluate(args, model, act, features, e_feat, labels, index, loss_func):
    model.eval()
    with torch.no_grad():
        if args.model == 'mygat':
            logits = model(features, e_feat)
        else:
            logits = model(features)
    loss = loss_func(act(logits[index]), labels[index])
    prediction, micro_f1, macro_f1 = score(logits[index], labels[index])

    return prediction, loss, micro_f1, macro_f1

def run_model_DBLP(args):
    #feats_type = args.feats_type
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    g, features_list, in_dims, labels, train_idx, val_idx, test_idx,\
    e_feat, dl = feature_preprocess(args.dataset, args.feats_type, device)
#    print(g.num_nodes)
#    print(labels.shape[0])
#    print(dl.labels_train['count'])
    for _ in range(args.repeat):
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        if args.model == 'gat':
            net = GAT(g, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, \
                  args.dropout, args.slope, True)
        elif args.model == 'mygat':
            net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes,\
                  args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        elif args.model == 'gcn':
            net = GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
        else:
            raise Exception('{} model is not defined!'.format(args.model))
        
        #print(net)
        net.to(device)
        
        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, 
            save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        loss_func = F.nll_loss
        logsm = nn.LogSoftmax(dim=1)
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()

            if args.model == 'mygat':
                logits = net(features_list, e_feat)
            else:
                logits = net(features_list)
            
            train_loss = loss_func(logsm(logits[train_idx]), labels[train_idx])
            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            _, train_micro_f1, _ = score(logits[train_idx], labels[train_idx])
            # validation
            _, val_loss, val_micro_f1, val_macro_f1 = evaluate(args, net, logsm, features_list, e_feat, labels, val_idx, loss_func)
            early_stop = early_stopping.step(val_loss.data.item(), val_micro_f1, net)
            t_end = time.time()
            # print model info
            print('Epoch {:05d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Val Loss {:.4f} | '
              'Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Time(s) {:.4f}'.format(
            epoch + 1, train_loss.item(), train_micro_f1, val_loss.item(), \
            val_micro_f1, val_macro_f1, t_end - t_start))

            # early stopping
            if early_stop:
                print('Early stopping!\n\n')
                break
        
        # best performance validation
        model = early_stopping.load_checkpoint(net)
        _, best_loss, best_micro_f1, best_macro_f1 = evaluate(args, net, logsm, features_list, e_feat, labels, val_idx, loss_func)
        print('Besst Performance: Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
        best_loss, best_micro_f1, best_macro_f1))


        # testing with evaluate_results_nc
        pred, _, _, _ = evaluate(args, net, logsm, features_list, e_feat, labels, test_idx, loss_func)
        dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_path=f"{args.dataset}_{args.run}.txt")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used.' +
                         '0 - loaded features;' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2; ' +
                         '4 - only term features (id vec for others);' + 
                         '5 - only term features (zero vec for others).')
    ap.add_argument('--model', type=str)
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=500, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=50, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=1)

    args = ap.parse_args()
    print(args)

    run_model_DBLP(args)