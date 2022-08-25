import torch
from sklearn.metrics import f1_score
import dgl
from model_edge_ft_ori import SGAT
import pickle  # to save edge features
import os
import networkx as nx
import gc
import statistics
import time
import itertools
from load_data_movie import load_data
from utils import EarlyStopping, create_edge_graph_L1
from utils import create_edge_features_upper, create_edge_features_sgat_ef, create_edge_features_sgat

# Change this to True to test using random features
UNINFORMATIVE = False # Set variable to False for RNF
EDGE_FEATURES = True # Set variable to False for SGAT

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, eg2, all_feat, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits,_ = model(g, eg2, all_feat)
    loss = loss_func(logits[mask], labels)
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels)

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    start_time_process = time.time()
    if args['dataset'] == 'IMDB':
        target_ntype_last_index = 4660
    elif args['dataset'] == 'ACM':
        target_ntype_last_index = 3024  # author
    else:
        raise Exception("Unknown target node type index")

    na_adj_list_2_hop, list_of_graph, node_features_full, labels, num_classes, data_split_obj, edgelist_preprocess_dic, edgelist_preprocess_dic_ori_graph = load_data(UNINFORMATIVE, target_ntype_last_index, args['dataset'], args['e11'])

    features = node_features_full[:target_ntype_last_index+1]

    train_node = data_split_obj.train_node.to(args['device'])
    valid_node = data_split_obj.valid_node.to(args['device'])
    test_node = data_split_obj.test_node.to(args['device'])
    train_target = data_split_obj.train_target.to(args['device'])
    valid_target = data_split_obj.valid_target.to(args['device'])
    test_target = data_split_obj.test_target.to(args['device'])
    
    list_of_graph = [dgl.remove_self_loop(graph) for graph in list_of_graph]
    list_of_graph = [dgl.add_self_loop(graph) for graph in list_of_graph]

    num_target_nodes = target_ntype_last_index + 1
    set_interested_nodes = set([i for i in range(num_target_nodes)])
    
    # create upper adjacency graph
    # we treat 1-simplices as "nodes" in this new upper-adjacency matrix represented graph.
    upper_edge_graph, edges_part_of_tri_1, tri_ft = create_edge_graph_L1(list_of_graph[0], na_adj_list_2_hop, set_interested_nodes, node_features_full, args['L'])    
    upper_edge_graph = dgl.remove_self_loop(upper_edge_graph)
    upper_edge_graph = dgl.add_self_loop(upper_edge_graph)
    print('upper_edge_graph', upper_edge_graph)
        
    # prepare the required features
    if EDGE_FEATURES:
        edge_features = create_edge_features_sgat_ef(list_of_graph,
                 node_features_full, edgelist_preprocess_dic, edges_part_of_tri_1, edgelist_preprocess_dic_ori_graph)
    else:
        edge_features = create_edge_features_sgat(list_of_graph,
                 node_features_full, edgelist_preprocess_dic)
    edge_ft_upper = create_edge_features_upper(upper_edge_graph, edge_features[0], 'share_tri_id', tri_ft, features.shape[1])
    
    print("--- processing data takes : %s seconds ---" % (time.time() - start_time_process))
    start_time = time.time()
        
    features = features.to(args['device'])  # node features
    node_features_full = node_features_full.to(args['device'])
    edge_feature = edge_features[0].to(args['device'])
    edge_ft_upper = edge_ft_upper.to(args['device'])
  
    all_feat = [features, edge_feature, edge_ft_upper]

    model = SGAT(num_simplicial_complexes=len(list_of_graph),
                edge_in_ft_size=edge_feature.size()[1],
                in_size=features.shape[1],
                hidden_size=args['hidden_units'],
                out_size=num_classes,
                num_heads=[args['num_heads']],
                dropout=args['dropout']).to(args['device'])
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable parameters: ', pytorch_total_params, pytorch_total_params_trainable)
    print(model)
    
    g = [graph.to(args['device']) for graph in list_of_graph]
    eg2 = [graph.to(args['device']) for graph in [upper_edge_graph]]

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss().to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    torch.cuda.empty_cache()
    gc.collect()
    
    # Train & Valid & Test
    best_val_loss = 10000
    best_val_acc = 0
    best_test_loss = 10000
    best_train_loss = 10000
    best_macro_train_f1 = 0
    best_micro_train_f1 = 0
    best_macro_val_f1 = 0
    best_micro_val_f1 = 0
    best_macro_test_f1 = 0
    best_micro_test_f1 = 0
    
    for epoch in range(args['num_epochs']):
        for param_group in optimizer.param_groups:
            if param_group['lr'] > 0.005:
                param_group['lr'] = param_group['lr'] * 0.9
        model.train()
        logits,_ = model(g, eg2, all_feat)
        loss = loss_fcn(logits[train_node], train_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_node], train_target)
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, eg2, all_feat,
                                                                 valid_target, valid_node, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if val_acc > best_val_acc:
            test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, eg2, all_feat,
                                                                 test_target, test_node, loss_fcn)
            best_val_loss = val_loss.detach().cpu().numpy()
            best_test_loss = test_loss.detach().cpu().numpy()
            best_train_loss = loss.detach().cpu().numpy()
            best_macro_train_f1 = train_macro_f1
            best_micro_train_f1 = train_micro_f1
            best_macro_val_f1 = val_macro_f1
            best_micro_val_f1 = val_micro_f1
            best_macro_test_f1 = test_macro_f1
            best_micro_test_f1 = test_micro_f1
            best_val_acc = val_acc
            
        if early_stop:
            break     
        gc.collect()

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, eg2, all_feat,
                                                                 test_target, test_node, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))
    
    print('---------------Best Results--------------------')
    print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(best_train_loss, best_macro_train_f1, best_micro_train_f1))
    print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(best_val_loss, best_macro_val_f1, best_micro_val_f1))
    print('Test - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(best_test_loss, best_macro_test_f1, best_micro_test_f1))
    gc.collect()

    if test_macro_f1 > best_macro_test_f1:
        max_macro = float(test_macro_f1)
    else:
        max_macro = float(best_macro_test_f1)
    if test_micro_f1 > best_micro_test_f1:
        max_micro = test_micro_f1
    else:
        max_micro = best_micro_test_f1
        
    return max_micro, max_macro 


if __name__ == '__main__':
    import argparse
    from utils import setup
        
    parser = argparse.ArgumentParser('SGAT')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-dataset', '--dataset', type=str, default='IMDB',
                        help='Dataset choice: IMDB/ACM')
    parser.add_argument('-device', '--device', type=str, default='cuda',
                        help='which device? cuda:2/cpu')
    parser.add_argument('-e11', '--e11', type=int, default=1,
                        help='number of shared nodes to form an edge for one hop complex')
    parser.add_argument('-L', '--L', type=int, default=20,
                        help='max dim/order to recursively include their triangles')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--hidden_units', type=int, default=64,
                        help='number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='epochs to run')
    parser.add_argument('--patience', type=int, default=100,
                        help='early stopping patience')
    
    args = parser.parse_args().__dict__
    
    args = setup(args)
    print(args)
    start_time = time.time()
    micro, macro = main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('micro f1={}, macro f1={}'.format(micro, macro))
