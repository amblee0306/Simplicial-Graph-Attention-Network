'''
# the features of movie/IMDB
- 0-4660 is movies
- 4661 - 6930 are directors
- 6931 - 12771 are actors
'''
import pickle
import pickle5
import torch
import os.path
import networkx as nx
import dgl
from collections import defaultdict
from utils import add_non_target_node_with_onehot_edge_ft
import numpy as np

DATA_ROOT_DIRECTORY = '../data/'


def sort_key(key):
    return key[0], key[2], key[1]


def form_one_simplices(adj_list_2_hop, e11):
    one_simplices_to_shared_node_dict = defaultdict(list)
    one_simplices_edgelist = set()
    connected_nodes = set()  # to ensure all vertices are in the contructed simplicial complex
    adj_list_2_hop_new = set()

    for e in adj_list_2_hop:
        s, d, m = e
        edge = str(s) + " " + str(d)  # this is the constructed one simplex "edge"
        one_simplices_to_shared_node_dict[edge].append(m[0])

    for k, list_of_shared_nodes in one_simplices_to_shared_node_dict.items():
        if len(list_of_shared_nodes) >= e11:
            s, d = k.split()
            one_simplices_edgelist.add(k)
            connected_nodes.add(int(s))
            connected_nodes.add(int(d))
            for shared_node in list_of_shared_nodes:
                adj_list_2_hop_new.add(tuple([int(s), int(d), shared_node]))
                
    return adj_list_2_hop_new, connected_nodes, one_simplices_edgelist, one_simplices_to_shared_node_dict


def create_adjacency_list_representation(dataset, A, num_of_nodes):
    overall_A = A[:,:,0]+A[:,:,1]+A[:,:,2]+A[:,:,3]
    adj_list = [[] for i in range(num_of_nodes)]

    for i in range(num_of_nodes):
        for j in range(i, num_of_nodes):
            if float(overall_A[i][j]) > 0 and (i!=j):
                adj_list[i].append(j)
                adj_list[j].append(i)

    with open(DATA_ROOT_DIRECTORY + 'processed_data/{}/na_adj_list.pickle'.format(dataset), 'wb') as handle:
        pickle.dump(adj_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return adj_list


def load_adj_list(dataset, A, num_of_nodes):
    adj_list_filename = DATA_ROOT_DIRECTORY + 'processed_data/{}/na_adj_list.pickle'.format(dataset)
    if os.path.isfile(adj_list_filename):
        print("File exist")
        with open(adj_list_filename, 'rb') as handle:
            adj_list = pickle5.load(handle)
    else:
        adj_list = create_adjacency_list_representation(dataset, A, num_of_nodes)
    return adj_list


def form_graph(connected_nodes_set, target_ntype_last_index, one_simplices_edgelist_set, one_simplices_to_shared_node_dict):
    # form graph (simplicial complex dimension 1) using constructed one-simplices
    all_movie_id = set([i for i in range(target_ntype_last_index+1)])
    missing_movie_nodes = all_movie_id - connected_nodes_set

    for n in missing_movie_nodes:
        edge = str(n) + " " + str(n)  # add self loop when it is disconnected
        one_simplices_edgelist_set.add(edge)
        one_simplices_to_shared_node_dict[edge].append(n)

    one_simplices_edgelist = list(one_simplices_edgelist_set)
    nxg = nx.parse_edgelist(one_simplices_edgelist, nodetype=int, create_using=nx.DiGraph())
    g = dgl.from_networkx(nxg)
    return g
            

def find_k_hop_between_two_target_nodes_dfs(k_hop, n_hop, start_node, nodeid, path, target_ntype_last_index, adj_list):
    def _find_k_hop_between_two_target_nodes_dfs(k_hop, n_hop, start_node, nodeid, path, adj_list_2_hop):
        # path is a list of node id
        # get and return all instances of two target nodes sharing one non-target node using DFS.
        if k_hop == n_hop:
            if nodeid <= target_ntype_last_index and start_node < nodeid:
                middle_feature = path[1:-1].copy()
                adj_list_2_hop.append((start_node, nodeid, middle_feature))
            return
        for next_node_id in adj_list[nodeid]:
            if next_node_id in path:
                # to remove the case where any of the nodes is revisited
                continue
            path.append(next_node_id)
            _find_k_hop_between_two_target_nodes_dfs(k_hop + 1, n_hop, start_node, next_node_id, path, adj_list_2_hop)
            path.pop()
    adj_list_2_hop = []
    _find_k_hop_between_two_target_nodes_dfs(k_hop, n_hop, start_node, nodeid, path, adj_list_2_hop)
    return adj_list_2_hop


class DataSplit:
    def __init__(self, train_node, train_target, valid_node, valid_target, test_node, test_target):
        self.train_node = train_node
        self.train_target = train_target
        self.valid_node = valid_node
        self.valid_target = valid_target
        self.test_node = test_node
        self.test_target = test_target


def load_data(UNINFORMATIVE, target_ntype_last_index, dataset, e11=1):
    with open(DATA_ROOT_DIRECTORY + dataset+'/node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)
    with open(DATA_ROOT_DIRECTORY + dataset+'/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(DATA_ROOT_DIRECTORY + dataset+'/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
        labels = [list(pairs) for pairs in labels]
    num_nodes = edges[0].shape[0]

    # different type of edges have different adjacency matrix
    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

    # whether to utilise uninformative node features
    if UNINFORMATIVE:
        print("Un-informative node features used (Random features)")
        node_features = np.random.randn(num_nodes, node_features.shape[1])

    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    num_of_nodes = node_features.size()[0]  # total number of nodes
    total_labelled_nodes = len(labels[0]) + len(labels[1]) + len(labels[2])
    
    train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.LongTensor)
    print('-------------------------------------')
    print('train_node.size()[0]', train_node.size()[0])
    print('valid_node.size()[0]', valid_node.size()[0])
    print('test_node.size()[0]', test_node.size()[0])
    print('total_labelled_nodes', total_labelled_nodes)
    print('-------------------------------------')
        
    num_classes = torch.max(train_target).item()+1

    adj_list = load_adj_list(dataset, A, num_of_nodes)

    adj_list_2_hop = []  # [(4632, 4636, [8855]), (4632, 4636, [10745])]
    for start_node in range(target_ntype_last_index+1):
        adj_list_2_hop.extend(find_k_hop_between_two_target_nodes_dfs(0, 2, start_node, start_node, [start_node], target_ntype_last_index, adj_list))
    
    adj_list_2_hop_new, connected_nodes, one_simplices_edgelist, one_simplices_to_shared_node_dict = form_one_simplices(adj_list_2_hop, e11)
    
    adj_list_2_hop_new = sorted(adj_list_2_hop_new, key=sort_key)
    adj_list_2_hop_new = [list(l) for l in adj_list_2_hop_new]
    adj_list_2_hop_new = [(s, d, [m]) for (s,d,m) in adj_list_2_hop_new]

    # the g formed is the simplicial complex of dimension 1 (for one hop simplicial complex)
    g = form_graph(connected_nodes, target_ntype_last_index, one_simplices_edgelist, one_simplices_to_shared_node_dict)
    # TODO: fix later (future me)
    one_simplices_to_shared_node_dict = [one_simplices_to_shared_node_dict]

    # constructs edge features for each of the edge in the original graph
    # input edge to obtain edge feature
    edgelist_preprocess_dic_ori_graph = add_non_target_node_with_onehot_edge_ft(A, node_features)

    data_split_obj = DataSplit(train_node, train_target, valid_node, valid_target, test_node, test_target)
    return adj_list_2_hop_new, [g], node_features, labels, num_classes, \
           data_split_obj, one_simplices_to_shared_node_dict, edgelist_preprocess_dic_ori_graph

"""
# can extract into multiple functions for loading data
yyy, zzz = load_data_base(xxx)
two_hop, g = load_data_2_hop(yyy, zzz)
four_hop, g = load_data_4_hop(yyy, zzz)

all = [two_hop, four_hop]
"""
