import datetime
import dgl
import networkx as nx
import errno
import numpy as np
import os
import pickle
import random
import torch
from scipy.sparse import coo_matrix
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import gudhi as gd
from collections import defaultdict


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=False):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


def setup(args):
    set_random_seed(args['seed'])
    args['log_dir'] = setup_log_dir(args)
    return args


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))


# find filled triangles and higher order simplices
def find_filled_tri(na_adj_list_2_hop, set_interested_nodes, L=10):
    tri_set = set()
    filled_tri = []
    filled_tri_shared_node = []
    previous_m = None
    max_simplex_order = L
    max_dim = 0
    for i, (s, d, m) in enumerate(na_adj_list_2_hop):
        if (s in set_interested_nodes) and (d in set_interested_nodes):
            if previous_m != m[0]:
                # note we are taking all simplices that are order 3
                if 3 <= len(tri_set) <= max_simplex_order:
                    filled_tri.append(tri_set)
                    filled_tri_shared_node.append(previous_m)
                if len(tri_set) > max_dim:
                    max_dim = len(tri_set)
                previous_m = m[0]
                tri_set = set()
            tri_set.add(s)
            tri_set.add(d)
        else:
            pass

    if 3 <= len(tri_set) <= max_simplex_order:
        filled_tri.append(tri_set)
        filled_tri_shared_node.append(previous_m)
    print('max simplex dimension possible: ', max_dim)
    
    filled_tri = [sorted(list(st)) for st in filled_tri]  # this one has simplices order >= 3
    st = gd.SimplexTree()
    tri_dict = dict() # input triangle to obtain a list of shared non-target nodes later.
    for tri, shared_node in zip(filled_tri, filled_tri_shared_node):
        st.insert(tri, shared_node)
        if frozenset(tri) in tri_dict:
            tri_dict[frozenset(tri)].append(shared_node)
        else:
            tri_dict[frozenset(tri)] = [shared_node]
            
    # this will let us find all the triangles after the expansion
    # meaning we get the faces of high order simplices (up to order L) that are triangles.
    st_gen = st.get_filtration()  
    filled_tri_new = []
    filled_tri_shared_node_new = []

    for splx in st_gen:
        sim, val = splx
        val = int(val)
        if len(sim) == 3:
            filled_tri_new.append(sim)
            filled_tri_shared_node_new.append(val)
            if frozenset(sim) in tri_dict:
                tri_dict[frozenset(sim)].append(val)
            else:
                tri_dict[frozenset(sim)] = [val]

    st_new = gd.SimplexTree()
    for tri, shared_node in zip(filled_tri_new, filled_tri_shared_node_new):
        if frozenset(tri) in tri_dict:
            st_new.insert(tri, shared_node)
        else:
            continue
         
    return filled_tri, st_new, tri_dict


def create_edge_graph_L1(g, na_adj_list_2_hop, set_interested_nodes, node_features_full, L):
    # create a graph representing upper adjacency between edges
    # two edges are connected if they are part of the same triangle.
    filled_tri, st, tri_dict = find_filled_tri(na_adj_list_2_hop, set_interested_nodes, L)

    fixed_triangles = set()   
    st_gen = st.get_filtration()
    simplices_2_tri = []
    for splx in st_gen:
        sim, val = splx
        if len(sim) == 3:
            simplices_2_tri.append(splx)
            fixed_triangles.add(tuple(sim))
    print('len(simplices_2_tri)/ num of triangles: ', len(simplices_2_tri))

    # index the triangles
    simplices_2 = dict()
    edges_part_of_tri = set()
    index = 0
    tri_ft = []

    for simplices in simplices_2_tri:
        s, filtration_val = simplices
        s_t = sorted(list(s))
        s = frozenset(s)
        edges_part_of_tri.add(tuple([s_t[0],s_t[1]]))
        edges_part_of_tri.add(tuple([s_t[1],s_t[2]]))
        edges_part_of_tri.add(tuple([s_t[0],s_t[2]]))
        simplices_2[s] = index # s = {m1_id,m2_id,m3_id}:index(tri)
        
        # 2-simplices features are the average of all non-target nodes shared by the three target nodes.
        if s in tri_dict:
            tri_ft_tmp = 0
            tri_id_list = tri_dict[s] # what are the non-target nodes that the 3 target nodes share
            fil_present = False
            count = 0
            for t in tri_id_list:
                if t == int(filtration_val):
                    fil_present = True
                tri_ft_tmp += node_features_full[int(t)]
                count += 1
            if fil_present == False:
                tri_ft_tmp += node_features_full[int(filtration_val)]
                count += 1
            tri_ft_tmp = tri_ft_tmp/count 
        else:
            tri_ft_tmp = node_features_full[int(filtration_val)]
        tri_ft.append(tri_ft_tmp)
        index += 1
    tri_ft = torch.stack(tri_ft)

    U, V, EID = g.edges(form='all', order='eid')
    get_eid = defaultdict(list)  # input src and dst to get eid

    edge_self_loop = []
    for eid in EID:
        eid_i = int(eid)
        src, dst = int(U[eid_i]), int(V[eid_i])
        get_eid[frozenset([src, dst])] = eid_i

        eid_i = int(eid)
        tmp = str(eid_i) + ' ' + str(eid_i) + ' ' + str(eid_i)
        edge_self_loop.append(tmp)

    upper_adj_edgelist_l1 = []
    list_tri_edgelist = []
    utilised_edges = []
    for simplex in simplices_2_tri:
        tri_list, filtration_val = simplex
        filtration_val = int(filtration_val)
        combination = [frozenset([tri_list[0], tri_list[1]]), frozenset([tri_list[1], tri_list[2]]),
                       frozenset([tri_list[0], tri_list[2]])]
        tri_edgelist = set()
        for item in combination:
            if item in get_eid:
                tri_edgelist.add(get_eid[item])
                utilised_edges.append(get_eid[item])
        if len(tri_edgelist) == 3:
            tmp = sorted(list(tri_edgelist))
            upper_adj_edgelist_l1.append(str(tmp[0]) + ' ' + str(tmp[1]) + ' ' + str(simplices_2[frozenset(tri_list)]))
            upper_adj_edgelist_l1.append(str(tmp[1]) + ' ' + str(tmp[2]) + ' ' + str(simplices_2[frozenset(tri_list)]))
            upper_adj_edgelist_l1.append(str(tmp[0]) + ' ' + str(tmp[2]) + ' ' + str(simplices_2[frozenset(tri_list)]))
            list_tri_edgelist.append(tuple([tri_edgelist, filtration_val]))

    # need to add self loops prior to these edges to ensure all "edges" are included as nodes in new graph
    # this new graph will give a set of eids
    # and we put in the tri feat on the edges using the data field.

    adj_edge_list_preprocess = list(set(upper_adj_edgelist_l1))
    adj_edge_list_preprocess.extend(edge_self_loop)

    nxg = nx.parse_edgelist(adj_edge_list_preprocess, nodetype=int, create_using=nx.DiGraph(),
                            data=(("share_tri_id", int),))
    upper_edge_graph = dgl.from_networkx(nxg, edge_attrs=['share_tri_id'])

    return upper_edge_graph, edges_part_of_tri, tri_ft


# prepare the features on the edges of the contructed 1-hop simplicial complex
# incorporating edge features constructed for the original edges.
def create_edge_features_sgat_ef(list_of_graph, node_features_full, edgelist_preprocess_dic, edges_part_of_tri_1, edgelist_preprocess_dic_ori_graph):
    output = [] # output is a list, one for each eta simplicial complex
    for i, g in enumerate(list_of_graph):
        print('Creating edge features.. This might take some time..')
        U, V, EID = g.edges(form='all', order='eid')
        edge_features = list()
        for index in range(g.number_of_edges()):
            key = str(int(U[index])) + " " + str(int(V[index]))
            if int(U[index]) == int(V[index]):  
                f3 = edgelist_preprocess_dic_ori_graph[key] # constructed edge feature on original edge
                tmp = node_features_full[int(U[index])] # self-loop; use back its own feature
                edge_features.append(torch.cat([tmp,f3], dim=0))   
            else:
                list_of_share_node_ids = edgelist_preprocess_dic[i][key] # list of lists
                num_of_list = len(list_of_share_node_ids) # before transforming into 1 list.
                # sum --> average of paths
                sum_ft_part1 = 0
                sum_ft_part2 = 0
                for share_node_id in list_of_share_node_ids:
                    sum_ft_part1 += node_features_full[share_node_id]
                    f1 = 0.5*edgelist_preprocess_dic_ori_graph[str(share_node_id) + " " + str(int(U[index]))]
                    f2 = 0.5*edgelist_preprocess_dic_ori_graph[str(share_node_id) + " " + str(int(V[index]))]
                    sum_ft_part2 += (f1+f2)
                sum_ft_part1 = sum_ft_part1/num_of_list
                sum_ft_part2 = sum_ft_part2/num_of_list
                
                edge_features.append(torch.cat([sum_ft_part1,sum_ft_part2], dim=0))   
        edge_features = torch.stack(edge_features)
        output.append(edge_features) 
    return output

def create_edge_features_sgat(list_of_graph, node_features_full, edgelist_preprocess_dic):
    output = [] # output is a list, one for each simplicial complex
    for i, g in enumerate(list_of_graph):
        print('Creating edge features... This might take some time...')
        U, V, EID = g.edges(form='all', order='eid')
        edge_features = list()
        for index in range(g.number_of_edges()):
            key = str(int(U[index])) + " " + str(int(V[index]))
            if int(U[index]) == int(V[index]):  # if it is self loop. just use back its own feature.
                tmp = node_features_full[int(U[index])]
                edge_features.append(tmp)   
            else:
                list_of_share_node_ids = edgelist_preprocess_dic[i][key] # list of lists
                num_of_list = len(list_of_share_node_ids) # before transforming into 1 list.
                sum_ft = 0
                for share_node_id in list_of_share_node_ids:
                    sum_ft += node_features_full[share_node_id]
                sum_ft = sum_ft/num_of_list
                edge_features.append(sum_ft)   
        edge_features_v2 = torch.stack(edge_features)
        output.append(edge_features_v2) # output is a list
    return output


def create_edge_features_upper(upper_edge_graph, edge_features, field_name, tri_ft, node_ft_size):
    ft_id = upper_edge_graph.edata[field_name]
    U, V, EID = upper_edge_graph.edges(form='all', order='eid')

    edge_ft_upper_original = list()
    for index in range(upper_edge_graph.number_of_edges()):
        key = frozenset([int(U[index]), int(V[index])])
        if len(key) == 1:  # if it is self loop. just use back its own feature.
            edge_ft_upper_original.append(edge_features[int(U[index])][:node_ft_size]) 
        else:
            share_node_id = int(ft_id[index])
            edge_ft_upper_original.append(tri_ft[share_node_id]) # tri_ft is average of non-target nodes' features
            
    edge_ft_upper_original = torch.stack(edge_ft_upper_original)
    return edge_ft_upper_original


def add_non_target_node_with_onehot_edge_ft(A, node_features):
    """
    TODO:
    edge_v2 = some_method_to_get_identity_matrix_tensor
    for i in range(A.shape[2]):  # and double check how to get the size of 3rd dimension of A. Not sure if it's .shape[2]
        fill_dict(dgl.from_scipy(coo_matrix(A[:, :, i])), edgelist_preprocess_dic, edge_v2[i])
    """
    edge = np.array([[1,0,0,0,0], [0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    edge_v2 = torch.as_tensor(edge)
    edgelist_preprocess_dic = dict()
    
    from scipy.sparse import coo_matrix
    g_edge_1 = dgl.from_scipy(coo_matrix(A[:, :, 0]))
    g_edge_2 = dgl.from_scipy(coo_matrix(A[:, :, 1]))
    g_edge_3 = dgl.from_scipy(coo_matrix(A[:, :, 2]))
    g_edge_4 = dgl.from_scipy(coo_matrix(A[:, :, 3]))
    g_edge_5 = dgl.from_scipy(coo_matrix(A[:, :, 4]))
    
    def fill_dict(g, edgelist_preprocess_dic, edge_onehot_ft):
        U, V, EID = g.edges(form='all', order='eid')
        for s,d in zip(U,V):
            s = int(s)
            d = int(d)
            tmp_edge = str(s) + " " + str(d)
            s_ft = node_features[s]
            d_ft = node_features[d]
            output_edge_ft = torch.cat([s_ft, d_ft, edge_onehot_ft], dim=0)
            edgelist_preprocess_dic[tmp_edge] = output_edge_ft
            
    fill_dict(g_edge_1, edgelist_preprocess_dic, edge_v2[0])
    fill_dict(g_edge_2, edgelist_preprocess_dic, edge_v2[1])
    fill_dict(g_edge_3, edgelist_preprocess_dic, edge_v2[2])
    fill_dict(g_edge_4, edgelist_preprocess_dic, edge_v2[3])
    fill_dict(g_edge_5, edgelist_preprocess_dic, edge_v2[4])
    return edgelist_preprocess_dic
