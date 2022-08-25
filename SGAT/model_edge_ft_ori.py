import torch
import torch.nn as nn
import torch.nn.functional as F
from gatconv_edge_ft import GATConv_edge as GATConv


class SGATLayer(nn.Module):
    """
    SGAT layer.
    Arguments
    ---------
    num_simplicial_complexes : number of simplicial complexes formed.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_simplicial_complexes, edge_in_ft_size, in_size, out_size, layer_num_heads, dropout):
        super(SGATLayer, self).__init__()

        self.gat_layers = nn.ModuleList()
        self.egat_upper_layers_l1 = nn.ModuleList()
        
        for i in range(num_simplicial_complexes):
            self.egat_upper_layers_l1.append(GATConv(in_size, edge_in_ft_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
            self.gat_layers.append(GATConv(edge_in_ft_size, in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu)) 

    def forward(self, gs, eg2s,  h, edge_features, edge_ft_upper):
        node_embeddings_list, edge_embeddings_list = [], []
        for i, g in enumerate(gs):
            tmp_h, _, a = self.gat_layers[i](g, h, edge_features[i])
            tmp_h = tmp_h.flatten(1)
            node_embeddings_list.append(tmp_h)
            
        for i, eg2 in enumerate(eg2s):
            tmp_eh2, _, _ = self.egat_upper_layers_l1[i](eg2, edge_features[i], edge_ft_upper[i])
            tmp_eh2 = tmp_eh2.flatten(1)
            edge_embeddings_list.append(tmp_eh2)

        node_embeddings = node_embeddings_list[0]
        edge_embeddings = edge_embeddings_list[0]
        
        return node_embeddings, edge_embeddings


class OutputLayer(nn.Module):
    def __init__(self, num_simplicial_complexes, edge_in_ft_size, in_size, out_size, layer_num_heads, dropout):
        super(OutputLayer, self).__init__()
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_simplicial_complexes):
            self.gat_layers.append(GATConv(edge_in_ft_size,in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))

    def forward(self, gs, h, edge_features):
        node_semantic_embeddings = []
        for i, g in enumerate(gs):
            if i == 0:
                tmp_h, _ ,_= self.gat_layers[i](g, h, edge_features[i])
                tmp_h = tmp_h.flatten(1)
                node_semantic_embeddings.append(tmp_h)
            
        node_semantic_embeddings = node_semantic_embeddings[0]
        return node_semantic_embeddings


class SGAT(nn.Module):
    def __init__(self, num_simplicial_complexes, edge_in_ft_size, in_size, hidden_size, out_size, num_heads, dropout):
        super(SGAT, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(SGATLayer(num_simplicial_complexes, edge_in_ft_size, in_size, hidden_size, num_heads[0], dropout))
        self.output_layer = OutputLayer(num_simplicial_complexes, hidden_size*num_heads[0], hidden_size*num_heads[0], hidden_size, num_heads[0], dropout)
        
        for l in range(1, len(num_heads)):
            self.layers.append(SGATLayer(num_simplicial_complexes, edge_in_ft_size, hidden_size * num_heads[l-1], hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size*num_heads[0] + hidden_size * num_heads[0], out_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.predict.weight, gain=gain)

    def forward(self, g, eg2, all_feat):
        '''
        h: node features
        edge_feature: feature of 1-simplices
        edge_ft_upper: feature of upper adjacency edges. 
        This means the edges of the graph described by the upper adjacency matrix.
        In this upper adjacency matrix represented graph, the "vertices" are the 1-simplices.
        '''
        h, edge_feature, edge_ft_upper = all_feat
        
        for gnn in self.layers:
            h, edge_feature = gnn(g, eg2, h, [edge_feature], edge_ft_upper)

        h2 = self.output_layer(g, h, [edge_feature])
        h_prime = torch.cat([h, h2], dim=1)

        return self.predict(h_prime), h_prime
