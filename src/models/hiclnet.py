import torch
from torch import nn


class HICLNet(nn.Module):
    """
    Hierarchical network that contains all layers
    """
    def __init__(self, submodel_type, submodel_params, hicl_depth, use_motion, use_reid_edge, use_pos_edge,
                 share_weights, edge_level_embed, node_level_embed):
        """
        :param model_type: Network to use at each layer
        :param model_params: Parameters of the model for each layer
        :param depth: Number of layers in the hierarchical model
        """
        super(HICLNet, self).__init__()
        
        for per_layer_params in (use_motion, use_reid_edge, use_pos_edge):
            assert hicl_depth == len(per_layer_params), f"{hicl_depth }, {per_layer_params}"

        assert share_weights in ('none', 'all_but_first', 'all') 
        _SHARE_WEIGHTS_IDXS = {'none': range(hicl_depth), 
                               'all_but_first':[0]+ (hicl_depth - 1)*[1], # e.g. [0, 1, 1, 1]
                               'all': hicl_depth*[0]} # e.g. [0, 0, 0, 0]

        layer_idxs = _SHARE_WEIGHTS_IDXS[share_weights]

        # All hierarchical layers are contained in a nn.ModuleList
        layers = [submodel_type(submodel_params, motion=motion, pos_feats=pos_feats, reid=reid) 
                  for motion, pos_feats, reid in zip(use_motion, use_pos_edge, use_reid_edge)]

        self.layers = nn.ModuleList([layers[idx] for idx in layer_idxs])

        if edge_level_embed:
            edge_dim=submodel_params['encoder_feats_dict']['edge_out_dim']
            self.edge_level_embed = nn.Embedding(hicl_depth, edge_dim)
        
        else:
            self.edge_level_embed = None

        if node_level_embed:
            node_dim=submodel_params['encoder_feats_dict']['node_out_dim']
            self.node_level_embed = nn.Embedding(hicl_depth, node_dim)
        
        else:
            self.node_level_embed = None

    def forward(self, data, ix_layer):
        """
        Forward pass with the self.layers[ix_layer]
        """
        edge_level_embed=node_level_embed=None
        if self.edge_level_embed is not None:
            edge_level_embed = self.edge_level_embed.weight[ix_layer]

        if self.node_level_embed is not None:
            node_level_embed = self.node_level_embed.weight[ix_layer]

        return self.layers[ix_layer](data, node_level_embed=node_level_embed, edge_level_embed=edge_level_embed)
