import torch
from torch import nn

from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch import nn
from copy import deepcopy

class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)

class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)

        Returns: Updated Node and Edge Feature matrices

        """
        row, col = edge_index

        # Edge Update
        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # Node Update
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)

class EdgeModel(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        return self.edge_mlp(out)

class TimeAwareNodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, flow_in_mlp, flow_out_mlp, node_mlp, node_agg_fn):
        super(TimeAwareNodeModel, self).__init__()

        self.flow_in_mlp = flow_in_mlp
        self.flow_out_mlp = flow_out_mlp
        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        flow_out_mask = row < col
        flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)
        flow_out = self.flow_out_mlp(flow_out_input)
        flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))

        flow_in_mask = row > col
        flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_mlp(flow_in_input)

        flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        flow = torch.cat((flow_in, flow_out), dim=1)

        return self.node_mlp(flow)

class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two MLPs, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.

    This class is based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, edge_in_dim = None, node_in_dim = None, edge_out_dim = None, node_out_dim = None,
                 node_fc_dims = None, edge_fc_dims = None, dropout_p = None, use_batchnorm = None):
        super(MLPGraphIndependent, self).__init__()

        if node_in_dim is not None :
            self.node_mlp = MLP(input_dim=node_in_dim, fc_dims=list(node_fc_dims) + [node_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_mlp = None

        if edge_in_dim is not None :
            self.edge_mlp = MLP(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_mlp = None

    def forward(self, edge_feats = None, nodes_feats = None):

        if self.node_mlp is not None:
            out_node_feats = self.node_mlp(nodes_feats)

        else:
            out_node_feats = nodes_feats

        if self.edge_mlp is not None:
            out_edge_feats = self.edge_mlp(edge_feats)

        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats

class HiclFeatsEncoder(nn.Module):
    def __init__(self, node_dim, detach_hicl_grad, merge_method='cat', skip_conn=False, ignore_mpn_out=False,
                 use_layerwise=False):
        super().__init__()
        assert merge_method in ('cat', 'sum')

        self.merge_method = merge_method
        self.ignore_mpn_out = ignore_mpn_out
        self.skip_conn = skip_conn  
        self.detach_hicl_grad = detach_hicl_grad

        self.encoder_hicl_feats_post_mpn = nn.Sequential(*[nn.Linear(node_dim, node_dim),nn.ReLU(),  nn.Linear(node_dim, node_dim),nn.ReLU()])
        self.encoder_hicl_feats = nn.Sequential(*[nn.Linear(node_dim, node_dim), nn.ReLU(),  nn.Linear(node_dim, node_dim), nn.ReLU()])
        
        self.merge_skip_conn = nn.Sequential(*[nn.Linear(2*node_dim, 2*node_dim),nn.ReLU(),  nn.Linear(2*node_dim, node_dim), nn.ReLU()])
        if self.merge_method == 'cat':
            self.merge_hicl_feats = nn.Sequential(*[nn.Linear(2*node_dim, 2*node_dim),nn.ReLU(),  nn.Linear(2*node_dim, node_dim), nn.ReLU()])
            
        else: 
            self.merge_hicl_feats = None
        
        self.use_layerwise = use_layerwise
        if self.use_layerwise:
            self.layerwise_merge = nn.Linear(2*node_dim, node_dim)
        
        
    def pool_node_feats(self, node_feats, labels):
        return scatter_mean(node_feats, torch.as_tensor(labels, device=node_feats.device).long(), dim=0)        

    def forward(self, latent_node_feats, hicl_feats):
        hicl_feats = self.encoder_hicl_feats(hicl_feats)
        
        if self.use_layerwise:
            hicl_feats = torch.cat((hicl_feats, latent_node_feats), dim=1)
            hicl_feats = self.layerwise_merge(hicl_feats)

        return hicl_feats

    def post_mpn_encode_node_feats(self, latent_node_feats, initial_hicl_feats, initial_node_feats):
        if initial_hicl_feats is None:
            initial_hicl_feats=initial_node_feats

        if self.ignore_mpn_out:
            return initial_hicl_feats
        
        if self.detach_hicl_grad:
            latent_node_feats = latent_node_feats.detach()
        
        latent_node_feats = self.encoder_hicl_feats_post_mpn(latent_node_feats)

        if self.skip_conn and initial_hicl_feats is not None:
            latent_node_feats = torch.cat((latent_node_feats,initial_hicl_feats), dim=1)
            latent_node_feats = self.merge_skip_conn(latent_node_feats)
            #latent_node_feats = latent_node_feats + initial_hicl_feats

        return latent_node_feats



class MOTMPNet(nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.

    This class was initially based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """
    def __init__(self, model_params, bb_encoder = None, motion=None, pos_feats=None, reid=None):
        """
        Defines all components of the model
        Args:
            bb_encoder: (might be 'None') CNN used to encode bounding box apperance information.
            model_params: dictionary contaning all model hyperparameters
        """
        super(MOTMPNet, self).__init__()

        self.node_cnn = bb_encoder
        self.model_params = model_params

        # Define Encoder and Classifier Networks
        encoder_feats_dict = deepcopy(model_params['encoder_feats_dict'])            
        if motion:
            encoder_feats_dict['edge_in_dim'] += 1
                
        if not reid:
            encoder_feats_dict['edge_in_dim'] -= 1

        if not pos_feats:
            encoder_feats_dict['edge_in_dim'] -= 4


        # print("EXPECTED EDGE INPUT DIM??", encoder_feats_dict['edge_in_dim'])

        classifier_feats_dict = model_params['classifier_feats_dict']

        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(model_params=model_params, encoder_feats_dict=encoder_feats_dict)

        self.num_enc_steps = model_params['num_enc_steps']
        self.num_class_steps = model_params['num_class_steps']

        if model_params['do_hicl_feats']:
            self.hicl_feats_encoder= HiclFeatsEncoder(node_dim = encoder_feats_dict['node_out_dim'], 
                                                      **model_params['hicl_feats_encoder'])

        else:
            self.hicl_feats_encoder=None

    def _build_core_MPNet(self, model_params, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            model_params: dictionary contaning all model hyperparameters
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = model_params['node_agg_fn']
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all MLPs involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = model_params['reattach_initial_nodes']
        self.reattach_initial_edges = model_params['reattach_initial_edges']

        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1

        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim']
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']

        # Define all MLPs used within the MPN
        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict = model_params['node_model_feats_dict']

        edge_mlp = MLP(input_dim=edge_model_in_dim,
                       fc_dims=edge_model_feats_dict['fc_dims'],
                       dropout_p=edge_model_feats_dict['dropout_p'],
                       use_batchnorm=edge_model_feats_dict['use_batchnorm'])

        flow_in_mlp = MLP(input_dim=node_model_in_dim,
                          fc_dims=node_model_feats_dict['fc_dims'],
                          dropout_p=node_model_feats_dict['dropout_p'],
                          use_batchnorm=node_model_feats_dict['use_batchnorm'])

        flow_out_mlp = MLP(input_dim=node_model_in_dim,
                           fc_dims=node_model_feats_dict['fc_dims'],
                           dropout_p=node_model_feats_dict['dropout_p'],
                           use_batchnorm=node_model_feats_dict['use_batchnorm'])

        node_mlp = nn.Sequential(*[nn.Linear(2 * node_model_feats_dict['fc_dims'][-1], encoder_feats_dict['node_out_dim']),
                                   nn.ReLU(inplace=True)])

        # Define all MLPs used within the MPN
        return MetaLayer(edge_model=EdgeModel(edge_mlp = edge_mlp),
                         node_model=TimeAwareNodeModel(flow_in_mlp = flow_in_mlp,
                                                       flow_out_mlp = flow_out_mlp,
                                                       node_mlp = node_mlp,
                                                       node_agg_fn = node_agg_fn))


    def forward(self, data, edge_level_embed=None, node_level_embed=None):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x_is_img = len(x.shape) == 4
        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)

            emb_dists = nn.functional.pairwise_distance(x[edge_index[0]], x[edge_index[1]]).view(-1, 1)
            edge_attr = torch.cat((edge_attr, emb_dists), dim = 1)

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x)
        # if self.traj:
        #     x_track, x_tracklens = data.x_track, data.x_tracklens
        #     latent_node_track_feats = self.trajectory_encoder(x_track, x_tracklens)
        #     latent_node_feats = torch.cat((latent_node_feats, latent_node_track_feats), dim=1)
        
        if node_level_embed is not None:
            n_nodes = latent_node_feats.shape[0]
            node_embed = node_level_embed.unsqueeze(0).expand(n_nodes, -1)
            latent_node_feats = node_embed
            print("GOT it here node")
            #print("NODE LEVEL EMBED!!. Layer", ix_layer)

        if edge_level_embed is not None:
            # print("GOT it here edge")
            n_edges = latent_edge_feats.shape[0]
            edge_embed = edge_level_embed.unsqueeze(0).expand(n_edges, -1)
            latent_edge_feats = latent_edge_feats + edge_embed
            #print("EDGE LEVEL EMBED!!. Layer", ix_layer)

        if hasattr(data, 'hicl_feats') and data.hicl_feats is not None and self.hicl_feats_encoder is not None:
            hicl_feats = data.hicl_feats
            latent_node_feats = self.hicl_feats_encoder(latent_node_feats, hicl_feats)

        else:
            hicl_feats = None
        
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1
        outputs_dict = {'classified_edges': []}
        for step in range(1, self.num_enc_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges:
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes:
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)

            # Message Passing Step
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            if step >= first_class_step:
                # Classification Step
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict['classified_edges'].append(dec_edge_feats)
        
        if self.hicl_feats_encoder is not None:
            outputs_dict['node_feats']= self.hicl_feats_encoder.post_mpn_encode_node_feats(latent_node_feats, hicl_feats, initial_node_feats)

        return outputs_dict