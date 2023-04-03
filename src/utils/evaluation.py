from torch_scatter import scatter_add


def compute_constr_satisfaction_rate(graph_obj, edges_out, undirected_edges=True, return_flow_vals=False):
    """
    Determines the proportion of Flow Conservation inequalities that are satisfied.
    For each node, the sum of incoming (resp. outgoing) edge values must be less or equal than 1.

    Args:
        graph_obj: 'Graph' object
        edges_out: BINARIZED output values for edges (1 if active, 0 if not active)
        undirected_edges: determines whether each edge in graph_obj.edge_index appears in both directions (i.e. (i, j)
        and (j, i) are both present (undirected_edges =True), or only (i, j), with  i<j (undirected_edges=False)
        return_flow_vals: determines whether the sum of incoming /outglong flow for each node must be returned

    Returns:
        constr_sat_rate: float between 0 and 1 indicating the proprtion of inequalities that are satisfied

    """
    # Get tensors indicataing which nodes have incoming and outgoing flows (e.g. nodes in first frame have no in. flow)
    edge_ixs = graph_obj.edge_index
    if undirected_edges:
        sorted, _ = edge_ixs.t().sort(dim = 1)
        sorted = sorted.t()
        div_factor = 2. # Each edge is predicted twice, hence, we divide by 2
    else:
        sorted = edge_ixs # Edges (i.e. node pairs) are already sorted
        div_factor = 1.  # Each edge is predicted once, hence, hence we divide by 1.

    # Compute incoming and outgoing flows for each node
    flow_out = scatter_add(edges_out, sorted[0],dim_size=graph_obj.num_nodes) / div_factor
    flow_in = scatter_add(edges_out, sorted[1], dim_size=graph_obj.num_nodes) / div_factor


    # Determine how many inequalitites are violated
    violated_flow_out = (flow_out > 1).sum()
    violated_flow_in = (flow_in > 1).sum()

    # Compute the final constraint satisfaction rate
    violated_inequalities = (violated_flow_in + violated_flow_out).float()
    flow_out_constr, flow_in_constr= sorted[0].unique(), sorted[1].unique()
    num_constraints = len(flow_out_constr) + len(flow_in_constr)
    constr_sat_rate = 1 - violated_inequalities / num_constraints
    if not return_flow_vals:
        return constr_sat_rate.item()

    else:
        return constr_sat_rate.item(), flow_in, flow_out