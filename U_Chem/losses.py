import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll_separate(args, generative_model,
                                  nodes_dist, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    nll = generative_model(x, h, node_mask, edge_mask, context)
    N = node_mask.squeeze(2).sum(1).long()
    log_pN = nodes_dist.log_prob(N)
    # assert anion_nll.size() == anion_log_pN.size()
    nll = nll - log_pN
    # Average over batch.
    nll = nll.mean(0)
    reg_term = torch.tensor([0.]).to(nll.device)
    mean_abs_z = 0.
    return nll, reg_term, mean_abs_z


def compute_loss_and_nll(args, generative_model,
                         anion_nodes_dist, anion_x, anion_h, anion_node_mask, anion_edge_mask,
                         cation_nodes_dist, cation_x, cation_h, cation_node_mask, cation_edge_mask,
                         anion_context, cation_context):
    anion_bs, anion_n_nodes, anion_n_dims = anion_x.size()
    cation_bs, cation_n_nodes, cation_n_dims = cation_x.size()

    anion_edge_mask = anion_edge_mask.view(anion_bs, anion_n_nodes * anion_n_nodes)
    cation_edge_mask = cation_edge_mask.view(cation_bs, cation_n_nodes * cation_n_nodes)

    # assert_correctly_masked(x, node_mask)

    # NOTE Here x is a position tensor, and h is a dictionary with keys
    # 'categorical' and 'integer'.
    anion_nll, cation_nll = generative_model(anion_x, anion_h, anion_node_mask, anion_edge_mask,
                                             cation_x, cation_h, cation_node_mask, cation_edge_mask,
                                             anion_context, cation_context)

    anion_N = anion_node_mask.squeeze(2).sum(1).long()
    anion_log_pN = anion_nodes_dist.log_prob(anion_N)
    # assert anion_nll.size() == anion_log_pN.size()
    anion_nll = anion_nll - anion_log_pN
    # Average over batch.
    anion_nll = anion_nll.mean(0)

    cation_N = cation_node_mask.squeeze(2).sum(1).long()
    cation_log_pN = cation_nodes_dist.log_prob(cation_N)
    # assert cation_nll.size() == cation_log_pN.size()
    cation_nll = cation_nll - cation_log_pN
    # Average over batch.
    cation_nll = cation_nll.mean(0)

    reg_term = torch.tensor([0.]).to(anion_nll.device)
    mean_abs_z = 0.

    return anion_nll, cation_nll, reg_term, mean_abs_z
