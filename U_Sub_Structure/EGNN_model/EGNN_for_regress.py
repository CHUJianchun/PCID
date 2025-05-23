from torch import nn
from U_Sub_Structure.EGNN_model.util_for_regress import *
from torch.nn.init import zeros_, xavier_uniform_

from U_Sub_Structure.MLP import MLP


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()

    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat


class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False,
                 t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq = t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        # if recurrent:
        # self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            # out = self.gru(out, h)
        return out


class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf),
                                 act_fn,
                                 layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x * self.reg
        return x_out


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True,
                 coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            nn.BatchNorm1d(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            nn.BatchNorm1d(hidden_nf),
            act_fn)
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            nn.BatchNorm1d(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        for layer in self.node_mlp:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), nn.BatchNorm1d(hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
            for l_ in self.att_mlp:
                if isinstance(l_, nn.Linear):
                    xavier_uniform_(l_.weight.data)
                    zeros_(l_.bias.data)

        for l_ in self.edge_mlp:
            if isinstance(l_, nn.Linear):
                xavier_uniform_(l_.weight.data)
                zeros_(l_.bias.data)

        for l_ in self.node_mlp:
            if isinstance(l_, nn.Linear):
                xavier_uniform_(l_.weight.data)
                zeros_(l_.bias.data)

        for l_ in self.coord_mlp:
            if isinstance(l_, nn.Linear) and l_.bias is not None:
                xavier_uniform_(l_.weight.data)
                zeros_(l_.bias.data)

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100,
                            max=100)
        # This is never activated but just in case it exposed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class E_GCL_vel(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True,
                 coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim,
                       act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention,
                       norm_diff=norm_diff, tanh=tanh)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        coord += self.coord_mlp_vel(h) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class GCL_rf_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        # layer.weight.uniform_(-0.1, 0.1)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh())  # we had to add the tanh to keep this method stable

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True,
                 coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim,
                       act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg * self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial * 0.1, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        # coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN_basic(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN_basic, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                     edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                                                     act_fn=act_fn, recurrent=True, coords_weight=coords_weight,
                                                     attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


class EGNN_Regress(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN_Regress, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        zeros_(self.embedding.bias.data)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                     edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                                                     act_fn=act_fn, recurrent=True, coords_weight=coords_weight,
                                                     attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      nn.BatchNorm1d(self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))
        for layer in self.node_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf + 2, self.hidden_nf),
                                       nn.BatchNorm1d(self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        for layer in self.graph_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, t, p):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(torch.cat((h, t, p), dim=1))
        return pred.squeeze(1)


class EGNN_Regress_Seperate(nn.Module):
    def __init__(self, anion_in_node_nf, anion_in_edge_nf, anion_hidden_nf,
                 cation_in_node_nf, cation_in_edge_nf, cation_hidden_nf,
                 device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN_Regress_Seperate, self).__init__()
        self.anion_hidden_nf = anion_hidden_nf
        self.cation_hidden_nf = cation_hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.anion_embedding = nn.Sequential(nn.Linear(anion_in_node_nf, anion_hidden_nf),
                                             nn.BatchNorm1d(anion_hidden_nf),
                                             act_fn,
                                             )
        self.cation_embedding = nn.Sequential(nn.Linear(cation_in_node_nf, cation_hidden_nf),
                                              nn.BatchNorm1d(cation_hidden_nf),
                                              act_fn,
                                              )
        xavier_uniform_(self.anion_embedding[0].weight.data)
        xavier_uniform_(self.cation_embedding[0].weight.data)
        zeros_(self.anion_embedding[0].bias.data)
        zeros_(self.cation_embedding[0].bias.data)
        self.node_attr = node_attr
        if node_attr:
            anion_n_node_attr = anion_in_node_nf
            cation_n_node_attr = cation_in_node_nf
        else:
            anion_n_node_attr = 0
            cation_n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module(
                "anion_gcl_%d" % i, E_GCL_mask(self.anion_hidden_nf, self.anion_hidden_nf, self.anion_hidden_nf,
                                               edges_in_d=anion_in_edge_nf, nodes_attr_dim=anion_n_node_attr,
                                               act_fn=act_fn, recurrent=True, coords_weight=coords_weight,
                                               attention=attention))
            self.add_module(
                "cation_gcl_%d" % i, E_GCL_mask(self.cation_hidden_nf, self.cation_hidden_nf, self.cation_hidden_nf,
                                                edges_in_d=cation_in_edge_nf, nodes_attr_dim=cation_n_node_attr,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight,
                                                attention=attention))

        self.anion_node_dec = nn.Sequential(nn.Linear(self.anion_hidden_nf, self.anion_hidden_nf),
                                            nn.BatchNorm1d(self.anion_hidden_nf),
                                            act_fn,
                                            )
        self.cation_node_dec = nn.Sequential(nn.Linear(self.cation_hidden_nf, self.cation_hidden_nf),
                                             nn.BatchNorm1d(self.cation_hidden_nf),
                                             act_fn,
                                             )
        for layer in self.anion_node_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
                xavier_uniform_(layer.weight.data)
        for layer in self.cation_node_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
                xavier_uniform_(layer.weight.data)
        self.graph_dec = nn.Sequential(nn.Linear(self.anion_hidden_nf + self.cation_hidden_nf + 2, 1))
        for layer in self.graph_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
                xavier_uniform_(layer.weight.data)
        self.to(self.device)

    def forward(self, anion_h0, anion_x, anion_edges, anion_edge_attr, anion_node_mask, anion_edge_mask, anion_n_nodes,
                cation_h0, cation_x, cation_edges, cation_edge_attr, cation_node_mask, cation_edge_mask, cation_n_nodes,
                t, p):
        anion_h = self.anion_embedding(anion_h0)
        cation_h = self.cation_embedding(cation_h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                anion_h, _, _ = self._modules["anion_gcl_%d" % i](
                    anion_h, anion_edges, anion_x, anion_node_mask, anion_edge_mask, edge_attr=anion_edge_attr,
                    node_attr=anion_h0, n_nodes=anion_n_nodes)
                cation_h, _, _ = self._modules["cation_gcl_%d" % i](
                    cation_h, cation_edges, cation_x, cation_node_mask, cation_edge_mask, edge_attr=cation_edge_attr,
                    node_attr=cation_h0, n_nodes=cation_n_nodes)
            else:
                anion_h, _, _ = self._modules["anion_gcl_%d" % i](
                    anion_h, anion_edges, anion_x, anion_node_mask, anion_edge_mask, edge_attr=anion_edge_attr,
                    node_attr=None, n_nodes=anion_n_nodes)
                cation_h, _, _ = self._modules["cation_gcl_%d" % i](
                    cation_h, cation_edges, cation_x, cation_node_mask, cation_edge_mask, edge_attr=cation_edge_attr,
                    node_attr=None, n_nodes=cation_n_nodes)

        # anion_h = self.anion_node_dec(anion_h)
        anion_h = anion_h * anion_node_mask
        anion_h = anion_h.view(-1, anion_n_nodes, self.anion_hidden_nf)
        anion_h = torch.mean(anion_h, dim=1)

        # cation_h = self.cation_node_dec(cation_h)
        cation_h = cation_h * cation_node_mask
        cation_h = cation_h.view(-1, cation_n_nodes, self.cation_hidden_nf)
        cation_h = torch.mean(cation_h, dim=1)

        pred = self.graph_dec(torch.cat((anion_h, cation_h, t, p), dim=1))
        return pred.squeeze(1)


class EGNN_Encoder(nn.Module):
    def __init__(self, anion_in_node_nf, anion_in_edge_nf, anion_hidden_nf,
                 cation_in_node_nf, cation_in_edge_nf, cation_hidden_nf,
                 embedding_dim, device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN_Encoder, self).__init__()
        self.anion_hidden_nf = anion_hidden_nf
        self.cation_hidden_nf = cation_hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.properties_mlp = MLP(nin=10, nout=embedding_dim, nh=embedding_dim)
        # Encoder
        self.anion_embedding = nn.Sequential(nn.Linear(anion_in_node_nf, anion_hidden_nf),
                                             nn.BatchNorm1d(anion_hidden_nf),
                                             act_fn,
                                             )
        self.cation_embedding = nn.Sequential(nn.Linear(cation_in_node_nf, cation_hidden_nf),
                                              nn.BatchNorm1d(cation_hidden_nf),
                                              act_fn,
                                              )
        xavier_uniform_(self.anion_embedding[0].weight.data)
        xavier_uniform_(self.cation_embedding[0].weight.data)
        zeros_(self.anion_embedding[0].bias.data)
        zeros_(self.cation_embedding[0].bias.data)
        self.node_attr = node_attr
        if node_attr:
            anion_n_node_attr = anion_in_node_nf
            cation_n_node_attr = cation_in_node_nf
        else:
            anion_n_node_attr = 0
            cation_n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module(
                "anion_gcl_%d" % i, E_GCL_mask(self.anion_hidden_nf, self.anion_hidden_nf, self.anion_hidden_nf,
                                               edges_in_d=anion_in_edge_nf, nodes_attr_dim=anion_n_node_attr,
                                               act_fn=act_fn, recurrent=True, coords_weight=coords_weight,
                                               attention=attention))
            self.add_module(
                "cation_gcl_%d" % i, E_GCL_mask(self.cation_hidden_nf, self.cation_hidden_nf, self.cation_hidden_nf,
                                                edges_in_d=cation_in_edge_nf, nodes_attr_dim=cation_n_node_attr,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight,
                                                attention=attention))

        self.anion_node_dec = nn.Sequential(nn.Linear(self.anion_hidden_nf, self.anion_hidden_nf),
                                            nn.BatchNorm1d(self.anion_hidden_nf),
                                            act_fn,
                                            )
        self.cation_node_dec = nn.Sequential(nn.Linear(self.cation_hidden_nf, self.cation_hidden_nf),
                                             nn.BatchNorm1d(self.cation_hidden_nf),
                                             act_fn,
                                             )
        for layer in self.anion_node_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
                xavier_uniform_(layer.weight.data)
        for layer in self.cation_node_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
                xavier_uniform_(layer.weight.data)
        self.anion_graph_dec = nn.Sequential(nn.Linear(self.anion_hidden_nf, int(embedding_dim / 3)))
        self.cation_graph_dec = nn.Sequential(nn.Linear(self.cation_hidden_nf, embedding_dim - int(embedding_dim / 3)))
        for layer in self.anion_graph_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
                xavier_uniform_(layer.weight.data)
        for layer in self.cation_graph_dec:
            if isinstance(layer, nn.Linear):
                zeros_(layer.bias.data)
                xavier_uniform_(layer.weight.data)
        self.to(self.device)

    def forward(self, anion_h0, anion_x, anion_edges, anion_edge_attr, anion_node_mask, anion_edge_mask, anion_n_nodes,
                cation_h0, cation_x, cation_edges, cation_edge_attr, cation_node_mask, cation_edge_mask, cation_n_nodes,
                properties):
        anion_h = self.anion_embedding(anion_h0)
        cation_h = self.cation_embedding(cation_h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                anion_h, _, _ = self._modules["anion_gcl_%d" % i](
                    anion_h, anion_edges, anion_x, anion_node_mask, anion_edge_mask, edge_attr=anion_edge_attr,
                    node_attr=anion_h0, n_nodes=anion_n_nodes)
                cation_h, _, _ = self._modules["cation_gcl_%d" % i](
                    cation_h, cation_edges, cation_x, cation_node_mask, cation_edge_mask, edge_attr=cation_edge_attr,
                    node_attr=cation_h0, n_nodes=cation_n_nodes)
            else:
                anion_h, _, _ = self._modules["anion_gcl_%d" % i](
                    anion_h, anion_edges, anion_x, anion_node_mask, anion_edge_mask, edge_attr=anion_edge_attr,
                    node_attr=None, n_nodes=anion_n_nodes)
                cation_h, _, _ = self._modules["cation_gcl_%d" % i](
                    cation_h, cation_edges, cation_x, cation_node_mask, cation_edge_mask, edge_attr=cation_edge_attr,
                    node_attr=None, n_nodes=cation_n_nodes)

        # anion_h = self.anion_node_dec(anion_h)
        anion_h = anion_h * anion_node_mask
        anion_h = anion_h.view(-1, anion_n_nodes, self.anion_hidden_nf)
        anion_h = torch.mean(anion_h, dim=1)

        # cation_h = self.cation_node_dec(cation_h)
        cation_h = cation_h * cation_node_mask
        cation_h = cation_h.view(-1, cation_n_nodes, self.cation_hidden_nf)
        cation_h = torch.mean(cation_h, dim=1)
        anion_embedding = self.anion_graph_dec(anion_h)
        cation_embedding = self.cation_graph_dec(cation_h)
        graph_embedding = torch.cat((anion_embedding, cation_embedding), dim=1)
        property_embedding = self.properties_mlp(properties)
        return graph_embedding, property_embedding

    @torch.no_grad()
    def get_property_embedding(self, properties):
        return self.properties_mlp(properties)
