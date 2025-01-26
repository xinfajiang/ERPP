import torch
from torch import nn
from . import layers
import torch.nn.functional as F
import math
from base_nbfnet import BaseNBFNet
class ERPP(nn.Module):
    def __init__(self, device, dataset, args):
        super(ERPP, self).__init__()
        self.device = device
        self.args = args
        self.ent_nums = []
        self.rel_nums = []
        for i in range(5):
            self.ent_nums.append(dataset.snapshots[i].num_ent)
            self.rel_nums.append(dataset.snapshots[i].num_rel)
        self.RelEmbeddings = nn.Embedding(self.rel_nums[0], 64).to(self.device)
        self.IrpCpa = IRPCPA(self.args['dim'], self.args['dim'], self.rel_nums[0],
                             self.args['message_func'], self.args['aggregate_func'], self.args['layer_num'])
        self.hppas=self.args.hypas
        self.current_snaps = -1
        for name, param in self.named_parameters():
            self.register_buffer(('old.' + name).replace(".", "_"), torch.tensor([[]]))

    def forward(self, entity_graph_data, batch, r_ind):

        score = self.IrpCpa(entity_graph_data, batch, r_ind, self.RelEmbeddings.weight.data)

        loss_old = 0
        if self.current_snaps > 0 and self.training:
            loss_old = []
            for name, param in self.named_parameters():
    
                if name == 'RelEmbeddings.weight':
                    old_data = getattr(self, ('old.' + name).replace(".", "_"))
                    new_data = param[:old_data.size(0)]
                    are_equal = torch.equal(new_data, old_data)
                    loss_old.append((( (new_data - old_data) ** 2).sum()))
                    
                elif 'RelationProjectionWeight' in name:
                    old_data = getattr(self, ('old.' + name).replace(".", "_"))
                    new_data = param[:old_data.size(0)]
                    are_equal = torch.equal(new_data, old_data)
                    loss_old.append((( (new_data - old_data) ** 2).sum()))

                elif 'RelationProjectionBias' in name:
                    old_data = getattr(self, ('old.' + name).replace(".", "_"))
                    new_data = param[:old_data.size(0)]
                    are_equal = torch.equal(new_data, old_data)
                    loss_old.append((( (new_data - old_data) ** 2).sum()))

                else:
                    old_data = getattr(self, ('old.' + name).replace(".", "_"))
                    are_equal = torch.equal(param, old_data)
                    loss_old.append((( (param - old_data) ** 2).sum()))
            return score, self.hper[self.current_snaps]*sum(loss_old)
        return score, loss_old

    def Inherit(self, index):
        num = 0
        for name, param in self.named_parameters():
            value = param.data
            self.register_buffer(('old.' + name).replace(".", "_"), value.clone())

            if "RelationProjectionWeight1" in name:
                RelationProjectionWeight1 = nn.Parameter(torch.empty((self.rel_nums[index+1]*64, 64))).to(self.device) 
                nn.init.kaiming_uniform_(RelationProjectionWeight1, a=math.sqrt(5))
                NewRelationProjectionWeight1 = RelationProjectionWeight1.data
                NewRelationProjectionWeight1[:self.rel_nums[index]*64] = torch.nn.Parameter(value)
                self.EntityModel0.layers[num].RelationProjectionWeight1 = torch.nn.Parameter(NewRelationProjectionWeight1)
            if "RelationProjectionBias1" in name:
                RelationProjectionBias1 = nn.Parameter(torch.empty(self.rel_nums[index+1]*64)).to(self.device)
                nn.init.uniform_(RelationProjectionBias1, -1/math.sqrt(64), 1 / math.sqrt(64))
                NewRelationProjectionBias1 = RelationProjectionBias1.data
                NewRelationProjectionBias1[:self.rel_nums[index]*64] = torch.nn.Parameter(value)
                self.EntityModel0.layers[num].RelationProjectionBias1 = torch.nn.Parameter(NewRelationProjectionBias1)
                num += 1

        rel_embeddings = nn.Embedding(self.rel_nums[index+1], 64).to(self.device)
        new_rel_embeddings = rel_embeddings.weight.data
        new_rel_embeddings[:self.rel_nums[index]] = torch.nn.Parameter(self.RelEmbeddings.weight.data)
        self.RelEmbeddings.weight = torch.nn.Parameter(new_rel_embeddings)


class IRPCPA(BaseNBFNet):
    def __init__(self, input_dim, hidden_dims, num_relation=1, num=6, message_func='distmult', aggregate_func = 'sum', layer_norm=True):
        super().__init__(input_dim, hidden_dims, num_relation, num, message_func, aggregate_func, layer_norm)
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.layer_norm = layer_norm
        self.layer_num = num
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dims
        self.activation = 'relu'
        for i in range(self.layer_num):
            self.layers.append(
                layers.Layer(
                    self.input_dim, self.hidden_dim, num_relation,
                    self.hidden_dim, self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )
        feature_dim = self.input_dim + self.hidden_dim
        self.mlp = nn.Sequential()
        mlp = []
        mlp.append(nn.Linear(feature_dim, feature_dim))
        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, rel_embeddings,separate_grad=False):
        batch_size = len(r_index)

        query = rel_embeddings[r_index]
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary
        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, r_index, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) 
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)
        return {
            "node_feature": output,
            "edge_weights": edge_weights
        }

    def forward(self, data, batch, r_ind, rel_embeddings):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
        shape = h_index.shape
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, r_ind,num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        for num_layer, layer in enumerate(self.layers):
            layer.relation = rel_embeddings
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0], rel_embeddings)
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index) 
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    
    


