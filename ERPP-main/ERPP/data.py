import torch
from copy import deepcopy as dcopy
from torch_geometric.data import Data
def build_edge_index(s, o):
    index = [s + o, o + s]
    return torch.LongTensor(index)  

def load_fact(path):
    facts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            s, r, o = line[0], line[1], line[2]
            facts.append((s, r, o))
    return facts

class KnowledgeGraph():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = dict(), dict(), dict(), dict()
        self.relation2inv = dict()
        self.snapshots = {i: SnapShot(self.device) for i in range(int(5))}
        self.load_data()

    def load_data(self):
        edge_s_all, edge_r_all, edge_o_all = [], [], []
        edge_s_train_all, edge_r_train_all, edge_o_train_all = [], [], []
        last_relation2inv = dict()
        for ss_id in range(int(5)):
            train_facts = load_fact("data/"+ self.args['datasets'] + '/'+ str(ss_id) + '/' + 'train.txt')
            test_facts = load_fact("data/"+ self.args['datasets'] + '/'+ str(ss_id) + '/'  + 'test.txt')
            valid_facts = load_fact("data/"+ self.args['datasets'] + '/'+ str(ss_id) + '/'  + 'valid.txt')
            self.expand_entity_relation(train_facts, True)
            self.expand_entity_relation(valid_facts, False)
            self.expand_entity_relation(test_facts, False)
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts)
            test = self.fact2id(test_facts)
            edge_s, edge_r, edge_o = [], [], []
            edge_s, edge_o, edge_r = self.expand_kg(train, edge_s, edge_o, edge_r)
            edge_s_train_all, edge_o_train_all, edge_r_train_all = self.expand_kg(train,edge_s_train_all, edge_o_train_all, edge_r_train_all)
            edge_s_all, edge_o_all, edge_r_all = self.expand_kg(train, edge_s_all, edge_o_all, edge_r_all)
            edge_s_all, edge_o_all, edge_r_all = self.expand_kg(valid, edge_s_all, edge_o_all, edge_r_all)
            edge_s_all, edge_o_all, edge_r_all = self.expand_kg(test, edge_s_all, edge_o_all, edge_r_all)
            self.store_snapshot(ss_id, train, test, edge_s, edge_o, edge_r, edge_s_all, edge_r_all, edge_o_all, edge_s_train_all, edge_r_train_all, edge_o_train_all)
       
        
    def expand_entity_relation(self, facts, flag):
        for (s, r, o) in facts:
            if s not in self.entity2id.keys():
                self.entity2id[s] = self.num_ent
                self.num_ent += 1
            if o not in self.entity2id.keys():
                self.entity2id[o] = self.num_ent
                self.num_ent += 1
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                self.relation2id[r + '_inv'] = self.num_rel + 1
                self.relation2inv[self.num_rel] = self.num_rel + 1
                self.relation2inv[self.num_rel + 1] = self.num_rel
                self.num_rel += 2

    def fact2id(self, facts, order=False):
        fact_id = []
        if order:
            i = 0
            while len(fact_id) < len(facts):
                for (s, r, o) in facts:
                    if self.relation2id[r] == i:
                        fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
                i = i + 2
        else:
            for (s, r, o) in facts:
                fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return fact_id

    def expand_kg(self, facts, edge_s, edge_o, edge_r):
        for (h, r, t) in facts:
            edge_s.append(h)
            edge_r.append(r)
            edge_o.append(t)
        return edge_s, edge_o, edge_r

    def store_snapshot(self, ss_id, train_new, test, edge_s, edge_o, edge_r, edge_s_all, edge_r_all, edge_o_all, edge_s_train_all, edge_r_train_all, edge_o_train_all):

        self.snapshots[ss_id].num_ent = dcopy(self.num_ent)
        self.snapshots[ss_id].num_rel = dcopy(self.num_rel) 

        self.snapshots[ss_id].train_new = dcopy(train_new) 

        self.snapshots[ss_id].test = dcopy(test) 
        self.snapshots[ss_id].test_edge_index = torch.LongTensor(test)[:,[0,2]].T.to(self.device)
        self.snapshots[ss_id].test_edge_type = torch.LongTensor(test)[:,[1]].T.to(self.device)
        
        self.snapshots[ss_id].edge_index = build_edge_index(edge_s, edge_o).to(self.device)
        self.snapshots[ss_id].edge_type = torch.cat([torch.LongTensor(edge_r), torch.LongTensor(edge_r) + 1]).to(self.device)
        self.snapshots[ss_id].ent_graph = Data(edge_index=self.snapshots[ss_id].edge_index, edge_type=self.snapshots[ss_id].edge_type,
                                               num_nodes=self.snapshots[ss_id].num_ent, num_relations=self.snapshots[ss_id].num_rel)
        self.snapshots[ss_id].filter_edge_index = torch.LongTensor([edge_s_all,edge_o_all]).to(self.device)
        
        self.snapshots[ss_id].filter_graph = Data(edge_index=self.snapshots[ss_id].filter_edge_index.to(self.device),
                                                    edge_type=torch.LongTensor(edge_r_all).to(self.device),
                                               num_nodes=self.snapshots[ss_id].num_ent, num_relations=self.snapshots[ss_id].num_rel)
        
        self.snapshots[ss_id].edge_index_all = build_edge_index(edge_s_train_all, edge_o_train_all).to(self.device)
        self.snapshots[ss_id].edge_type_all = torch.cat([torch.LongTensor(edge_r_train_all), torch.LongTensor(edge_r_train_all) + 1]).to(self.device)
        self.snapshots[ss_id].ent_graph_all = Data(edge_index=self.snapshots[ss_id].edge_index_all, edge_type=self.snapshots[ss_id].edge_type_all,
                                               num_nodes=self.snapshots[ss_id].num_ent, num_relations=self.snapshots[ss_id].num_rel)
        
class SnapShot():
    def __init__(self, device):
        self.device = device
        self.num_ent, self.num_rel = 0, 0
        self.train_new, self.train_all, self.test, self.valid, self.valid_all, self.test_all = list(), list(), list(), list(), list(), list()
        self.edge_s, self.edge_r, self.edge_o = [], [], []
        self.sr2o_all = dict()
        self.edge_index, self.edge_type = None, None
        self.new_entities = []