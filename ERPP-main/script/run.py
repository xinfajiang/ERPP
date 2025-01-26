import os
import sys
import math
import pprint
from itertools import islice
from torch_scatter import scatter_add
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ERPP import tasks, util
from ERPP.models import ERPP
import numpy as np
from copy import deepcopy as dcopy
from collections import deque
from collections import deque

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
    def __init__(self, device):
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
            train_facts = load_fact("/data4/yj/jxw/jxf/lifelongultra/ULTRA-main/ULTRA-main/graph_higher/" + str(ss_id) + '/' + 'train.txt')
            test_facts = load_fact("/data4/yj/jxw/jxf/lifelongultra/ULTRA-main/ULTRA-main/graph_higher/" + str(ss_id) + '/' + 'test.txt')
            valid_facts = load_fact("/data4/yj/jxw/jxf/lifelongultra/ULTRA-main/ULTRA-main/graph_higher/" + str(ss_id) + '/' + 'valid.txt')
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


def train_and_validate(cfg, model, LFdataset, device, logger, filtered_data=None, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    relation_inv = torch.tensor([LFdataset.relation2inv[key] for key in range(len(LFdataset.relation2inv))]).to(device)
    cls = cfg.optimizer.pop("class") 
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    else:
        parallel_model = model
    train_epoch = [15, 10, 10, 10, 10]
    all_kpi = [[],[],[],[],[]]
    epochs = []

    for snap_num in range(5):
        if rank == 0:
            print('--------------------------------')
            print("start: ", snap_num)
        best_result = torch.tensor(float("-inf"), device=device)
        best_epoch = torch.tensor(-1, device=device)
        train_triplets = torch.cat([LFdataset.snapshots[snap_num].ent_graph.edge_index, LFdataset.snapshots[snap_num].ent_graph.edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
        train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)
        batch_per_epoch = batch_per_epoch or len(train_loader)
        batch_id = 0
        model.current_snaps = snap_num

        for epoch in range(train_epoch[snap_num]):
            parallel_model.train()
            if util.get_rank() == 0:
                logger.warning("Epoch %d begin" % epoch)
            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                batch = tasks.negative_sampling(LFdataset.snapshots[snap_num].ent_graph, batch, cfg.task.num_negative,
                                                    strict=cfg.task.strict_negative)
                pred, reloss = parallel_model(LFdataset.snapshots[snap_num].ent_graph, 0, batch, relation_inv,snap_num)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean() + reloss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning("Epoch %d end" % epoch)
                logger.warning("average binary cross entropy: %g" % avg_loss)


            if util.get_rank() == 0:
                logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
                state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                torch.save(state, "model_epoch_%d.pth" % epoch)
                torch.save(model, "save/"+ str(snap_num) + "/" + "model_epoch_%d.pth" % epoch)


            if rank == 0:
                logger.warning("Evaluate on valid")
            result = test(cfg, snap_num, snap_num, model, LFdataset, relation_inv,filtered_data=filtered_data, device=device, logger=logger)
            if result> best_result:
                best_result = result
                best_epoch = epoch
        epochs.append(best_epoch)
        
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
        state = torch.load("save/" + str(snap_num) + "/" + "model_epoch_%d.pth"%best_epoch, map_location=device)
        model.load_state_dict(state["model"])

        print("end: ", snap_num)
        for test_num in range(snap_num+1):
            if rank == 0:
                print("start: ", test_num, 'test')
            _ = test(cfg, test_num, test_num, model, LFdataset, relation_inv, filtered_data=filtered_data, device=device, logger=logger)

        if snap_num+1 < 5:
            model.Inherit(snap_num)
            optimizer = getattr(optim, cls)(filter(lambda p: p.requires_grad, model.parameters()), **cfg.optimizer)
            
        if rank == 0:
            print('------------------------------')
    

@torch.no_grad()
def test(cfg, test_num, snap_num, model, LFdataset, relation_idv,device, logger, filtered_data=None, return_metrics=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([LFdataset.snapshots[test_num].test_edge_index, LFdataset.snapshots[test_num].test_edge_type]).t()
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=None, shuffle=False)
    
    model.eval()
    rankings = []
    s = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []  
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(LFdataset.snapshots[test_num].ent_graph_all, batch)
        t_pred, _ = model(LFdataset.snapshots[test_num].ent_graph_all, 0, t_batch, relation_idv,test_num)
        h_pred, _ = model(LFdataset.snapshots[test_num].ent_graph_all, 0, h_batch, relation_idv,test_num)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(LFdataset.snapshots[test_num].filter_graph, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_scores = t_pred[torch.arange(batch.shape[0]), pos_t_index]
        h_scores = h_pred[torch.arange(batch.shape[0]), pos_h_index]
        s += [t_scores, h_scores] 
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]
    ss = torch.cat(s)
    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking 
                _num_neg = all_num_negative 
                _metric_name = metric
            
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            if metric == 'hits@1' or metric == 'hits@3' or metric == 'hits@10' or metric == 'mrr':
                logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()
    return mrr



if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    torch.manual_seed(args.seed + util.get_rank())
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    task_name = cfg.task["name"]
    device = util.get_device(cfg)
    LFdataset = KnowledgeGraph(device)
    model = ERPP(device=device, dataset = LFdataset,
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )
    model = model.to(device)
    train_and_validate(cfg, model, LFdataset, filtered_data=None, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)

