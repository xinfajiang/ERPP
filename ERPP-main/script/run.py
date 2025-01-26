import os
import sys
import math
import pprint
from itertools import islice
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ERPP import tasks, util
from ERPP.models import ERPP
from ERPP.data import KnowledgeGraph
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",  type=int, default=1024)
    parser.add_argument("--use_gpus", action="store_true", default=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("-dataset", type=str, default="ENTITY")
    parser.add_argument("--epochs", type=list, default=[15, 10, 10, 10, 10])
    parser.add_argument("--hypas", type=list, default=[0, 0.1, 0.1, 0.1, 0.1])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_negative", type=int, default=256)
    parser.add_argument("--strict_negative", action="store_true", default=True)
    parser.add_argument("--adversarial_temperature", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layer_num", type=int, default=6)
    parser.add_argument("--message_func", type=str, default="distmult")
    parser.add_argument("--aggregate_func", type=str, default="sum")
    parser.add_argument("--layer_norm", action="store_true", default=True)
    parser.add_argument("--dependent", action="store_true", default=True)
    parser.add_argument("--project_relations", action="store_true", default=True)
    args = parser.parse_args()
    args = vars(args)

    return args


def train_and_test(args, model, ERPP_dataset, device, logger, batch_per_epoch=None):

    world_size = util.get_world_size()
    rank = util.get_rank()

    relation_inv = torch.tensor([ERPP_dataset.relation2inv[key] for key in range(len(ERPP_dataset.relation2inv))]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    else:
        parallel_model = model
    train_epoch = args['epochs']
    for snap_num in range(5):
        if rank == 0:
            print('--------------------------------')
            print("start: ", snap_num)
        best_result = torch.tensor(float("-inf"), device=device)
        best_epoch = torch.tensor(-1, device=device)
        train_triplets = torch.cat([ERPP_dataset.snapshots[snap_num].ent_graph.edge_index, ERPP_dataset.snapshots[snap_num].ent_graph.edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
        train_loader = torch_data.DataLoader(train_triplets, args['batch_size'], sampler=sampler)
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
                batch = tasks.negative_sampling(ERPP_dataset.snapshots[snap_num].ent_graph, batch, args['num_negative'],
                                                    strict=args['strict_negative'])
                pred, reloss = parallel_model(ERPP_dataset.snapshots[snap_num].ent_graph, 0, batch, relation_inv,snap_num)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if args['adversarial_temperature'] > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / args['adversarial_temperature'], dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / args['num_negative']
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
            result = test(args, snap_num, snap_num, model, ERPP_dataset, relation_inv, device=device, logger=logger)
            if result> best_result:
                best_result = result
                best_epoch = epoch

        
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
        state = torch.load("save/" + str(snap_num) + "/" + "model_epoch_%d.pth"%best_epoch, map_location=device)
        model.load_state_dict(state["model"])

        print("end: ", snap_num)
        for test_num in range(snap_num+1):
            if rank == 0:
                print("start: ", test_num, 'test')
            _ = test(args, test_num, test_num, model, ERPP_dataset, relation_inv, device=device, logger=logger)

        if snap_num+1 < 5:
            model.Inherit(snap_num)
            optimizer = optim.Adam(model.parameters(), lr=args['lr'])
            
        if rank == 0:
            print('------------------------------')
    

@torch.no_grad()
def test(args, test_num, snap_num, model, ERPP_dataset, relation_idv,device, logger, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([ERPP_dataset.snapshots[test_num].test_edge_index, ERPP_dataset.snapshots[test_num].test_edge_type]).t()
    test_loader = torch_data.DataLoader(test_triplets, args['batch_size'], sampler=None, shuffle=False)
    
    model.eval()
    rankings = []
    s = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []  
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(ERPP_dataset.snapshots[test_num].ent_graph_all, batch)
        t_pred, _ = model(ERPP_dataset.snapshots[test_num].ent_graph_all, 0, t_batch, relation_idv,test_num)
        h_pred, _ = model(ERPP_dataset.snapshots[test_num].ent_graph_all, 0, h_batch, relation_idv,test_num)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(ERPP_dataset.snapshots[test_num].filter_graph, batch)
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
    args = util.parse_args()
    torch.manual_seed(args.seed + util.get_rank())
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
    device = util.get_device(args)
    ERPP_dataset = KnowledgeGraph(args, device)
    model = ERPP(device=device, dataset = ERPP_dataset,
                args=args, 
    )
    model = model.to(device)
    train_and_test(args, model, ERPP_dataset, filtered_data=None, device=device, logger=logger)

