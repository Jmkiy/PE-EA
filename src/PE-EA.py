from re import S
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import math
import argparse
from collections import defaultdict
import argparse
from preprocessing import DBpDataset
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import copy
from count import read_list
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops
from torch_scatter import scatter_add

from torch_sparse import spmm
from torch_geometric.utils import softmax

class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(hidden, 1, bias=False)
        
    def forward(self, x, edge_index):
        fill_value = 1
        edge_weight=torch.ones((edge_index.size(1), ), dtype=None,
                                     device=edge_index.device)
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, x.size(0))

        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i+e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x

class Encoder(torch.nn.Module):
    def __init__(self, name, hiddens, heads, activation, feat_drop, attn_drop, negative_slope, bias):
        super(Encoder, self).__init__()
        self.name = name
        self.hiddens = hiddens
        self.heads = heads
        self.num_layers = len(hiddens) - 1
        self.gnn_layers = nn.ModuleList()
        self.activation = activation
        self.feat_drop = feat_drop
        self.highways = nn.ModuleList()
        self.gat = GAT(hiddens[-1])
        for l in range(0, self.num_layers):
            if self.name == "gcn-align":
                self.gnn_layers.append(
                    GCNAlign_GCNConv(in_channels=self.hiddens[l], out_channels=self.hiddens[l+1], improved=False, cached=False, bias=bias)
                )
            
            else:
                raise NotImplementedError("bad encoder name: " + self.name)
        if self.name == "naea":
            self.weight = Parameter(torch.Tensor(self.hiddens[0], self.hiddens[-1]))
            nn.init.xavier_normal_(self.weight)
        # if self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: extra parameters'''

    def forward(self, edges, x, r=None):
        edges = edges.t()
        
        for l in range(self.num_layers):
            x = F.dropout(x, p=self.feat_drop)
            x_ = self.gnn_layers[l](x, edges)
            x = x_
            if l != self.num_layers - 1:
                x = self.activation(x)
        return x            

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.name, "\n".join([layer.__repr__() for layer in self.gnn_layers]))

class GCNAlign_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GCNAlign_GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(1, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.mul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

def func_e(e, e_dict):
    if e_dict.get(e, None):
        funce = 1 / len(e_dict[e])
    else:
        funce = 0
    return funce

def rfunc(triple_list):
        head = {}
        tail = {} 
        rel_count = {} 
        r_mat_ind = [] 
        r_mat_val = [] 
        cleaned_triple_list = [(int(h), int(r), int(t)) for h, r, t in triple_list]
        for triple in cleaned_triple_list:
            r_mat_ind.append([triple[0], triple[2]])
            r_mat_val.append(triple[1])
            if triple[1] not in rel_count:
                rel_count[triple[1]] = 1
                head[triple[1]] = set()
                tail[triple[1]] = set()
                head[triple[1]].add(triple[0])
                tail[triple[1]].add(triple[2])
            else:
                rel_count[triple[1]] += 1
                head[triple[1]].add(triple[0])
                tail[triple[1]].add(triple[2])
        
        return head, tail

def get_relation_dict(relation_triplet):
    relation_dict = {}
    r_id = set()
    for h, r, t in relation_triplet:
        r_id.add(int(r))
        if r not in relation_dict:
            relation_dict[r] = set()
        relation_dict[r].add((h, t))
    
    r_id = list(r_id)
    return r_id,relation_dict

def func_r(r_set, arg='second'):
    r_first = [i[0] for i in r_set]
    r_first = list(set(r_first))
    r_second = [i[1] for i in r_set]
    r_second = list(set(r_second))
    if arg == 'first':
        funcr = len(r_first) / len(r_set)
    else:
        funcr = len(r_second) / len(r_set)
    return funcr

class EAExplainer(torch.nn.Module):
    def __init__(self, model_name, G_dataset, test_indices, Lvec, Rvec, model, evaluator, split, splitr=0, lang='zh'):
        super(EAExplainer, self).__init__()
        self.model_name = model_name
        self.dist = nn.PairwiseDistance(p=2)
        self.split = split
        self.splitr = splitr
        self.lang = lang
        if lang == 'zh':
            e_embed = np.load('../save/dbp_z_e_gcnalign_ins.npy')
        
        self.G_dataset = G_dataset
        self.embed = torch.Tensor(e_embed)
        self.e_embed = self.embed
        self.e_sim =self.cosine_matrix(self.embed[:self.split], self.embed[self.split:])
        self.G_dataset = G_dataset
        self.r_embed = self.proxy_r()
        
        self.Lvec = Lvec
        self.Rvec = Rvec
        self.test_indices = test_indices
        self.args = args
        self.test_kgs = copy.deepcopy(self.G_dataset.kgs)
        self.test_kgs_no = copy.deepcopy(self.G_dataset.kgs)
        self.test_indices = test_indices
        self.test_pair = G_dataset.test_pair
        self.train_pair = G_dataset.train_pair
        self.model_pair = G_dataset.model_pair
        self.model_link = G_dataset.model_link
        self.train_link = G_dataset.train_link
        self.test_link = G_dataset.test_link
        self.evaluator = evaluator
        
        self.triple_list = self.G_dataset.kg1 + self.G_dataset.kg2
        self.head, self.tail = rfunc(self.triple_list) 
        r1,relation_dict1 = get_relation_dict(self.G_dataset.kg1)  
        r2,relation_dict2 = get_relation_dict(self.G_dataset.kg2)
        self.relations1 = r1
        self.relations2 = r2

        self.rt_dict_1, self.hr_dict_1 = self.generate_relation_triple_dict(self.G_dataset.kg1)
        self.rt_dict_2, self.hr_dict_2 = self.generate_relation_triple_dict(self.G_dataset.kg2)
        self.rt_dict_1 = {int(key): {(int(k), int(v)) for k, v in value} for key, value in self.rt_dict_1.items()}
        self.hr_dict_1 = {int(key): {(int(k), int(v)) for k, v in value} for key, value in self.hr_dict_1.items()}
        self.rt_dict_2 = {int(key): {(int(k), int(v)) for k, v in value} for key, value in self.rt_dict_2.items()}
        self.hr_dict_2 = {int(key): {(int(k), int(v)) for k, v in value} for key, value in self.hr_dict_2.items()}

        self.r_func1 = {}
        for key in relation_dict1:
            fun = func_r(relation_dict1[key], arg='first')
            ifun = func_r(relation_dict1[key], arg='second')
            self.r_func1[int(key)] = [fun, ifun]
        self.r_func2 = {}
        for key in relation_dict2:
            fun = func_r(relation_dict2[key], arg='first')
            ifun = func_r(relation_dict2[key], arg='second')
            self.r_func2[int(key)] = [fun, ifun]

    def proxy_r(self):
        r_list = defaultdict(list)

        for (h, r, t) in self.G_dataset.kg1:
            r_list[int(r)].append([int(h), int(t)])
        for (h, r, t) in self.G_dataset.kg2:
            r_list[int(r)].append([int(h), int(t)])
        
        r_embed = torch.Tensor(len(self.G_dataset.rel), self.embed.shape[1])
        for i in range(r_embed.shape[0]):
            cur_ent = torch.Tensor(r_list[i]).reshape(2,-1)
            h = self.embed[cur_ent[0].long()]
            t = self.embed[cur_ent[1].long()]
            r_embed[i] = (h - t).mean(dim=0)
        return r_embed

    def explain_EA(self, method, thred, num,  version = ''):
        self.version = version
        if method == 'PE-EA':
            #test_r = [self.relations1,self.relations2]
            self.version = version
            #prob_r, rsim_dict = self.pr_r(test_r, self.head, self.tail, self.embed)
            #torch.save(prob_r, 'prob_r_z.pt')
            
            self.rel1_index, self.rel2_index = {}, {}
            
            for i in range(len(self.relations1)):
                self.rel1_index[self.relations1[i]] = i
                self.rel1_index[self.relations1[i] + len(self.relations1) + len(self.relations2)] = i + len(self.relations1)
            for i in range(len(self.relations2)):
                self.rel2_index[self.relations2[i]] = i
                self.rel2_index[self.relations2[i] + len(self.relations1) + len(self.relations2)] = i + len(self.relations2)
            
            self.prob_r = torch.load('../save/prob_r_z.pt')
            
            if self.lang == 'zh':
                if self.version == 1:
                    with open('../datasets/dbp_z_e/exp_PE-EA', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_PE_EA(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_z_e/exp_PE-EA', str(version))

    def explain_PE_EA(self, e1, e2):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        cur_link = self.model_link
        pair = set()
        for cur in neigh1:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh2:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))

        new_pair=set()
        for p in pair:
            prob_e = self.pr_e([p[0], p[1]], self.embed, self.prob_r, self.r_func1, self.r_func2,self.rel1_index, self.rel2_index,3023)
            print(prob_e)
            if prob_e > 0.4:
                new_pair.add((p[0],p[1]))
        
        tri1_list = []
        tri2_list = []
        for p in new_pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            for tri in tri1:
                tri1_list.append(tri)
            for tri in tri2:
                tri2_list.append(tri)

        return tri1_list, tri2_list, pair
    
    def get_test_file_mask(self, file, thred, method=''):
        nec_tri = set()
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
                cur = cur.strip().split('\t')
                nec_tri.add((int(cur[0]), int(cur[1]), int(cur[2])))
        print('sparsity :', 1 - (len(nec_tri) / len(self.G_dataset.suff_kgs)))
        new_kg = self.G_dataset.kgs - nec_tri
        if self.lang == 'zh':
            with open('../datasets/dbp_z_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_z_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
   
    def init_1_hop(self, gid1, gid2):
        neigh1 = set()
        neigh2 = set()
        for cur in self.G_dataset.gid[gid1]:
            if cur[0] != int(gid1):
                neigh1.add(int(cur[0]))
            else:
                neigh1.add(int(cur[2]))
        for cur in self.G_dataset.gid[gid2]:
            if cur[0] != int(gid2):
                neigh2.add(int(cur[0]))
            else:
                neigh2.add(int(cur[2]))
        return neigh1, neigh2

    def search_1_hop_tri(self, source ,target):
        tri = set()
        for t in self.G_dataset.gid[source]:
            if t[0] == target or t[2] == target:
                tri.add((t[0], t[1], t[2]))
                continue

        return tri
    
    def cosine_matrix(self, A, B):
        A_sim = torch.mm(A, B.t())
        a = torch.norm(A, p=2, dim=-1)
        b = torch.norm(B, p=2, dim=-1)
        cos_sim = A_sim / a.unsqueeze(-1)
        cos_sim /= b.unsqueeze(-2)
        return cos_sim

    def generate_relation_triple_dict(self,tri_list):
        rt_dict, hr_dict = dict(), dict()
        for h, r, t in tri_list:
            rt_set = rt_dict.get(h, set())
            rt_set.add((r, t))
            rt_dict[h] = rt_set
            hr_set = hr_dict.get(t, set())
            hr_set.add((h, r))
            hr_dict[t] = hr_set
        # print("Number of rt_dict:", len(rt_dict))
        # print("Number of hr_dict:", len(hr_dict))
        return rt_dict, hr_dict
    
    def head_tail_cat(self, h_i, h_j, t_i, t_j, hi_num, hj_num, ti_num, tj_num):
        h_sim = (torch.nn.functional.cosine_similarity(h_i, h_j, dim=1) + 1.0) / 2.0  # 将[-1,1]范围转到[0,1]
        h_sim = h_sim.type(torch.float32).numpy().astype(np.float32)

        t_sim = (torch.nn.functional.cosine_similarity(t_i, t_j, dim=1) + 1.0) / 2.0
        t_sim = t_sim.type(torch.float32).numpy().astype(np.float32)
        
        return (h_sim + t_sim) / 2
    
    def pr_r(self, test_pair, head, tail, inlayer, rel_num=346):
        r2e = {}
        for ill in test_pair[0] + test_pair[1]: 
            r2e[ill] = [head.get(ill, set()), tail.get(ill, set())]
        rpairs = {}
        r_dict = {}

        p = 0
        for i in test_pair[0]:
            for j in test_pair[1]:
                h_i_e = torch.stack([inlayer[e] for e in r2e[i][0]])
                efunc_hi = torch.tensor([func_e(e, self.rt_dict_1) for e in r2e[i][0]]).unsqueeze(1)
                efunc_hi = efunc_hi / efunc_hi.sum()
                h_i = (efunc_hi * h_i_e).sum(dim=0, keepdim=True)
                t_i_e = torch.stack([inlayer[e] for e in r2e[i][1]])
                efunc_ti = torch.tensor([func_e(e, self.hr_dict_1) for e in r2e[i][1]]).unsqueeze(1)
                efunc_ti = efunc_ti / efunc_ti.sum()
                t_i = (efunc_ti * t_i_e).sum(dim=0, keepdim=True)  
                h_j_e = torch.stack([inlayer[e] for e in r2e[j][0]])
                efunc_hj = torch.tensor([func_e(e, self.rt_dict_2) for e in r2e[j][0]]).unsqueeze(1)
                efunc_hj = efunc_hj / efunc_hj.sum()
                h_j = (efunc_hj * h_j_e).sum(dim=0, keepdim=True)
                t_j_e = torch.stack([inlayer[e] for e in r2e[j][1]])
                efunc_tj = torch.tensor([func_e(e, self.hr_dict_2) for e in r2e[j][1]]).unsqueeze(1)
                efunc_tj = efunc_tj /efunc_tj.sum()
                t_j = (efunc_tj * t_j_e).sum(dim=0, keepdim=True)
                
                a1 = self.head_tail_cat(h_i, h_j, t_i, t_j, len(r2e[i][0]), len(r2e[j][0]), len(r2e[i][1]), len(r2e[j][1]))
                a2 = self.head_tail_cat(t_i, h_j, h_i, t_j, len(r2e[i][1]), len(r2e[j][0]), len(r2e[i][0]),
                                        len(r2e[j][1]))
                a3 = self.head_tail_cat(h_i, t_j, t_i, h_j, len(r2e[i][0]), len(r2e[j][1]), len(r2e[i][1]),
                                        len(r2e[j][0]))
                a4 = self.head_tail_cat(t_i, t_j, h_i, h_j, len(r2e[i][1]), len(r2e[j][0]), len(r2e[i][0]),
                                        len(r2e[j][1]))
                rpairs[(i, j)] = [a1, a2, a3, a4]
                
                r_dict[(i, j)], r_dict[i + rel_num, j], r_dict[i, j + rel_num], r_dict[
                    i + rel_num, j + rel_num] = a1, a2, a3, a4
                p = p + 1
                    
        coinc1 = []
        for row in test_pair[0]:
            list = []
            for col in test_pair[1]:
                list.append(rpairs[(row, col)][0])
            for col in test_pair[1]:
                list.append(rpairs[(row, col)][2])
            coinc1.append(list)
        coinc2 = []
        for row in test_pair[0]:
            list = []
            for col in test_pair[1]:
                list.append(rpairs[(row, col)][1])
            for col in test_pair[1]:
                list.append(rpairs[(row, col)][3])
            coinc2.append(list)
        
        coinc1 = torch.tensor(coinc1, dtype=torch.float32)
        coinc2 = torch.tensor(coinc2, dtype=torch.float32)
        coinc = torch.cat((coinc1, coinc2), dim=0)

        return coinc, r_dict
   
    def pr_e(self, test_pair, e_emb, rel_sim, r_func1, r_func2, rel_index1, rel_index2, rel_num=0): 
        x, y = int(test_pair[0]), int(test_pair[1]) 

        rt1 = self.rt_dict_1.get(x, set())
        hr1 = self.hr_dict_1.get(x, set())
        r1_out = [r[0] for r in rt1]
        r1_in = [r[1] + rel_num for r in hr1]
        
        rt1=list(rt1)
        e_emb = [torch_tensor.numpy() for torch_tensor in e_emb]
        
        r1_out_fun = np.array([r_func1[i[0]][1] for i in rt1]) 
        r1_in_fun = np.array([r_func1[i[1]][0] for i in hr1])
       
        e1_out = np.array([e_emb[e[1]] for e in rt1])
        e1_in = np.array([e_emb[e[0]] for e in hr1])
        
        if len(rt1) == 0 and len(hr1) == 0:
            return 0
        if len(rt1) == 0:
            r1 = r1_in
            r1_fun = r1_in_fun
            e1 = e1_in
        elif len(hr1) == 0:
            r1 = r1_out
            r1_fun = r1_out_fun
            e1 = e1_out
        else:
            r1 = r1_out + r1_in
            r1_fun =  np.concatenate((r1_out_fun, r1_in_fun))
            e1 =  np.concatenate((e1_out, e1_in))

        rt2 = self.rt_dict_2.get(y, set())
        hr2 = self.hr_dict_2.get(y, set())
        
        r2_out = [r[0] for r in rt2]
        r2_in = [r[1] + rel_num for r in hr2]
        r2_out_fun = np.array([r_func2[i[0]][1] for i in rt2])
        r2_in_fun = np.array([r_func2[i[1]][0] for i in hr2])
        e2_out = np.array([e_emb[e[1]] for e in rt2])
        e2_in = np.array([e_emb[e[0]] for e in hr2])
        
        if len(rt2) == 0 and len(hr2) == 0:
            return 0
        if len(rt2) == 0:
            r2 = r2_in
            r2_fun = r2_in_fun
            e2 = e2_in
        elif len(hr2) == 0:
            r2 = r2_out
            r2_fun = r2_out_fun
            e2 = e2_out
        else:
            r2 = r2_out + r2_in
            r2_fun =  np.concatenate((r2_out_fun, r2_in_fun))
            e2 =  np.concatenate((e2_out, e2_in))
        
        r_match = np.zeros((len(r1), len(r2)))
        for a in range(len(r1)):
            for b in range(len(r2)):
                aa, bb = rel_index1.get(r1[a]), rel_index2.get(r2[b])
                r_match[a, b] = rel_sim[aa, bb]
        rfun_match = np.matmul(r1_fun.reshape(-1, 1), r2_fun.reshape(1, -1))
        sim_mat = 1 - euclidean_distances(e1, e2)
        e_match = sim_mat.astype(np.float32)
        
        re_match = np.multiply(r_match, e_match)
        mask = (re_match.max(axis=0 if re_match.shape[0] > re_match.shape[1] else 1, keepdims=1) == re_match)
        re_match = mask * re_match
        thre_re = max(0.4, (np.max(re_match) + np.mean(re_match)) / 2)
        re_match = re_match * (re_match > thre_re) 
        re_match = np.multiply(re_match, rfun_match)
        numerator = sum(sum(re_match))
        denominator = sum(sum(np.multiply(rfun_match, mask)))
        return numerator / denominator
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(f'arguments for Explanation Generation')

    parser.add_argument('lang', type=str, help='which dataset', default='zh')
    parser.add_argument('method', type=str, help='Explanation Generation', default='repair')
    parser.add_argument('--version', type=int, help='the hop num of candidate neighbor', default=1)
    parser.add_argument('--num', type=str, help='the len of explanation', default=15)
    
    args = parser.parse_args()
    lang = args.lang
    method = args.method
    if args.version:
        version = args.version
    if args.num:
        num = args.num
    pair = '/GCNAlign_1000'


    device = 'cuda'
    if lang == 'zh':
        G_dataset = DBpDataset('../datasets/dbp_z_e/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_z_e/' + pair)


    Lvec = None
    Rvec = None
    model_name = 'mean_pooling'
    saved_model = None
    args = None
    in_d = None
    out_d = None
    m_adj=None
    e1=None
    e2=None
    device = 'cuda'
    model = None
    model_name = 'mean_pooling'

    if lang == 'zh':
        model = Encoder('gcn-align', [100,100,100], [1,1,1], activation=F.elu, feat_drop=0, attn_drop=0, negative_slope=0.2, bias=False)
        model.load_state_dict(torch.load('../saved_model/zh_model.pt'))
        model_name = 'load'
        split = len(read_list('../datasets/dbp_z_e/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_z_e/rel_dict1'))
    
    evaluator = None
    explain = EAExplainer(model_name, G_dataset, test_indices, Lvec, Rvec, model, evaluator, split, splitr, lang)
    explain.explain_EA(method,0.4, num, version)

