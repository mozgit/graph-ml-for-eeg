import pickle

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import networkx as nx
import numpy as np
import os.path as osp


class OurDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(OurDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["parsed_data.pkl", "parsed_data_nm2_gender.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices = self.read_data()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def split(self, data, batch):
        node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
        node_slice = torch.cat([torch.tensor([0]), node_slice])

        row, _ = data.edge_index
        edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
        edge_slice = torch.cat([torch.tensor([0]), edge_slice])

        # Edge indices should start at zero for every graph.
        data.edge_index -= node_slice[batch[row]].unsqueeze(0)

        slices = {'edge_index': edge_slice}
        if data.x is not None:
            slices['x'] = node_slice
        if data.edge_attr is not None:
            slices['edge_attr'] = edge_slice
        if data.y is not None:
            if data.y.size(0) == batch.size(0):
                slices['y'] = node_slice
            else:
                slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
        if data.pos is not None:
            slices['pos'] = node_slice

        return data, slices

    def read_data(self):
        batch = []
        pseudo = []
        y_list = []
        edge_att_list, edge_index_list, att_list = [], [], []

        file = open(self.raw_dir + "/parsed_data.pkl", 'rb')
        corrs = pickle.load(file)
        file.close()

        file = open(self.raw_dir + "/parsed_data_nm2_gender.pkl", 'rb')
        pcorrs = pickle.load(file)
        file.close()

        for i, t in enumerate(zip(corrs, pcorrs)):
            corr, pcorr = t
            assert corr["gender"] == pcorr["gender"]
            y = corr["gender"]
            corr, pcorr = np.array(corr['adjustency_matrix']), np.array(pcorr['adjacency'])

            num_nodes = pcorr.shape[0]

            # По идее это все делается через torch_geometric.utils import dense_to_sparse но надо проверить
            G = nx.from_numpy_array(pcorr)
            A = nx.to_scipy_sparse_array(G)
            adj = A.tocoo()
            edge_att = np.zeros(len(adj.row))
            for j in range(len(adj.row)):
                edge_att[j] = pcorr[adj.row[j], adj.col[j]]

            edge_index = np.stack([adj.row, adj.col])
            edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
            edge_index = edge_index.long()
            edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                            num_nodes)

            # =====
            edge_att_list.append(edge_att.data.numpy())
            edge_index_list.append(edge_index.data.numpy() + i * num_nodes)
            att_list.append(corr)
            y_list.append(np.array(y))
            batch.append([i] * num_nodes)
            pseudo.append(np.diag(np.ones(num_nodes)))

        edge_att_arr = np.concatenate(edge_att_list)
        edge_index_arr = np.concatenate(edge_index_list, axis=1)
        att_arr = np.concatenate(att_list, axis=0)
        pseudo_arr = np.concatenate(pseudo, axis=0)
        y_arr = np.stack(y_list)
        edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
        att_torch = torch.from_numpy(att_arr).float()
        y_torch = torch.from_numpy(y_arr).long()  # classification
        batch_torch = torch.from_numpy(np.hstack(batch)).long()
        edge_index_torch = torch.from_numpy(edge_index_arr).long()
        pseudo_torch = torch.from_numpy(pseudo_arr).float()
        data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos=pseudo_torch)

        data, slices = self.split(data, batch_torch)

        return data, slices
