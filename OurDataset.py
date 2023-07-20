import pickle
from pathlib import Path

import torch
from torch_geometric.data import Dataset, Data
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import networkx as nx
import numpy as np
import os.path as osp


class OurDataset(Dataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(OurDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["parsed_data.pkl", "parsed_data_nm2_gender.pkl"]

    @property
    def processed_file_names(self):
        return list((Path(self.root) / "processed").iterdir())

    def download(self):
        # Download to `self.raw_dir`.
        return

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def process(self):
        # Read data into huge `Data` list.
        for i, graph in enumerate(self.read_data()):
            torch.save(graph, osp.join(self.processed_dir, f'data_{i}.pt'))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def read_data(self):

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
            print("save " + str(i) + " graph")
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

            data = Data(x=torch.from_numpy(corr).float(), edge_index=edge_index, y=torch.from_numpy(np.array(y)).long(),
                        edge_attr=edge_att.float(), pos=torch.from_numpy(np.diag(np.ones(num_nodes))).float())

            data.x[data.x == float('inf')] = 0
            yield data
