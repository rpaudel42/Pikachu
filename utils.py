# ******************************************************************************
# utils.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 3/4/21   Paudel     Initial version,
# ******************************************************************************
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
from timeit import default_timer as timer
import random

class GraphUtils:
    def __init__(self, node_map):
        self.node_map = node_map
        pass

    def embedding_hadamard(self, u, v):
        return u * v

    def embedding_l1(self, u, v):
        return np.abs(u - v)

    def embedding_l2(self, u, v):
        return (u - v) ** 2

    def embedding_avg(self, u, v):
        return (u + v) / 2.0

    def create_graph(self, snapshot_df):
        G = nx.MultiGraph()
        anom_node = []
        for index, row in snapshot_df.iterrows():
            scomp = row.src_computer
            dcomp = row.dst_computer
            # host_name = row.src_user #row.host_name
            time = index #row.timestamp
            gid = row.snapshot
            is_anomaly = False
            if row.label == 1:
                # print(row)
                is_anomaly = True
                if scomp not in anom_node:
                    anom_node.append(scomp)
                if dcomp not in anom_node:
                    anom_node.append(dcomp)

            G.add_node(self.node_map[scomp], anom= (scomp in anom_node))
            G.add_node(self.node_map[dcomp], anom= (dcomp in anom_node))
            G.add_edge(self.node_map[scomp], self.node_map[dcomp], time=time, anom=is_anomaly, snapshot = gid, weight=1)
        # print("Auth N: %d E: %d \n" % (G.number_of_nodes(), G.number_of_edges()))
        return G

class DataUtils:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def get_node_map(self, data_df):
        print("... Generating Node Map ... \n")
        node_map = {}
        node_id = 0
        for index, row in tqdm(data_df.iterrows()):
            scomp = row.src_computer
            dcomp = row.dst_computer
            if scomp not in node_map:
                node_map[scomp] = node_id
                node_id += 1
            if dcomp not in node_map:
                node_map[dcomp] = node_id
                node_id += 1
        return node_map

    def get_data(self):
        data_df = pd.read_csv(self.data_folder, header=0)
        data_df = data_df[(data_df['snapshot'] >= 65) & (data_df['snapshot'] <= 75)]
        data_df.to_csv('dataset/optc/sample.csv')
        node_df = data_df[['src_computer', 'dst_computer']]
        node_df = node_df.drop_duplicates()
        node_map = self.get_node_map(node_df)
        return data_df, node_map

    def preprocess_lanl_data(self):
        auth_df = pd.DataFrame()
        for chunk_df in tqdm(pd.read_csv(self.data_folder, usecols=['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'auth_type', 'logon_type'], dtype={'timestamp': np.int32, 'src_user': str, 'dst_user': str,
                                     'src_computer': str, 'dst_computer': str, 'auth_type': 'category',
                                     'logon_type': 'category'}, chunksize=10000)):
            chunk_df = chunk_df[~((chunk_df['auth_type'] == '?') | (chunk_df['logon_type'] == '?'))]
            chunk_df = chunk_df[~((chunk_df['src_user'].str.contains(r'ANONYMOUS(?!$)')) | (
                chunk_df['src_user'].str.contains(r'LOCAL(?!$)')) | (chunk_df['src_user'].str.contains(r'NETWORK(?!$)')) | (
                chunk_df['src_user'].str.contains(r'ADMIN(?!$)')))]
            chunk_df = chunk_df[chunk_df['src_computer'] != chunk_df['dst_computer']]
            chunk_df = chunk_df.drop(['dst_user', 'auth_type', 'logon_type'], axis=1).reset_index(
                drop=True)
            auth_df = auth_df.append(chunk_df, ignore_index=True)

        rt_df = pd.read_csv('dataset/lanl/redteam.txt', header=0)
        rt_df.columns = ['timestamp', 'src_user', 'src_computer', 'dst_computer']
        filter_col_name = ['timestamp', 'src_user', 'src_computer', 'dst_computer']  # rt_df.columns.tolist()
        comm_df = pd.merge(auth_df.reset_index(), rt_df.reset_index(), how='inner', on=filter_col_name)
        # print("Anomalous rows: \n", comm_df)

        anom_row_index = comm_df.index_x.to_list()
        # print("Anom rows index: ", anom_row_index)

        # label row as anom or norm
        auth_df['label'] = 0
        auth_df.loc[anom_row_index, 'label'] = 1
        initial_time = min(auth_df['timestamp'])
        auth_df['delta'] = auth_df['timestamp'] - initial_time
        auth_df['snapshot'] = auth_df['delta'].sec // 3600
        auth_df = auth_df.drop(['delta'], axis=1).reset_index(
            drop=True)
        auth_df.to_csv("dataset/lanl/auth_all_anom_1hr.csv")
        self.get_data()

    def lanl_user_subset(self):
        lanl_df = pd.read_csv(self.data_folder, header=0, index_col=0, dtype={'timestamp': np.int32, 'src_user': str, 'src_computer': str, 'dst_computer': str, 'label': np.bool,
                                     'snapshot': int})
        anom_user_df = lanl_df[lanl_df['label'] == 1]
        anom_row_index = anom_user_df.index.to_list()
        print("Total Anom Edges: ", len(anom_row_index))
        print("Anom rows index: ", anom_row_index)

        anom_user = list(set(lanl_df.loc[anom_row_index, 'src_user'].tolist()))
        print("Anomalous Users: ", len(anom_user), anom_user)
        all_user = lanl_df.src_user.unique()
        print("total users: ", len(all_user))

        # anom_user = ['U748@DOM1', 'U1723@DOM1', 'U636@DOM1', 'U6115@DOM1', 'U620@DOM1']#, 'U737@DOM1', 'U825@DOM1', 'U1653@DOM1', 'U293@DOM1',

        norm_users = np.setdiff1d(all_user, anom_user).tolist()
        print("Norm users: ", len(norm_users))
        norm_users = random.sample(norm_users, len(anom_user) * 2)
        all_users = norm_users + anom_user
        print("all users: ", len(all_users), all_users)
        all_user_df = lanl_df[lanl_df['src_user'].isin(all_users)]
        all_user_df.to_csv("dataset/lanl/anom_full_2xuser_1hr.csv")

    def get_node_label(graphs, node_list):
        node_labels = []
        for G in graphs:
            label = np.zeros((len(node_list), 1), dtype=np.float32)
            for n, data in G.nodes(data=True):
                label[node_list.index(str(n))] = data['anom']
            node_labels.append(label)
        node_labels = np.array(node_labels)
        # print("Node Label: ", node_labels.shape)
        return node_labels

    def get_node(node_map, n):
        for k, v in node_map.items():
            if v == n:
                return k
        return None

    def generate_seq_lookback(static_emb, lookback):
        X_train = []
        for sample in range(static_emb.shape[0]):
            for i in range(static_emb.shape[1] - lookback + 1):
                X_train.append(static_emb[sample, i:i + lookback, :])
        return np.array(X_train, dtype=np.float32)


    def generate_seq(static_emb):
        X_train, Y_train, = [], []
        for sample in range(static_emb.shape[0]):
            for i in range(static_emb.shape[1]):
                X_train.append(static_emb[sample, i, :])
        X_train = np.array(X_train, dtype=np.float32)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        return np.array(X_train, dtype=np.float32)


    def rollback_seq(dynamic_emb, total_node, batch_size):
        graph_emb = []
        for node_idx in range(total_node):
            offset = (node_idx + 1) * batch_size
            node_emb = dynamic_emb[node_idx*batch_size:offset,:]
            graph_emb.append(node_emb)
        return np.array(graph_emb, dtype=np.float32)
