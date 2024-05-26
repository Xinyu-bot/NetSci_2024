import os
import torch

import pandas as pd
import networkx as nx

from torch_geometric.utils import from_networkx
from tqdm import tqdm


def read_data():
    df_addr_addr = pd.read_csv("data/AddrAddr_edgelist.csv")
    df_wallets_features = pd.read_csv("data/wallets_features_classes_combined.csv")
    return df_addr_addr, df_wallets_features


def preprocessing(df_wallets_features, df_addr_addr):
    # drop all rows with class = 3
    df_wallets_features = df_wallets_features[df_wallets_features["class"] != 3]

    # change class 1 to 0 and class 2 to 1 - this is needed for binary classification
    df_wallets_features.loc[:, "class"] = df_wallets_features["class"].apply(
        lambda x: 0 if x == 1 else 1
    )

    # order by Time step from smallest to largest
    df_wallets_features = df_wallets_features.sort_values("Time step", ascending=True)

    # drop Time step
    df_wallets_features.drop("Time step", axis=1, inplace=True)

    # remove reoccurences, keep the last one
    df_wallets_features = df_wallets_features.drop_duplicates(
        subset="address", keep="last"
    )
    df_wallets_features = df_wallets_features.reset_index(drop=True)

    # we only want to keep the edges if both input and output addresses are in the wallet features
    df_addr_addr = df_addr_addr[
        df_addr_addr["input_address"].isin(df_wallets_features["address"])
    ]
    df_addr_addr = df_addr_addr[
        df_addr_addr["output_address"].isin(df_wallets_features["address"])
    ]

    # normalized all columns except the address and class (index 0 and 1)
    df_wallets_features.iloc[:, 2:] = df_wallets_features.iloc[:, 2:].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    return df_wallets_features, df_addr_addr


def create_graph(df_addr_addr, df_wallets_features):
    # create a graph from the edges dataframe
    G = nx.from_pandas_edgelist(df_addr_addr, "input_address", "output_address")

    # add isolated nodes to the graph
    for address in df_wallets_features["address"]:
        if address not in G:
            G.add_node(address)

    print("Graph loaded.")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    return G


def construct_data_from_graph(G, df_wallets_features):
    # add node features and class labels to the graph
    for _, row in tqdm(
        df_wallets_features.iterrows(),
        total=len(df_wallets_features),
        desc="Adding node features",
    ):
        G.nodes[row["address"]].update(row.to_dict())
    # convert the NetworkX graph to PyTorch Geometric data
    data = from_networkx(G)
    # save the data to a file if file does not exist
    if not os.path.exists("./data.pt"):
        torch.save(data, "./data.pt")

    return data


def prepare_data():
    df_addr_addr, df_wallets_features = read_data()
    df_wallets_features, df_addr_addr = preprocessing(df_wallets_features, df_addr_addr)
    G = create_graph(df_addr_addr, df_wallets_features)
    data = construct_data_from_graph(G, df_wallets_features)
    return data, df_wallets_features, df_addr_addr


def load_df_data():
    df_addr_addr, df_wallets_features = read_data()
    df_wallets_features, df_addr_addr = preprocessing(df_wallets_features, df_addr_addr)
    return df_wallets_features, df_addr_addr
