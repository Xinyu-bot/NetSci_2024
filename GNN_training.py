import copy
from typing import List
import warnings
import torch
import random
import os
import numpy as np

from tqdm import tqdm
from time import time
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    classification_report,
)
from Model_GAT import GAT
from Model_GCN import GCN
from Model_TAGNN import TAGNN
from Model_GraphSAGE import GraphSAGE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
from data_utils import prepare_data, load_df_data
from config import Config

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def resample(data, df_wallets_features, sampling_strategy, verbose=False):
    # Prepare the feature matrix and labels
    feature_columns = [
        col for col in df_wallets_features.columns if col not in ["address", "class"]
    ]
    data.x = torch.tensor(
        df_wallets_features[feature_columns].values, dtype=torch.float
    )
    data.y = torch.tensor(df_wallets_features["class"].values, dtype=torch.long)

    # Split the data into training, validation, and test sets
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    assert train_ratio + val_ratio + test_ratio == 1.0
    assert 0 < sampling_strategy <= 1

    # Split indices into training and temp sets
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        np.arange(len(data.y)),
        data.y,
        stratify=data.y,
        test_size=(1 - train_ratio),
        random_state=42,
    )

    # Split temp set into validation and test sets
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx,
        temp_labels,
        stratify=temp_labels,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=42,
    )

    # Create boolean masks
    train_mask = torch.zeros(len(data.y), dtype=torch.bool)
    val_mask = torch.zeros(len(data.y), dtype=torch.bool)
    test_mask = torch.zeros(len(data.y), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Apply SMOTE to handle class imbalance only on the training set
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        data.x[train_mask].numpy(), data.y[train_mask].numpy()
    )

    # Create new tensors for resampled training features and labels
    new_train_features = torch.tensor(X_resampled, dtype=torch.float)
    new_train_labels = torch.tensor(y_resampled, dtype=torch.long)

    # Combine resampled training data with the original validation and test data
    combined_features = torch.cat(
        (new_train_features, data.x[val_mask], data.x[test_mask]), dim=0
    )
    combined_labels = torch.cat(
        (new_train_labels, data.y[val_mask], data.y[test_mask]), dim=0
    )

    # Update indices for the combined dataset
    new_train_idx = torch.arange(len(new_train_labels))
    new_val_idx = torch.arange(
        len(new_train_labels), len(new_train_labels) + len(data.y[val_mask])
    )
    new_test_idx = torch.arange(
        len(new_train_labels) + len(data.y[val_mask]), len(combined_labels)
    )

    # Create new boolean masks for the combined dataset
    new_train_mask = torch.zeros(len(combined_labels), dtype=torch.bool)
    new_val_mask = torch.zeros_like(new_train_mask)
    new_test_mask = torch.zeros_like(new_train_mask)

    new_train_mask[new_train_idx] = True
    new_val_mask[new_val_idx] = True
    new_test_mask[new_test_idx] = True

    # Update the data object
    data.x = combined_features
    data.y = combined_labels
    data.train_mask = new_train_mask
    data.val_mask = new_val_mask
    data.test_mask = new_test_mask

    # Verify the class distribution in the new training set
    if verbose:
        print(
            "Class distribution in the new training set:",
            data.y[data.train_mask].bincount().cpu().numpy(),
        )
        print(
            "Class distribution in validation set:",
            data.y[data.val_mask].bincount().cpu().numpy(),
        )
        print(
            "Class distribution in test set:",
            data.y[data.test_mask].bincount().cpu().numpy(),
        )

    data = data.to("cuda")

    return data


def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(mask, model, data, _):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[mask].max(dim=1)[1]
        true_labels = data.y[mask].cpu()
        pred_labels = pred.cpu()
        recall = recall_score(true_labels, pred_labels, pos_label=0)
        precision = precision_score(true_labels, pred_labels, pos_label=0)
        f1 = f1_score(true_labels, pred_labels, pos_label=0)
    return recall, precision, f1


def setup_model(
    model_class,
    input_channels,
    hidden_channels_1,
    hidden_channels_2,
    output_channels,
    dropout,
    learning_rate,
    weight_decay,
    gat_heads=1,
    data=None,
):
    # dynamically create the model based on the model_class
    if model_class == GAT:
        model = GAT(
            input_channels,
            hidden_channels_1,
            hidden_channels_2,
            output_channels,
            dropout,
            heads=gat_heads,
        )
    else:
        model = model_class(
            input_channels,
            hidden_channels_1,
            hidden_channels_2,
            output_channels,
            dropout,
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if data is None:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # adjust the loss function to give more importance to the minority class
        # recount the classes after augmenting
        augmented_class_counts = data.y[data.train_mask].bincount().cpu().numpy()

        # adjust the loss function to give more importance to the minority class
        class_weights = torch.tensor(
            [augmented_class_counts[1] / augmented_class_counts[0], 1.0],
            dtype=torch.float,
        ).to("cuda")
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = model.to("cuda")
    criterion = criterion.to("cuda")

    return model, optimizer, criterion


def train_model(model, data, optimizer, loss_fn, epochs, verbose=False):
    best_f1 = 0
    best_model_iter = 0
    best_model = None
    class_counts = data.y[data.train_mask].bincount().cpu().numpy()
    total_samples = sum(class_counts)
    class_weights = class_counts / total_samples

    if verbose:
        for epoch in range(epochs):
            loss = train(model, data, optimizer, loss_fn)
            _, _, train_f1 = evaluate(data.train_mask, model, data, class_weights)
            _, _, val_f1 = evaluate(data.val_mask, model, data, class_weights)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_iter = epoch
                best_model = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch: {epoch+1}, Loss: {loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}"
                )
    else:
        for epoch in tqdm(
            range(epochs), desc="Training model", total=epochs, leave=False
        ):
            loss = train(model, data, optimizer, loss_fn)
            _, _, train_f1 = evaluate(data.train_mask, model, data, class_weights)
            _, _, val_f1 = evaluate(data.val_mask, model, data, class_weights)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_iter = epoch
                best_model = copy.deepcopy(model.state_dict())

    if best_model is not None:
        if verbose:
            print(f"Loading the best model at iteration {best_model_iter}")
        model.load_state_dict(best_model)

    return model


def test_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[data.test_mask].max(dim=1)[1]
        correct = pred.eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
        # print(f"Test Accuracy: {acc:.4f}")
        # print(confusion_matrix(data.y[data.test_mask].cpu(), pred.cpu()))
        c = classification_report(data.y[data.test_mask].cpu(), pred.cpu())

    return acc, c


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_configs():
    hidden_channels_1 = [16, 32, 64]
    hidden_channels_2 = [4, 8, 16]
    sampling_strategy = [0.5, 0.7, 0.9]
    gat_heads = [2, 4, 8]

    configs = []
    for h1 in hidden_channels_1:
        for h2 in hidden_channels_2:
            for ss in sampling_strategy:
                for gh in gat_heads:
                    config = Config(h1, h2, ss, gh)
                    configs.append(config)

    return configs


def sort_results(GCN_results, GAT_results, TAGNN_results, SAGE_results):
    GCN_results = dict(
        sorted(
            GCN_results.items(),
            key=lambda x: (x[1]["accuracy"], x[1]["f1"]),
            reverse=True,
        )
    )
    GAT_results = dict(
        sorted(
            GAT_results.items(),
            key=lambda x: (x[1]["accuracy"], x[1]["f1"]),
            reverse=True,
        )
    )
    TAGNN_results = dict(
        sorted(
            TAGNN_results.items(),
            key=lambda x: (x[1]["accuracy"], x[1]["f1"]),
            reverse=True,
        )
    )
    SAGE_results = dict(
        sorted(
            SAGE_results.items(),
            key=lambda x: (x[1]["accuracy"], x[1]["f1"]),
            reverse=True,
        )
    )

    return GCN_results, GAT_results, TAGNN_results, SAGE_results


def single_run():
    # check if data.pt exists
    if not os.path.exists("./data.pt"):
        print("Data.pt file not found. Preparing data...")
        data_og, df_wallets_features, _ = prepare_data()
    else:
        print("Data.pt file found. Loading data...")
        data_og = torch.load("./data.pt")
        df_wallets_features, _ = load_df_data()

    # fixed hyperparameters
    EPOCH = 300
    INPUT_CHANNELS = 55
    OUTPUT_CHANNELS = 2
    WEIGHT_DECAY = 5e-4
    SEED = 3407

    # tunable hyperparameters
    LEARNING_RATE = 0.01
    DROPOUT = 0.5
    GAT_HEADS = 4
    HIDDEN_CHANNELS_1 = 16
    HIDDEN_CHANNELS_2 = 4
    SAMPLING_STRATEGY = 0.5

    cls = GraphSAGE  # one of {GAT, GCN, TAGNN, GraphSAGE}

    set_seed(SEED)
    model, optimizer, loss_fn = setup_model(
        cls,
        INPUT_CHANNELS,
        HIDDEN_CHANNELS_1,
        HIDDEN_CHANNELS_2,
        OUTPUT_CHANNELS,
        DROPOUT,
        LEARNING_RATE,
        WEIGHT_DECAY,
        gat_heads=GAT_HEADS,
    )

    set_seed(SEED)
    data = resample(data_og, df_wallets_features, SAMPLING_STRATEGY, verbose=False)

    set_seed(SEED)
    t = time()
    train_model(model, data, optimizer, loss_fn, EPOCH)
    print(f"Training time: {time()-t:.2f}s")
    acc, mat = test_model(model, data)

    print(f"Accuracy: {acc}")
    print(mat)


def main():
    # check if data.pt exists
    if not os.path.exists("./data.pt"):
        print("Data.pt file not found. Preparing data...")
        data_og, df_wallets_features, _ = prepare_data()
    else:
        print("Data.pt file found. Loading data...")
        data_og = torch.load("./data.pt")
        df_wallets_features, _ = load_df_data()

    # fixed hyperparameters
    EPOCH = 300
    INPUT_CHANNELS = 55
    OUTPUT_CHANNELS = 2
    WEIGHT_DECAY = 5e-4
    SEED = 3407

    # tunable hyperparameters
    LEARNING_RATE = 0.01
    DROPOUT = 0.5
    GAT_HEADS = 2
    HIDDEN_CHANNELS_1_GCN, HIDDEN_CHANNELS_1_GAT, HIDDEN_CHANNELS_1_TAGNN = 32, 32, 32
    HIDDEN_CHANNELS_2_GCN, HIDDEN_CHANNELS_2_GAT, HIDDEN_CHANNELS_2_TAGNN = 16, 16, 16
    SAMPLING_STRATEGY = 0.7

    configs: List[Config] = get_configs()
    GCN_results, GAT_results, TAGNN_results, SAGE_results = {}, {}, {}, {}

    run_gcn, run_gat, run_tagnn, run_sage = False, True, False, False

    print(
        f"Running GCN: {run_gcn}, Running GAT: {run_gat}, Running TAGNN: {run_tagnn}, Running SAGE: {run_sage}"
    )
    for config in tqdm(configs, desc="Exploring hyperparameters", total=len(configs)):
        HIDDEN_CHANNELS_1_GCN = config.HIDDEN_CHANNELS_1_GCN
        HIDDEN_CHANNELS_1_GAT = config.HIDDEN_CHANNELS_1_GCN
        HIDDEN_CHANNELS_1_TAGNN = config.HIDDEN_CHANNELS_1_GCN
        HIDDEN_CHANNELS_2_GCN = config.HIDDEN_CHANNELS_2_GCN
        HIDDEN_CHANNELS_2_GAT = config.HIDDEN_CHANNELS_2_GCN
        HIDDEN_CHANNELS_2_TAGNN = config.HIDDEN_CHANNELS_2_GCN
        SAMPLING_STRATEGY = config.SAMPLING_STRATEGY
        GAT_HEADS = config.GAT_HEADS

        # clone the data object to prevent data corruption
        data_clone = copy.deepcopy(data_og)
        set_seed(SEED)
        data = resample(
            data_clone, df_wallets_features, SAMPLING_STRATEGY, verbose=False
        )

        set_seed(SEED)
        gcn_model, gcn_optimizer, gcn_loss_fn = setup_model(
            GCN,
            INPUT_CHANNELS,
            HIDDEN_CHANNELS_1_GCN,
            HIDDEN_CHANNELS_2_GCN,
            OUTPUT_CHANNELS,
            DROPOUT,
            LEARNING_RATE,
            WEIGHT_DECAY,
            # data=data,
        )
        set_seed(SEED)
        gat_model, gat_optimizer, gat_loss_fn = setup_model(
            GAT,
            INPUT_CHANNELS,
            HIDDEN_CHANNELS_1_GAT,
            HIDDEN_CHANNELS_2_GAT,
            OUTPUT_CHANNELS,
            DROPOUT,
            LEARNING_RATE,
            WEIGHT_DECAY,
            GAT_HEADS,
            # data=data,
        )
        set_seed(SEED)
        tagnn_model, tagnn_optimizer, tagnn_loss_fn = setup_model(
            TAGNN,
            INPUT_CHANNELS,
            HIDDEN_CHANNELS_1_TAGNN,
            HIDDEN_CHANNELS_2_TAGNN,
            OUTPUT_CHANNELS,
            DROPOUT,
            LEARNING_RATE,
            WEIGHT_DECAY,
            # data=data,
        )
        set_seed(SEED)
        sage_model, sage_optimizer, sage_loss_fn = setup_model(
            GraphSAGE,
            INPUT_CHANNELS,
            HIDDEN_CHANNELS_1_TAGNN,
            HIDDEN_CHANNELS_2_TAGNN,
            OUTPUT_CHANNELS,
            DROPOUT,
            LEARNING_RATE,
            WEIGHT_DECAY,
            # data=data,
        )

        if run_gcn:
            set_seed(SEED)
            train_model(gcn_model, data, gcn_optimizer, gcn_loss_fn, EPOCH)
            gcn_acc, gcn_mat = test_model(gcn_model, data)
            gcn_f1 = float(gcn_mat.split("\n")[2].split()[3])
            GCN_results[config.as_key()] = {"accuracy": gcn_acc, "f1": gcn_f1}

        if run_gat:
            set_seed(SEED)
            train_model(gat_model, data, gat_optimizer, gat_loss_fn, EPOCH)
            gat_acc, gat_mat = test_model(gat_model, data)
            gat_f1 = float(gat_mat.split("\n")[2].split()[3])
            GAT_results[config.as_key()] = {"accuracy": gat_acc, "f1": gat_f1}

        if run_tagnn:
            set_seed(SEED)
            train_model(tagnn_model, data, tagnn_optimizer, tagnn_loss_fn, EPOCH)
            tagnn_acc, tagnn_mat = test_model(tagnn_model, data)
            tagnn_f1 = float(tagnn_mat.split("\n")[2].split()[3])
            TAGNN_results[config.as_key()] = {"accuracy": tagnn_acc, "f1": tagnn_f1}

        if run_sage:
            set_seed(SEED)
            train_model(sage_model, data, sage_optimizer, sage_loss_fn, EPOCH)
            sage_acc, sage_mat = test_model(sage_model, data)
            sage_f1 = float(sage_mat.split("\n")[2].split()[3])
            SAGE_results[config.as_key()] = {"accuracy": sage_acc, "f1": sage_f1}

    print("Sorting results...")
    # sort the results by accuracy first then by f1 score, highest to lowest
    GCN_results, GAT_results, TAGNN_results, SAGE_results = sort_results(
        GCN_results, GAT_results, TAGNN_results, SAGE_results
    )

    print("Saving results...")
    # save results to files
    with open("results_GCN.txt", "w") as f:
        f.write(str(GCN_results))
    with open("results_GAT.txt", "w") as f:
        f.write(str(GAT_results))
    with open("results_TAGNN.txt", "w") as f:
        f.write(str(TAGNN_results))
    with open("results_SAGE.txt", "w") as f:
        f.write(str(SAGE_results))

    print("Done!")


if __name__ == "__main__":
    # main()
    single_run()
