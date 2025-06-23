from ast import literal_eval
from argparse import ArgumentParser
import json
from itertools import cycle, islice
from tqdm import tqdm
import random
from pathlib import Path
import pickle as pkl
from numpy import percentile
import torch
from models.gnn.graph_encoder import GraphEncoder
from typing import Optional
from torch import device, tensor, save as torch_save, Tensor, where
from torch.cuda import empty_cache, is_available as cuda_is_available
from torch.nn import CosineEmbeddingLoss
from torch.optim import Adam, SGD
from torch_geometric.loader import DataLoader
from torch_geometric.utils import coalesce, remove_isolated_nodes

def parse_arguments():
    parser = ArgumentParser(description='Train a graph classifier')

    # Add the required arguments
    parser.add_argument('--train_file_path', type=str, required=True, help='Path to the training file')
    parser.add_argument('--test_file_path', type=str, required=True, help='Path to the test file')
    parser.add_argument('--model_output_dir', type=str, required=True, help='Directory to save the trained model and the edge threshold')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs to train')

    # Add the optional arguments
    parser.add_argument('--hidden_channel_dimensions', type=str, default="[128, 128]", help='List of integers for hidden channel dimensions as a literal string such as "[int_0, int_1]]" (default: [128, 128])')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to use during training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate to use during training (default: 5e-5)')
    parser.add_argument('--edge_weight_percentile', type=float, default=80, help='Percentage or sequence of percentages for the percentiles to compute. Values must be between 0 and 100 inclusive (default=0)')
    parser.add_argument('--max_pairs_per_class', type=int, default=-1, help='Maximum number of pairs to generate per class where any two elements of the same class are considered positive and any element of a different class is considered negative. (default: -1)')

    return parser.parse_args()


def load_pytorch_geometric_data(file_path:str) -> list:
    data_format_string = "The data is in an unexpected format. The function assumes the file path is a pickled list of torch_geometric.data.Data objects"
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    # assert isinstance(dataset, list), data_format_string
    # for data in dataset:
    #     assert isinstance(data, Data), data_format_string
    dataset = [data for data in dataset if data]
    return dataset         


def construct_pairs(dataset, max_pairs_per_class:int=-1):
    # create positive and negative pairs
    positive_pairs = []
    negative_pairs = []
    for i, graph in enumerate(dataset):
        for j in range(i+1, len(dataset)):
            if graph.y == dataset[j].y:
                positive_pairs.append((i,j,1))
            else:
                negative_pairs.append((i,j,-1))
    
    if max_pairs_per_class > 0:
        negative_pairs = random.sample(negative_pairs, min(len(positive_pairs)*3, len(negative_pairs)))
        
    return positive_pairs + negative_pairs

def construct_pairs(dataset, max_pairs_per_class:int=100000):
    labels = tensor([data.y for data in dataset])
    pairs = []
    for class_id in list(set(labels.tolist())):
        # create positive and negative pairs
        _positive_indices = where(labels==tensor([class_id]))[0].tolist()
        _negative_indices = where(labels!=tensor([class_id]))[0].tolist()
        _positive_pairs = []
        _negative_pairs = []
        for i, data_idx in enumerate(_positive_indices):
            for positive_idx in _positive_indices[i+1:]: # iterate over distinct elements of same class
                _positive_pairs.append((data_idx,positive_idx,1))
            for negative_idx in _negative_indices:
                _negative_pairs.append((data_idx,negative_idx,-1))
                
        if max_pairs_per_class > 0:
            class_pairs = list( # round robin select from list where positive pairs are listed first
                islice(
                    cycle(_positive_pairs + _negative_pairs),
                    max_pairs_per_class
                )
            )
        else:
            class_pairs=_positive_pairs + _negative_pairs
        # add class pairs to overall set of pairs
        pairs += class_pairs

    return pairs


def deduplicate_edges(pyg_dataset):
    for data in pyg_dataset:
        data.edge_index, data.edge_attr = coalesce(data.edge_index, data.edge_attr, reduce="mean")

def edge_filter(pyg_dataset, edge_weight_percentile:float, threshold:Optional[float]=None) -> float:
    if not threshold:
        edge_weights = []
        for data in pyg_dataset:
            edge_weights += data.edge_attr.flatten().tolist()
        threshold = percentile(edge_weights, edge_weight_percentile)
    for data in pyg_dataset:
        edge_attr = tensor(data.edge_attr)
        valid_edges = where(edge_attr > tensor(threshold))[0]
        data.edge_index = data.edge_index[:,valid_edges]
        # data.edge_index = data.edge_index[:,valid_edges][[1,0],:]
        data.edge_attr = edge_attr[valid_edges]
        
        # remove isolated nodes
        data.edge_index, data.edge_attr, mask = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes=data.x.shape[0])
        data.x = data.x[mask,:]
        data.node_types = data.node_types[mask]
        
    return threshold

def train_graph_classifier(
    train_file_path:str,
    test_file_path:str,
    model_output_dir:str,
    num_epochs:int,
    hidden_channel_dimensions:list=[128,128], # note GNNs can incur over-smoothing issues when the number of layers is too high. Consider the diameter of your graph when picking these values.
    batch_size:int=4,
    learning_rate:float=5e-5,
    edge_weight_percentile:int=0,
    max_pairs_per_class:int=-1
):
    # load_data assuming data is a list of pytorch geometric DATA
    train_dataset = load_pytorch_geometric_data(train_file_path)
    test_dataset = load_pytorch_geometric_data(test_file_path)
    
    # fix dataset duplicates
    deduplicate_edges(train_dataset)
    deduplicate_edges(test_dataset)
    
    # generate positive and negative pairs
    training_pairs = construct_pairs(train_dataset, max_pairs_per_class)
    test_pairs = construct_pairs(test_dataset, max_pairs_per_class)
        
    # filter edges beneath edge_weight_percentile and remove isolated nodes
    for graph in train_dataset:
        assert(graph.edge_index.shape[1]==graph.edge_attr.shape[0])
    for graph in test_dataset:
        assert(graph.edge_index.shape[1]==graph.edge_attr.shape[0])        
    if "edge_attr" in train_dataset[0] and edge_weight_percentile != 0:
        edge_weight_threshold = edge_filter(train_dataset, edge_weight_percentile, threshold=None)
        edge_weight_threshold = edge_filter(test_dataset, edge_weight_percentile, threshold=edge_weight_threshold)
    else:
        edge_weight_threshold = 0
    
    # create dataloaders

    train_loader = DataLoader(training_pairs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_pairs, batch_size=batch_size, shuffle=True)
    
    # determine whether a gpu is available
    torch_device = device("cuda") if cuda_is_available() else device("cpu")
    
    # instantiate model, evaluation loss, and optimizer
    model_metadata = {
        "hidden_channel_dimensions": [train_dataset[0].x.shape[1]] + hidden_channel_dimensions, # dynamically creates gatconv layers and pulls dataset feature size
        "edge_dim": train_dataset[0].edge_attr.shape[1]
        # "num_classes": len(set([data.y for data in train_dataset])) # find distinct labels in training dataset
    }
    model = GraphEncoder(**model_metadata).to(torch_device)
    criterion = CosineEmbeddingLoss(margin=0.5).to(torch_device)
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer = SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9)
    
    NODE_TYPE = max(train_dataset[0].node_types)
    
    def train():
        model.train()
        losses = []
        for pair_data in train_loader:  # Iterate in batches over the training dataset.
            source_indices, target_indices, labels = pair_data[0].tolist(), pair_data[1].tolist(), pair_data[2]
            unique_indices = sorted(set(source_indices).union(set(target_indices)))
            
            # update the index mapping to generate embeddings once
            subset_index_map = {data_idx: new_idx for new_idx, data_idx in enumerate(unique_indices)}
            
            # generate graph embeddings
            for data in DataLoader([train_dataset[idx] for idx in unique_indices], batch_size=len(unique_indices), shuffle=False):
                data = data.to(torch_device)
                embeddings = model(
                    node_features=data.x.to(torch.float32),
                    edge_index=data.edge_index,
                    batch=data.batch,
                    edge_attr=data.edge_attr.to(torch.float32),
                    # node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
                ) # Perform a single forward pass.
            
            # create anchors, positives, negatives from implicit indexing map
            sources, targets = embeddings[[subset_index_map[idx] for idx in source_indices]], embeddings[[subset_index_map[idx] for idx in target_indices]]
            if sources.shape==targets.shape:
                loss = criterion(sources, targets, labels.to(torch_device))  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                losses.append(loss.detach().item())
        empty_cache()
        return tensor(losses).mean()

    def test(pair_loader, dataset):
        model.eval()
        losses = []
        for pair_data in pair_loader:  # Iterate in batches over the training dataset.
            source_indices, target_indices, labels = pair_data[0].tolist(), pair_data[1].tolist(), pair_data[2]
            unique_indices = sorted(set(source_indices).union(set(target_indices)))
            
            # update the index mapping to generate embeddings once
            subset_index_map = {data_idx: new_idx for new_idx, data_idx in enumerate(unique_indices)}
            
            # generate graph embeddings
            for data in DataLoader([dataset[idx] for idx in unique_indices], batch_size=len(unique_indices), shuffle=False):
                data = data.to(torch_device)
                embeddings = model(
                    node_features=data.x.to(torch.float32),
                    edge_index=data.edge_index,
                    batch=data.batch,
                    edge_attr=data.edge_attr.to(torch.float32),
                    # node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
                ) # Perform a single forward pass.
            
            # create anchors, positives, negatives from implicit indexing map
            sources, targets = embeddings[[subset_index_map[idx] for idx in source_indices]], embeddings[[subset_index_map[idx] for idx in target_indices]]
            if sources.shape==targets.shape:
                loss = criterion(sources, targets, labels.to(torch_device))  # Compute the loss.
                losses.append(loss.detach().item())
        empty_cache()
        return tensor(losses).mean()
                        
    for epoch in tqdm(range(0, num_epochs)):
        train_loss = train()
        if epoch % 5 == 0:
            test_loss = test(test_loader, test_dataset)
            print(f'Epoch: {epoch:03d}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        else:
            print(f'Epoch: {epoch:03d}, Training Loss: {train_loss:.4f}')
        
    # run on final training epoch
    train_acc = test(train_loader, train_dataset)
    test_acc = test(test_loader, test_dataset)
    print(f'Epoch: {epoch:03d}, Training Loss: {train_acc:.4f}, Test Loss: {test_acc:.4f}')
    
    # use a scikit learn classifier to predict features from GNN
    from sklearn.svm import SVC
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import classification_report
    clf = SVC(kernel=cosine_similarity)
    train_embeddings = []
    train_labels = []
    for graphs in DataLoader(train_dataset, shuffle=False):
        graphs = graphs.to(torch_device) # Clone data and delete to avoid GPU memory issues
        graphs.x = graphs.x.to(torch.float32)
        graphs.edge_attr = graphs.edge_attr.to(torch.float32)
        
        embeddings = model(
            node_features=graphs.x,
            edge_index=graphs.edge_index,
            batch=graphs.batch,
            edge_attr=graphs.edge_attr,
            # node_subset_indices=where(graphs.node_types==tensor([NODE_TYPE]).to(torch_device))[0]
        ).detach().tolist() # Perform a single forward pass.
        if len(embeddings) == len(graphs.y):
            train_embeddings += embeddings
            train_labels += graphs.y.tolist()
        empty_cache()
    
    clf.fit(train_embeddings, train_labels)
    train_predictions = clf.predict(train_embeddings)
    print(classification_report(y_true=train_labels, y_pred=train_predictions))
    
    test_embeddings = []
    test_labels = []
    for graphs in DataLoader(test_dataset, shuffle=False):
        graphs = graphs.to(torch_device) # Clone data and delete to avoid GPU memory issues
        graphs.x = graphs.x.to(torch.float32)
        graphs.edge_attr = graphs.edge_attr.to(torch.float32)

        embeddings = model(
            node_features=graphs.x,
            edge_index=graphs.edge_index,
            batch=graphs.batch,
            edge_attr=graphs.edge_attr,
            node_subset_indices=where(graphs.node_types==tensor([NODE_TYPE]).to(torch_device))[0]
        ).detach().tolist() # Perform a single forward pass.
        if len(embeddings) == len(graphs.y):
            test_embeddings += embeddings
            test_labels += graphs.y.tolist()
        empty_cache()
        
    test_predictions = clf.predict(test_embeddings)
    print(classification_report(y_true=test_labels, y_pred=test_predictions))

    # save model
    model.to(device("cpu")) # move model back to cpu prior to saving
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    torch_save(model.state_dict(), Path(model_output_dir, "model.pt"))  
    model_metadata['edge_weight_threshold'] = edge_weight_threshold
    with open(Path(model_output_dir, "model_metadata.json"), 'w') as f:
        json.dump(model_metadata, f)

def main():
    args = parse_arguments()
    train_graph_classifier(
        train_file_path=args.train_file_path,
        test_file_path=args.test_file_path,
        model_output_dir=args.model_output_dir,
        num_epochs=args.num_epochs,
        hidden_channel_dimensions=literal_eval(args.hidden_channel_dimensions),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        edge_weight_percentile=args.edge_weight_percentile,
        max_pairs_per_class=args.max_pairs_per_class
    )

if __name__ == "__main__":
    main()

# example usage
# python -m model_training.graph_encoding_contrastive --train_file_path="" --test_file_path="" --model_output_dir="" --num_epochs=500 --hidden_channel_dimensions=[1024] --batch_size=256 --edge_weight_percentile=90 --learning_rate=5e-4

# python3 -m model_training.graph_encoding_contrastive --train_file_path=grading/data/watt/train_data.pkl --test_file_path=grading/data/watt/test_data.pkl --model_output_dir=grading/data/models --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=32 --edge_weight_percentile=0 --learning_rate=5e-4
