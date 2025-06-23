from ast import literal_eval
from argparse import ArgumentParser
import json
from tqdm import tqdm
from pathlib import Path
import pickle as pkl
from numpy import percentile
import torch
from models.gnn.graph_classifier import GraphEncoder
from typing import Optional
from torch import device, tensor, save as torch_save, Tensor, where
from torch.cuda import empty_cache, is_available as cuda_is_available
from torch.nn import CosineSimilarity, PairwiseDistance, TripletMarginWithDistanceLoss
from torch.nn.functional import cosine_similarity
from torch.optim import Adam, SGD
from torch_geometric.data import Data
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
    parser.add_argument('--max_triplets_per_class', type=int, default=10000, help='Maximum number of triplets to generate per class where any two elements of the same class are considered positive and any element of a different class is considered negative.')

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


def construct_triplets(dataset, max_triplets_per_class:int=10000):
    labels = tensor([data.y for data in dataset])
    triplets = []
    for class_id in list(set(labels.tolist())):
        # create positive and negative pairs
        _positive_indices = where(labels==tensor([class_id]))[0]
        _negative_indices = where(labels!=tensor([class_id]))[0]
        
        # randomly sample positive, negative indices to generate (anchor, positive, negative) triplets
        sample_size = max_triplets_per_class
        anchor_indices = _positive_indices[torch.randint(0, len(_positive_indices), (sample_size,))].tolist()
        positive_indices = _positive_indices[torch.randint(0, len(_positive_indices), (sample_size,))].tolist()
        negative_indices = _negative_indices[torch.randint(0, len(_negative_indices), (sample_size,))].tolist()

        # add class triplets to overall set of triplets
        triplets += [
            (anchor, positive, negative) 
        for anchor, positive, negative in zip(anchor_indices, positive_indices, negative_indices) if anchor != positive]
        
    # deduplicate triplets
    triplets = torch.unique(tensor(triplets), dim=0).T
    return triplets

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
    max_triplets_per_class:int=10000
):
    # load_data assuming data is a list of pytorch geometric DATA
    train_dataset = load_pytorch_geometric_data(train_file_path)
    test_dataset = load_pytorch_geometric_data(test_file_path)
    
    # fix dataset duplicates
    deduplicate_edges(train_dataset)
    deduplicate_edges(test_dataset)
    
    # generate positive and negative pairs
    training_triplets = construct_triplets(train_dataset, max_triplets_per_class=max_triplets_per_class)
    test_triplets = construct_triplets(test_dataset, max_triplets_per_class=max_triplets_per_class)
        
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

    train_loader = DataLoader(training_triplets, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_triplets, batch_size=batch_size, shuffle=False)
    
    # determine whether a gpu is available
    torch_device = device("cuda") if cuda_is_available() else device("cpu")
    
    # instantiate model, evaluation loss, and optimizer
    model_metadata = {
        "hidden_channel_dimensions": [train_dataset[0].x.shape[1]] + hidden_channel_dimensions, # dynamically creates gatconv layers and pulls dataset feature size
        "edge_dim": train_dataset[0].edge_attr.shape[1]
    }
    model = GraphEncoder(**model_metadata).to(torch_device)
    # criterion = TripletMarginWithDistanceLoss(distance_function=PairwiseDistance())
    criterion = (TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - cosine_similarity(x, y), margin=1))
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer = SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.8)
    
    NODE_TYPE = max(train_dataset[0].node_types) # max value corresponds to node representing the user question
    
    def train():
        model.train()
        losses = []
        for pair_data in train_loader:  # Iterate in batches over the training dataset.
            anchor_indices, positive_indices, negative_indices = pair_data[0].tolist(), pair_data[1].tolist(), pair_data[2].tolist()
            unique_indices = sorted(set(anchor_indices).union(set(positive_indices)).union(negative_indices))
            
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
                    node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
                ) # Perform a single forward pass.
            
            # create anchors, positives, negatives from implicit indexing map
            anchors, positives, negatives = embeddings[[subset_index_map[idx] for idx in anchor_indices]], embeddings[[subset_index_map[idx] for idx in positive_indices]], embeddings[[subset_index_map[idx] for idx in negative_indices]]
            if anchors.shape==positives.shape and anchors.shape==negatives.shape:
                loss = criterion(anchors, positives, negatives)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                losses.append(loss.detach().item())
        empty_cache()
        return tensor(losses).mean()

    def test(triplet_loader, dataset):
        model.eval()
        losses = []
        for pair_data in triplet_loader:  # Iterate in batches over the training dataset.
            anchor_indices, positive_indices, negative_indices = pair_data[0].tolist(), pair_data[1].tolist(), pair_data[2].tolist()
            unique_indices = sorted(set(anchor_indices).union(set(positive_indices)).union(negative_indices))
            
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
                    node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
                ) # Perform a single forward pass.
            
            # create anchors, positives, negatives from implicit indexing map
            anchors, positives, negatives = embeddings[[subset_index_map[idx] for idx in anchor_indices]], embeddings[[subset_index_map[idx] for idx in positive_indices]], embeddings[[subset_index_map[idx] for idx in negative_indices]]
            if anchors.shape==positives.shape and anchors.shape==negatives.shape:
                loss = criterion(anchors, positives, negatives)  # Compute the loss.
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity, euclidean_distances
    from sklearn.metrics import classification_report
    # clf = LogisticRegression()
    clf = SVC(kernel=sk_cosine_similarity)
    # clf = SVC(kernel=euclidean_distances)
    train_embeddings = []
    train_labels = []
    for graphs in DataLoader(train_dataset, batch_size=1, shuffle=False):
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
            train_embeddings += embeddings
            train_labels += graphs.y.tolist()
        empty_cache()
    
    clf.fit(train_embeddings, train_labels)
    train_predictions = clf.predict(train_embeddings)
    print(classification_report(y_true=train_labels, y_pred=train_predictions))
    
    test_embeddings = []
    test_labels = []
    for graphs in DataLoader(test_dataset, batch_size=1, shuffle=False):
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
        max_triplets_per_class=args.max_triplets_per_class
    )

if __name__ == "__main__":
    main()

# example usage
# python -m model_training.graph_classification_triplet_temp --train_file_path="" --test_file_path="" --model_output_dir="" --num_epochs=500 --hidden_channel_dimensions=[256,32] --batch_size=256 --edge_weight_percentile=90 --learning_rate=5e-4 --max_triplets_per_class=10000