from argparse import ArgumentParser
from collections import Counter
import copy
import json
from itertools import cycle, islice, permutations
from tqdm import tqdm
import random
from pathlib import Path
import pickle as pkl
from numpy import percentile
import numpy as np
import random
from typing import Optional

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch import device, tensor, save as torch_save, Tensor, where
from torch.cuda import empty_cache, is_available as cuda_is_available
from torch.nn import CosineEmbeddingLoss
from torch.optim import Adam, SGD
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import coalesce, remove_isolated_nodes

from models.gnn.graph_classifier import GraphEncoder

def parse_arguments():
    parser = ArgumentParser(description='Train a graph classifier')

    # Add the required arguments
    parser.add_argument('--train_file_path', type=str, required=True, help='Path to the training file')
    parser.add_argument('--test_file_path', type=str, required=True, help='Path to the test file')
    parser.add_argument('--model_output_dir', type=str, required=True, help='Directory to save the trained model and the edge threshold')

    # Add the optional arguments
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--hidden_channel_dimensions', type=str, default="[128, 128]", help='List of integers for hidden channel dimensions as a literal string such as "[int_0, int_1]]" (default: [128, 128])')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to use during training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate to use during training (default: 5e-5)')
    parser.add_argument('--edge_weight_percentile', type=float, default=80, help='Percentage or sequence of percentages for the percentiles to compute. Values must be between 0 and 100 inclusive (default=0)')
    parser.add_argument('--max_samples', type=int, default=-1, help='Maximum number of pairs to generate per class where any two elements of the same class are considered positive and any element of a different class is considered negative. (default: -1)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer default SGD')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='early_stopping_patience default 5')
    
    return parser.parse_args()


def load_pytorch_geometric_data(file_path:str) -> list:
    data_format_string = "The data is in an unexpected format. The function assumes the file path is a pickled list of torch_geometric.data.Data objects"
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    # assert isinstance(dataset, list), data_format_string
    # for data in dataset:
    #     assert isinstance(data, Data), data_format_string
    # dataset = [data for data in dataset if data]
    dataset = [data for data in dataset if not isinstance(data, list)]
    return dataset         


def construct_pairs(dataset, max_samples:int=2**12, class_weights:list[float]=[]):
    labels = tensor([data.y for data in dataset])
    pairs = []
    unique_labels = set(labels.tolist())
    for key in unique_labels:
        assert isinstance(key, int), "labels are assumed to be consecutive integers [0,#num_classes)"
    if not class_weights:
        assert len(class_weights) <= len(unique_labels), "class_weights are assumed to correspond to each class by index."
        class_weights = [1.0/len(unique_labels)]
    else: # normalize class_weights
        class_weights = [weight/float(sum(class_weights)) for weight in class_weights]
        
    for class_id in list(unique_labels):
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

        if max_samples > 0:
            num_class_samples = int(max_samples * class_weights[class_id])
            approximate_half = int(num_class_samples / 2)
            positive_pairs = list( # round robin select from list where positive pairs are listed first
                islice(
                    cycle([_positive_pairs[i] for i in np.random.permutation(len(_positive_pairs))]),
                    approximate_half
                )
            ) 
            negative_pairs = list( # round robin select from list where positive pairs are listed first
                islice(
                    cycle([_negative_pairs[i] for i in np.random.permutation(len(_negative_pairs))]),
                    num_class_samples - approximate_half
                )
            )
            class_pairs = positive_pairs + negative_pairs
        else:
            class_pairs = _positive_pairs + _negative_pairs
            
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

def train_graph_encoder(
    train_file_path:str,
    test_file_path:str,
    model_output_dir:str,
    num_epochs:int,
    hidden_channel_dimensions:list=[128,128], # note GNNs can incur over-smoothing issues when the number of layers is too high. Consider the diameter of your graph when picking these values.
    batch_size:int=4,
    learning_rate:float=5e-5,
    edge_weight_percentile:int=0,
    max_samples:int=100000,
    optimizer_type:str='SGD',
    early_stopping_patience:int=3,
    n_fold:int=5,
    run_ind:int=0
):
    def train():
        model.train()
        losses = []
        _batch_losses = []
        _class_losses = [0.0 for _ in unique_labels]
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
                    node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
                ) # Perform a single forward pass.
            
            # create anchors, positives, negatives from implicit indexing map
            sources, targets = embeddings[[subset_index_map[idx] for idx in source_indices]], embeddings[[subset_index_map[idx] for idx in target_indices]]

            if sources.shape==targets.shape:
                loss = criterion(sources, targets, labels.to(torch_device))  # Compute the loss.
                loss.mean().backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                _batch_losses = loss.detach().tolist()
                losses.append(loss.mean().detach().item())

            for source_idx, target_idx, element_loss in zip(source_indices, target_indices, _batch_losses):
                _class_losses[idx_class_map[source_idx]] += element_loss
                _class_losses[idx_class_map[target_idx]] += element_loss
        empty_cache()
        return tensor(losses).mean(), _class_losses

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
                    node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
                ) # Perform a single forward pass.
            
            # create anchors, positives, negatives from implicit indexing map
            sources, targets = embeddings[[subset_index_map[idx] for idx in source_indices]], embeddings[[subset_index_map[idx] for idx in target_indices]]

            if sources.shape==targets.shape:
                loss = criterion(sources, targets, labels.to(torch_device))  # Compute the loss.
                loss.mean().backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                losses.append(loss.mean().detach().tolist())
        empty_cache()
        return tensor(losses).mean()

    
    # load_data assuming data is a list of pytorch geometric DATA
    train_dataset = load_pytorch_geometric_data(train_file_path)
    test_dataset = load_pytorch_geometric_data(test_file_path)
    
    # fix dataset duplicates
    deduplicate_edges(train_dataset)
    deduplicate_edges(test_dataset)
    
    # read off user question node type
    NODE_TYPE = max(train_dataset[0].node_types)

    alllabs=[data.y for data in train_dataset]

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
    
    # generate positive and negative pairs create dataloaders
    # test_pairs = construct_pairs(test_dataset, max_samples=-1)
    test_pairs = construct_pairs(test_dataset, max_samples=-1)
    test_loader = DataLoader(test_pairs, batch_size=batch_size, shuffle=False)
    
    # k fold validation
    #kf=KFold(n_splits=2, shuffle=True)
    #kf=KFold(n_splits=n_fold, shuffle=True)
    kf=StratifiedKFold(n_splits=5,shuffle=True)

    fold=0
    val_loss_all=[]

    # for train_ind,val_ind in kf.split(train_dataset):
    for train_ind,val_ind in kf.split(train_dataset,alllabs):
        fold+=1
        no_improvement_counter=0
        best_val_loss=2.0
        best_model_state=None
        # get class distribution
        tr_lab=[alllabs[i] for i in train_ind]
        print(Counter(tr_lab))
        
        train_subset=Subset(train_dataset,train_ind)
        val_subset=Subset(train_dataset,val_ind)
        
        # generate positive and negative pairs
        # training_pairs = construct_pairs(train_subset, max_samples=max_samples)
        val_pairs = construct_pairs(val_subset, max_samples=-1)
        # val_pairs = construct_pairs(val_subset, max_samples=-1)

        # train_loader = DataLoader(training_pairs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_pairs, batch_size=batch_size, shuffle=True)

        # keep track of classes
        idx_class_map = {i: data.y for i,data in enumerate(train_dataset)}
        unique_labels = set(alllabs)
        
        # determine whether a gpu is available
        torch_device = device("cuda") if cuda_is_available() else device("cpu")
        
        # instantiate model, evaluation loss, and optimizer
        model_metadata = {
            "hidden_channel_dimensions": [train_dataset[0].x.shape[1]] + hidden_channel_dimensions, # dynamically creates gatconv layers and pulls dataset feature size
            "edge_dim": train_dataset[0].edge_attr.shape[1]
        }
        model = GraphEncoder(**model_metadata).to(torch_device)
        criterion = CosineEmbeddingLoss(margin=0.5, reduction='none').to(torch_device)
        if optimizer_type=='sgd':
            optimizer = SGD(model.parameters(), lr=learning_rate, nesterov=False, momentum=0.9)
        elif optimizer_type=='adam':
            optimizer = Adam(model.parameters(), lr=learning_rate)

        class_losses = [1.0 for _ in range(len(set(alllabs)))]
        for epoch in tqdm(range(0, num_epochs)):
            training_pairs = construct_pairs( # sample pairs weighted by loss
                train_subset,
                max_samples,
                class_weights=[loss/sum(class_losses) for loss in class_losses]
            )
            train_loader = DataLoader(training_pairs, batch_size=batch_size, shuffle=True)
            # train()
            _, class_losses = train()
            if epoch % 20 == 0:
                train_acc = test(train_loader, train_subset)
                val_loss = test(val_loader, val_subset)
                print(f'Epoch: {epoch:03d}, Train Loss: {train_acc:.4f}, Val Loss: {val_loss:.4f}. no_improvement_counter={no_improvement_counter}')
                if val_loss<best_val_loss:
                    best_val_loss=val_loss.detach().item()
                    no_improvement_counter=0
                    best_model_state=copy.deepcopy(model.state_dict())
                else:
                    no_improvement_counter+=1
    
                if no_improvement_counter>=early_stopping_patience:
                    print(f"Early stopping. Best Val Loss:{best_val_loss}")
                    val_loss_all.append(best_val_loss)
                    break


        # run on final training epoch
        train_loss = test(train_loader, train_subset)
        test_loss = test(test_loader, test_dataset)
        val_loss = test(val_loader, val_subset)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} Test Loss: {test_loss:.4f}')
        
        # predict on test set    
        # use a scikit learn classifier to predict features from GNN
        clf = SVC(kernel=cosine_similarity, class_weight="balanced")
        train_embeddings = []
        train_labels = []
        for graphs in DataLoader(train_dataset, shuffle=False):
            graphs = graphs.to(torch_device) # Clone data and delete to avoid GPU memory issues 
            embeddings = model(
                node_features=graphs.x.to(torch.float32),
                edge_index=graphs.edge_index,
                batch=graphs.batch,
                edge_attr=graphs.edge_attr.to(torch.float32),
                node_subset_indices=where(graphs.node_types==tensor([NODE_TYPE]).to(torch_device))[0]
            ).detach().tolist() # Perform a single forward pass.

            if len(embeddings) == len(graphs.y):
                train_embeddings += embeddings
                train_labels += graphs.y.tolist()
            empty_cache()
        
        clf.fit(train_embeddings, train_labels)
        # train_predictions = clf.predict(train_embeddings)
        # print(classification_report(y_true=train_labels, y_pred=train_predictions))
        
        test_embeddings = []
        test_labels = []
        for graphs in DataLoader(test_dataset, shuffle=False):
            graphs = graphs.to(torch_device) # Clone data and delete to avoid GPU memory issues
            embeddings = model(
                node_features=graphs.x.to(torch.float32),
                edge_index=graphs.edge_index,
                batch=graphs.batch,
                edge_attr=graphs.edge_attr.to(torch.float32),
                node_subset_indices=where(graphs.node_types==tensor([NODE_TYPE]).to(torch_device))[0]
            ).detach().tolist() # Perform a single forward pass.
            if len(embeddings) == len(graphs.y):
                test_embeddings += embeddings
                test_labels += graphs.y.tolist()
            empty_cache()
            
        test_predictions = clf.predict(test_embeddings)
        print(classification_report(y_true=test_labels, y_pred=test_predictions))
    
        # save model
        #model.to(device("cpu")) # move model back to cpu prior to saving
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        torch_save(best_model_state, Path(model_output_dir, f"Fold_{fold}_loss_{best_val_loss:4f}_Runind_{run_ind}_model.pt"))  
        model_metadata['edge_weight_threshold'] = edge_weight_threshold
        model_metadata['batch_size'] = batch_size
        model_metadata['lr'] = learning_rate
        model_metadata['optimizer'] = optimizer_type
        model_metadata['hidden_chn'] = hidden_channel_dimensions
        with open(Path(model_output_dir, "model_metadata.json"), 'w') as f:
            json.dump(model_metadata, f)

    avg_val_loss=np.mean(np.array(val_loss_all))
    std_val_loss=np.std(np.array(val_loss_all))
    print(f"Avg test loss:{avg_val_loss} std:{std_val_loss}")


def main():
    param_grid={
        'learning_rate':[3e-4],
        'optimizer':['sgd'],
        'batch_size':[2**10],
        'edge_weight_percentile':[0,50],
        'hidden_channel_dimensions':[[512]],
        'epochs':[2000],
        'early_stopping_patience':[10],
        'max_samples':[2**10]
    }
    num_itr=20
    all_params=list(ParameterGrid(param_grid))
    random_params=random.sample(all_params,min(num_itr,len(all_params)))
    #print(random_params)
    args = parse_arguments()
    
    for ind,params in enumerate(random_params):
        torch.cuda.empty_cache()
        print(f"Params:{params}")
        train_graph_encoder(
            train_file_path=args.train_file_path,
            test_file_path=args.test_file_path,
            model_output_dir=args.model_output_dir,   
            num_epochs=params['epochs'],
            hidden_channel_dimensions=params['hidden_channel_dimensions'], 
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            edge_weight_percentile=params['edge_weight_percentile'],
            max_samples=params['max_samples'],
            optimizer_type=params['optimizer'],
            early_stopping_patience=params['early_stopping_patience'],
            n_fold=5,
            run_ind=ind
        )
        
if __name__ == "__main__":
    main()

# python -m model_training.graph_encoding_contrastive_dynamic --train_file_path="/home/azureuser/a100-fs-share1/shufan/saia-finetuning-post/tooling/model_output/train_attention_graphs.pkl" --test_file_path="/home/azureuser/a100-fs-share1/shufan/saia-finetuning-post/tooling/model_output/test_attention_graphs.pkl" --model_output_dir="/mount/splunka100groupstorage/a100-fs-share1/lbetthauser/models/orchestrator/ming_spl/shufan_llama3_1_grid_search_5"
