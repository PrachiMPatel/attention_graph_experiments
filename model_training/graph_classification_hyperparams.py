from ast import literal_eval
from argparse import ArgumentParser
import copy
import json
from tqdm import tqdm
from pathlib import Path
import pickle as pkl
from numpy import percentile
from sklearn.metrics import classification_report
from models.gnn.graph_classifier import GraphClassifier
from typing import Optional
from torch import device, tensor, save as torch_save, Tensor, where, float32 as torch_float32
from torch.cuda import is_available as cuda_is_available
from torch.cuda import empty_cache
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_isolated_nodes
from sklearn.model_selection import ParameterGrid
import random
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset
import numpy as np
from collections import Counter
from torch_geometric.utils import coalesce, remove_isolated_nodes
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
    parser.add_argument('--dropout', type=float, default=0.75, help='Dropout default 0.75')
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
    dataset = [data for data in dataset if data]
    return dataset         

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
    dropout:float=0.75,
    optimizer_type:str='SGD',
    early_stopping_patience:int=3,
    n_fold:int=5,
    run_ind:int=0
):
    with open(Path(model_output_dir, "label_mapping.json"), "r") as f:
        skill_mapping = json.load(f)
    reverse_skill_mapping = {v:k for k,v in skill_mapping.items()}

    def train(loader):
        model.train()
        for data in loader:  # Iterate in batches over the training dataset.
            data = data.to(torch_device)
            out = model(
                node_features=data.x.to(torch_float32),
                edge_index=data.edge_index,
                batch=data.batch,
                edge_attr=data.edge_attr.to(torch_float32),
                node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
            ) # Perform a single forward pass.

            if len(out) == len(data.y):
                loss = criterion(out, data.y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
        empty_cache()

    def test(loader):
        model.eval()
        correct = 0
        y_pred, y_true = [], []
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(torch_device)
            out = model(
                node_features=data.x.to(torch_float32),
                edge_index=data.edge_index,
                batch=data.batch,
                edge_attr=data.edge_attr.to(torch_float32),
                node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
            ) # Perform a single forward pass.
            
            if len(out) == len(data.y):
                preds = out.argmax(dim=1)  # Use the class with highest probability.
                y_pred.extend([reverse_skill_mapping[pred.item()] for pred in preds])
                y_true.extend([reverse_skill_mapping[each.item()] for each in data.y])
                correct += int((preds == data.y).sum())  # Check against ground-truth labels.
        empty_cache()
        with open(Path(model_output_dir, "output2.txt"), "a") as f:
            f.write(classification_report(y_true, y_pred))
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    def validation(loader):
        model.eval()
        correct = 0
        y_pred, y_true = [], []
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(torch_device)
            out = model(
                node_features=data.x.to(torch_float32),
                edge_index=data.edge_index,
                batch=data.batch,
                edge_attr=data.edge_attr.to(torch_float32),
                node_subset_indices=where(data.node_types==tensor([NODE_TYPE]).to(torch_device))[0],
            ) # Perform a single forward pass.
            
            if len(out) == len(data.y):
                preds = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((preds == data.y).sum())  # Check against ground-truth labels.
        empty_cache()
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
    
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
    
    # create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # k fold validation
    #kf=KFold(n_splits=2, shuffle=True)
    #kf=KFold(n_splits=n_fold, shuffle=True)
    kf=StratifiedKFold(n_splits=5,shuffle=True)

    fold=0
    test_acc_all=[]

    # for train_ind,val_ind in kf.split(train_dataset):
    for train_ind,val_ind in kf.split(train_dataset,alllabs):
        fold+=1
        no_improvement_counter=0
        best_test_acc=0.0
        best_model_state=None
        # get class distribution
        tr_lab=[alllabs[i] for i in train_ind]
        print(Counter(tr_lab))
        
        train_subset=Subset(train_dataset,train_ind)
        val_subset=Subset(train_dataset,val_ind)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=True)

        # determine whether a gpu is available
        torch_device = device("cuda") if cuda_is_available() else device("cpu")
        
        # instantiate model, evaluation loss, and optimizer
        model_metadata = {
            "hidden_channel_dimensions": [train_dataset[0].x.shape[1]] + hidden_channel_dimensions, # dynamically creates gatconv layers and pulls dataset feature size
            "edge_dim": train_dataset[0].edge_attr.shape[1],
            "num_classes": len(set([data.y for data in train_dataset])) # find distinct labels in training dataset
        }
        model = GraphClassifier(**model_metadata).to(torch_device)
        criterion = CrossEntropyLoss().to(torch_device)
        if optimizer_type=='sgd':
            optimizer = SGD(model.parameters(), lr=learning_rate, nesterov=False, momentum=0.9)
        elif optimizer_type=='adam':
            optimizer = Adam(model.parameters(), lr=learning_rate)


        for epoch in tqdm(range(0, num_epochs)):
            train(train_loader)
            if epoch % 20 == 0:
                empty_cache()
                train_acc = validation(train_loader)
                empty_cache()
                test_acc = validation(val_loader)
                with open(Path(model_output_dir,"output2.txt"), "a") as f:
                    f.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {test_acc:.4f}. no_improvement_counter={no_improvement_counter}' + "\n")
                if test_acc>best_test_acc:
                    best_test_acc=test_acc
                    no_improvement_counter=0
                    best_model_state=copy.deepcopy(model.state_dict())
                else:
                    no_improvement_counter+=1
    
                if no_improvement_counter>=early_stopping_patience:
                    with open(Path(model_output_dir, "output2.txt"), "a") as f:
                        f.write(f"Early stopping. Best Val Accuracy:{best_test_acc}" + "\n")
                    test_acc_all.append(best_test_acc)
                    break
        

        # run on final testing epoch
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        with open(Path(model_output_dir, "output2.txt"), "a") as f:
            f.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}' + "\n")
    
        # save model; might consider change to a new directory each time and save the model metadata together
        Path(model_output_dir, "_".join([str(dim) for dim in hidden_channel_dimensions]) +"_Acc_"+str(test_acc)).mkdir(parents=True, exist_ok=True)
        # Save model and optimizer state

        # torch.save({
        #     'model_state_dict': best_model_state,
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }, Path(model_output_dir, "_".join([str(dim) for dim in hidden_channel_dimensions]) +"_Acc_"+str(best_test_acc), "model.pt"))

        torch_save(best_model_state, Path(model_output_dir, "_".join([str(dim) for dim in hidden_channel_dimensions]) +"_Acc_"+str(test_acc), "model.pt"))  
        model_metadata['edge_weight_threshold'] = edge_weight_threshold
        model_metadata['batch_size'] = batch_size
        model_metadata['lr'] =  learning_rate
        model_metadata['dropout'] = dropout
        model_metadata['optimizer'] = optimizer_type
        model_metadata['hidden_chn'] = hidden_channel_dimensions
        with open(Path(model_output_dir, "_".join([str(dim) for dim in hidden_channel_dimensions]) +"_Acc_"+str(test_acc), "model_metadata.json"), 'w') as f:
            json.dump(model_metadata, f)

    avg_test_acc=np.mean(np.array(test_acc_all))
    std_test_acc=np.std(np.array(test_acc_all))
    print(f"Avg test acc:{avg_test_acc} std:{std_test_acc}")

def main():
    param_grid={
    'learning_rate':[1e-4, 0.5e-4, 1e-5, 1e-3],
    'optimizer':['sgd','adam'],
    'batch_size':[16,8],
    'dropout':[0.75,0.9, 0.5],
    'edge_weight_percentile':[90,70, 50, 95],
    'hidden_channel_dimensions':[[64],[128,64],  [256,128]],
    'epochs':[1000],
    'early_stopping_patience':[10]
    }
    num_itr=20
    all_params=list(ParameterGrid(param_grid))
    random_params=random.sample(all_params,num_itr)
    args = parse_arguments()
    
    for ind,params in enumerate(random_params):
        empty_cache()
        print(f"Params:{params}")
        train_graph_classifier(
            train_file_path=args.train_file_path,
            test_file_path=args.test_file_path,
            model_output_dir=args.model_output_dir,   
            num_epochs=params['epochs'],
            hidden_channel_dimensions=params['hidden_channel_dimensions'], 
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            edge_weight_percentile=params['edge_weight_percentile'],
            dropout=params['dropout'],
            optimizer_type=params['optimizer'],
            early_stopping_patience=params['early_stopping_patience'],
            n_fold=5,
            run_ind=ind
            )
    # for single iteration
    # train_graph_classifier(
    #     train_file_path=args.train_file_path,
    #     test_file_path=args.test_file_path,
    #     model_output_dir=args.model_output_dir,
    #     num_epochs=args.num_epochs,
    #     hidden_channel_dimensions=literal_eval(args.hidden_channel_dimensions),
    #     batch_size=args.batch_size,
    #     learning_rate=args.learning_rate,
    #     edge_weight_percentile=args.edge_weight_percentile,
    #     dropout=args.dropout,
    #     optimizer_type=args.optimizer,
    #     early_stopping_patience=args.early_stopping_patience

    # )

if __name__ == "__main__":
    main()

# example usage
# python -m model_training.graph_classification --train_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/spl/v1/gnn_inputs/train_data.pkl --test_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/spl/v1/gnn_inputs/test_data.pkl --model_output_dir=/home/ec2-user/lbetthauser/models/orchestrator/spl/v1/test_model --num_epochs=100 --hidden_channel_dimensions=[512,32] --batch_size=16 --edge_weight_percentile=95
# python -m model_training.graph_classification --train_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/security/v0/gnn_inputs/train_data.pkl --test_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/security/v0/gnn_inputs/test_data.pkl --model_output_dir=/home/ec2-user/lbetthauser/models/orchestrator/spl/v0/test_model_2 --num_epochs=100 --hidden_channel_dimensions=[512,32] --batch_size=16 --edge_weight_percentile=95
# python -m model_training.graph_classification --train_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/spl/v2/gnn_inputs/train_data.pkl --test_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/spl/v2/gnn_inputs/test_data.pkl --model_output_dir=/home/ec2-user/lbetthauser/models/orchestrator/spl/v2/test_model_2 --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=16 --edge_weight_percentile=90 --learning_rate=1e-4



# python -m model_training.graph_classification --train_file_path=/home/ec2-user/prachi/saia-finetuning/grading/data/v1/train_data.pkl --test_file_path=/home/ec2-user/prachi/saia-finetuning/grading/data/v1/test_data.pkl --model_output_dir=/home/ec2-user/prachi/leo/saia-finetuning/models/trained_models/v1/ --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=16 --edge_weight_percentile=90 --learning_rate=1e-4


# python -m model_training.graph_classification --train_file_path=/home/ec2-user/prachi/saia-finetuning/grading/data/v2/train_data.pkl --test_file_path=/home/ec2-user/prachi/saia-finetuning/grading/data/v2/test_data.pkl --model_output_dir=/home/ec2-user/prachi/leo/saia-finetuning/models/trained_models/v2/ --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=16 --edge_weight_percentile=95 --learning_rate=1e-4