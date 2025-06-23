from sklearn.metrics import classification_report
from argparse import ArgumentParser
import pickle
import json
from torch import device, tensor, where
from torch_geometric.loader import DataLoader
from pathlib import Path
from torch import load as torch_load, float32 as torch_float32
from torch.cuda import is_available as cuda_is_available
from torch.cuda import empty_cache
from models.gnn.graph_classifier import GraphClassifier
from torch_geometric.utils import coalesce

def parse_arguments():
    parser = ArgumentParser(description='Test the graph classifier')
    parser.add_argument('--model_path', type=str, required=True, help='The path for graph model')
    parser.add_argument('--test_data', type=str, required=True, help='The path for test data')
    parser.add_argument('--label_mapping_path', type=str, required=True, help='The path for label mapping')
    return parser.parse_args()

def deduplicate_edges(pyg_dataset):
    for data in pyg_dataset:
        data.edge_index, data.edge_attr = coalesce(data.edge_index, data.edge_attr, reduce="mean")

def main():
    args = parse_arguments()    
    with open(Path(args.model_path, "model_metadata.json"), 'r') as file:
        config = json.load(file)
    torch_device = device("cuda") if cuda_is_available() else device("cpu")
    model_metadata = {
        "hidden_channel_dimensions": config["hidden_channel_dimensions"], # dynamically creates gatconv layers and pulls dataset feature size
        "edge_dim": config['edge_dim'],
        "num_classes": config['num_classes'] # find distinct labels in training dataset
        }
    model = GraphClassifier(**model_metadata).to(torch_device)
    model.load_state_dict(torch_load(Path(args.model_path, "model.pt")), strict=True)
    model.eval()
    model.to(torch_device)
    
    with open(args.test_data, 'rb') as f:
        test_data = pickle.load(f)
    # fix dataset duplicates
    deduplicate_edges(test_data)

    with open(args.label_mapping_path, "r") as f:
        skill_mapping = json.load(f)
    reverse_skill_mapping = {v:k for k,v in skill_mapping.items()}

    model.eval()
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # read off user question node type
    NODE_TYPE = max(test_data[0].node_types)
    
    correct = 0
    y_pred, y_true = [], []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
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
    print(classification_report(y_true, y_pred))
    empty_cache()
    print("ACC:", correct / len(test_loader.dataset))  # Derive ratio of correct predictions.

if __name__ == "__main__":
    main()
    
# usage: python -m model_training.graph_skill_inference --model_path="/home/azureuser/a100-fs-share1/shufan/saia-finetuning-post/tooling/model_output_middle/64_Acc_0.9627329192546584" --test_data="/home/azureuser/a100-fs-share1/shufan/saia-finetuning-post/tooling/model_output_middle/test_attention_graphs.pkl" --label_mapping_path="/home/azureuser/a100-fs-share1/shufan/saia-finetuning-post/tooling/model_output_middle/label_mapping.json"