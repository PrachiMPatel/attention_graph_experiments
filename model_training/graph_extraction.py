
from tooling.llm_engine.gorilla2_function import GorillaFunctionCalling as llm
from experiments.orchestration.latent_graph_extraction_spl import extract_llm_graphs
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter, defaultdict
import pickle
import pdb
import random
import json
import torch


def parse_arguments():
    parser = ArgumentParser(description='Generate the graph features from decoder model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the test file (.csv)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the attention data for training GNN model')
    parser.add_argument('--model_name', type=str, required=True, help='the chosen decoder for getting attentions')
    parser.add_argument('--train_data_percentage', type=float, required=True, help='training data portion (float)')
    parser.add_argument("--function_count", type=int, default=-1)
    return parser.parse_args()

# construct training/test data for GNN training/test: this will return two pickle files for train and test seperately and their label mapping
def data_split(graph_data, TRAIN_PERCENTAGE):
    with open(graph_data, "rb") as f:
        graph_data = pickle.load(f)
    labels = Counter([graph.y for graph in graph_data])
    MIN_NUM_EXAMPLES = 2
    MAX_NUM_EXAMPLES = 20
    training_counts = {label: min(max(int(labels[label]*TRAIN_PERCENTAGE),MIN_NUM_EXAMPLES),MAX_NUM_EXAMPLES) for label in labels}

    train_dataset = []
    test_dataset = []

    example_count = defaultdict(list)
    for idx, graph in enumerate(graph_data):
        if graph:
            if len(example_count[graph.y]) < training_counts[graph.y]:
                train_dataset.append(graph)
                example_count[graph.y].append(idx)
            else:
                test_dataset.append(graph)

    return train_dataset, test_dataset

def main():
    args = parse_arguments()

    attention_graphs, label_mapping = extract_llm_graphs(input_file=args.input_file, output_directory=args.output_dir, function_template_file_path="/home/ec2-user/shufan/saia-finetuning/tooling/agents/orchestrator/orchestrator_data/spl_function_metadata.json", limit=-1, num_function=args.function_count, model=llm())
    
    with open(Path(args.output_dir, "attention_graphs.pkl"), "wb") as f:
        pickle.dump(attention_graphs, f)
    with open(Path(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)
    

if __name__ == "__main__":
    main()
    
# usage: python -m model_training.graph_extraction --input_file="/home/azureuser/shufan/saia-finetuning-pre/data_processing/orchestrator/testdata_emma.csv" --output_dir="/home/azureuser/shufan/saia-finetuning/tooling/orchestrator_output" --model_name="gorilla" --train_data_percentage=0.8