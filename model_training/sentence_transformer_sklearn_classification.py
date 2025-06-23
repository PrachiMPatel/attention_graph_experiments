from ast import literal_eval
from argparse import ArgumentParser
from collections import Counter
import itertools
import json
from tqdm import tqdm
from pandas import DataFrame, read_csv
from pathlib import Path
import pickle as pkl
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from typing import Any, List

from models.nn.mlp_classifier import MLPClassifier
from torch import device, tensor
from torch.cuda import is_available as cuda_is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from sentence_transformers import SentenceTransformer
from torch_geometric.loader import DataLoader


def parse_arguments():
    parser = ArgumentParser(description='Train a graph classifier')

    # Add the required arguments
    parser.add_argument('--train_file_path', type=str, required=True, help='Path to the training .csv file containing a column names "prompt", "label" or specified as the "--prompt_key", "---label_key" parameters')
    parser.add_argument('--test_file_path', type=str, required=True, help='Path to the test .csv file containing a column names "prompt", "label" or specified as the "--prompt_key", "---label_key" parameters')
    parser.add_argument('--model_output_directory', type=str, required=True, help='Directoy to save the trained model and the label mapping')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs to train for (default: 1000)')

    # Add the optional arguments
    parser.add_argument('--prompt_key', type=str, default="prompt", help='column header for both the train and test files specifying the text to encode')
    parser.add_argument('--label_key', type=str, default="label", help='column header for both the train and test files specifying the label')

    return parser.parse_args()


def sentence_transformer_encode(prompts:List[str], encoder_name:str="WhereIsAI/UAE-Large-V1"):
    model = SentenceTransformer(encoder_name)
    if cuda_is_available:
        model = model.cuda()
    features = model.encode(prompts)
    del model
    return features


def encode_labels(labels:list, label_map:dict[Any,int]={}) -> tuple:
    if not label_map: # convert labels to integers
        label_map = {label: i for i, label in enumerate(list(set(labels)))}
    labels = tensor([label_map[label] for label in labels])    
    return labels, label_map


def load_csv_data(file_path:str, prompt_key:str="prompt", label_key:str="label") -> tuple:
    data_format_string = "The data is in an unexpected format. The function assumes the file path is a .csv file containing the {prompt_key} and {class_key} columns"
    with open(file_path, 'rb') as f:
        df = read_csv(f)
    assert isinstance(df, DataFrame), data_format_string
    assert prompt_key in df.columns, data_format_string
    assert label_key in df.columns, data_format_string

    return df[prompt_key], df[label_key]
    
def train(classifier, features, labels):
    classifier.fit(X=features, y=labels)

def test(classifier, features, labels):
    pred = classifier.predict(features)
    return accuracy_score(y_true=labels, y_pred=pred)  # Derive ratio of correct predictions.

def train_classifier(
    train_file_path:str,
    test_file_path:str,
    model_output_directory:str,
    num_epochs:int,
    prompt_key:str="prompt",
    label_key:str="label",
):
    # load_data assuming data is a csv
    train_prompts, train_labels = load_csv_data(train_file_path, prompt_key, label_key)
    test_prompts, test_labels = load_csv_data(test_file_path, prompt_key, label_key)
    
    # encode labels
    train_labels, label_map = encode_labels(train_labels, label_map={})
    test_labels, label_map = encode_labels(test_labels, label_map=label_map)
    
    # experiment_args = itertools.product(ENCODERS, CLASSIFIERS)
    ENCODERS = [
        "WhereIsAI/UAE-Large-V1",
        "sentence-transformers/paraphrase-mpnet-base-v2"
    ]
    for encoder in ENCODERS:
        train_features = tensor(sentence_transformer_encode(train_prompts, encoder_name=encoder))
        test_features = tensor(sentence_transformer_encode(test_prompts, encoder_name=encoder))
        train_features.to("cpu")
        test_features.to("cpu")
            
        CLASSIFIERS = [
            LogisticRegression(max_iter=num_epochs),
            KNeighborsClassifier(n_neighbors=min(Counter(test_labels.numpy()).values())),
            SVC(max_iter=num_epochs)
        ]
        for classifier in CLASSIFIERS:
            train(classifier, train_features, train_labels)
            train_acc = test(classifier, train_features, train_labels)
            test_acc = test(classifier, test_features, test_labels)
            print(f'(Encoder {encoder}, Classifier {classifier}): Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # # save model
    # model.to(device("cpu")) # move model back to cpu prior to saving
    Path(model_output_directory).mkdir(parents=True, exist_ok=True)
    # with open(Path(model_output_directory, "model.pkl"), 'wb') as f:
    #     pkl.dump(model, f)
    with open(Path(model_output_directory, "label_map.json"), 'w') as f:
        json.dump(label_map, f)


def main():
    args = parse_arguments()
    train_classifier(
        train_file_path=args.train_file_path,
        test_file_path=args.test_file_path,
        model_output_directory=args.model_output_directory,
        num_epochs=args.num_epochs,
        prompt_key=args.prompt_key,
        label_key=args.label_key,
    )

if __name__ == "__main__":
    main()

# example usage
# python -m model_training.sentence_transformer_classification_sklearn --train_file_path=train_data.csv --test_file_path=test_data.csv --prompt_key=input --label_key=reference_skill --model_output_directory=test_model --num_epochs=50 --hidden_channel_dimensions=[1024] --batch_size=8