from ast import literal_eval
from argparse import ArgumentParser
import json
from tqdm import tqdm
from pandas import DataFrame, read_csv
from pathlib import Path
import pickle as pkl

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
    parser.add_argument('--encoder', type=str, default="WhereIsAI/UAE-Large-V1", help='Sentence transformer model alias.')
    parser.add_argument('--prompt_key', type=str, default="prompt", help='column header for both the train and test files specifying the text to encode')
    parser.add_argument('--label_key', type=str, default="label", help='column header for both the train and test files specifying the label')
    parser.add_argument('--hidden_channel_dimensions', type=str, default="[128, 128]", help='List of integers for hidden channel dimensions as a literal string such as "[int_0, int_1]]" (default: [128, 128])')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to use during training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate to use during training (default: 5e-5)')

    return parser.parse_args()


def sentence_transformer_encode(encoder:str, prompts:list):
    model = SentenceTransformer(encoder)
    if cuda_is_available:
        model = model.cuda()
    return model.encode(prompts)


def construct_dataset(prompts:list, labels:list, encoder:str) -> tuple:
    # convert labels 
    label_map = {label: i for i, label in enumerate(list(set(labels)))}
    labels = tensor([label_map[label] for label in labels])
    features = tensor(sentence_transformer_encode(encoder, prompts))
    
    return features, labels, label_map


def load_csv_data(file_path:str, prompt_key:str="prompt", label_key:str="label") -> tuple:
    data_format_string = "The data is in an unexpected format. The function assumes the file path is a .csv file containing the {prompt_key} and {class_key} columns"
    with open(file_path, 'rb') as f:
        df = read_csv(f)
    assert isinstance(df, DataFrame), data_format_string
    assert prompt_key in df.columns, data_format_string
    assert label_key in df.columns, data_format_string

    return df[prompt_key], df[label_key]
    
    
def train_mlp_classifier(
    train_file_path:str,
    test_file_path:str,
    model_output_directory:str,
    num_epochs:int,
    encoder:str="WhereIsAI/UAE-Large-V1",
    prompt_key:str="prompt",
    label_key:str="label",
    hidden_channel_dimensions:list=[128,128],
    batch_size:int=4,
    learning_rate:float=5e-5,
):
    # load_data assuming data is a csv
    train_prompts, train_labels = load_csv_data(train_file_path, prompt_key, label_key)
    test_prompts, test_labels = load_csv_data(test_file_path, prompt_key, label_key)
        
    # create dataloaders
    train_features, train_labels, label_map = construct_dataset(train_prompts, train_labels, encoder)
    test_features, test_labels, _ = construct_dataset(test_prompts, test_labels, encoder)
    
    # set up DataLoader for training set
    train_loader = DataLoader(list(zip(train_features, train_labels)), shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(list(zip(test_features, test_labels)), shuffle=True, batch_size=batch_size)
    
    # determine whether a gpu is available
    torch_device = device("cuda") if cuda_is_available() else device("cpu")
    
    # instantiate model, evaluation loss, and optimizer
    model = MLPClassifier(
        hidden_channel_dimensions = [train_features.shape[1]] + hidden_channel_dimensions, # dynamically creates Linear layers and pulls dataset feature size
        num_classes = len(label_map), # find distinct labels in training dataset
    ).to(torch_device)
    criterion = CrossEntropyLoss().to(torch_device)
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    def train(loader):
        model.train()
        
        for X_batch, y_batch in loader:  # Iterate in batches over the training dataset.
            out = model(X_batch.to(torch_device))  # Perform a single forward pass.
            loss = criterion(out, y_batch.to(torch_device))  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            # data.to(torch.device("cpu"))

    def test(loader):
        model.eval()

        correct = 0
        for X_batch, y_batch in loader:  # Iterate in batches over the training/test dataset.
            out = model(X_batch.to(torch_device))
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == y_batch.to(torch_device)).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    for epoch in tqdm(range(1, num_epochs+1)):
        train(train_loader)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # save model
    model.to(device("cpu")) # move model back to cpu prior to saving
    Path(model_output_directory).mkdir(parents=True, exist_ok=True)
    with open(Path(model_output_directory, "model.pkl"), 'wb') as f:
        pkl.dump(model, f)
    with open(Path(model_output_directory, "label_map.json"), 'w') as f:
        json.dump(label_map, f)
    with open(Path(model_output_directory, "metadata.json"), 'w') as f:
        json.dump({"encoder": encoder}, f)


def main():
    args = parse_arguments()
    train_mlp_classifier(
        train_file_path=args.train_file_path,
        test_file_path=args.test_file_path,
        model_output_directory=args.model_output_directory,
        num_epochs=args.num_epochs,
        encoder=args.encoder,
        prompt_key=args.prompt_key,
        label_key=args.label_key,
        hidden_channel_dimensions=literal_eval(args.hidden_channel_dimensions),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate 
    )

if __name__ == "__main__":
    main()

# example usage
# python -m model_training.sentence_transformer_classification_sklearn --train_file_path=train_data.csv --test_file_path=test_data.csv --prompt_key=input --label_key=reference_skill --model_output_directory=test_model --num_epochs=50 --hidden_channel_dimensions=[1024] --batch_size=8