from argparse import ArgumentParser
import json
from pathlib import Path
from tooling.llm_engine import GenerativeModel
from tooling.llm_engine.gorilla2_function import GorillaFunctionCalling as llm
from tooling.huggingface_latent_representations.transformers.attention_featurization import extract_latent_feature_graph, extract_fine_grained_latent_feature_graph
import torch
from torch import device
from tqdm import tqdm
from tooling.agents.data import load_csv_file, load_json_file
from time import time
from pandas import DataFrame
import pickle
from torch import arange

def extract_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/home/azureuser/lbetthauser/data/hybrid_orchestration/security/saia_intent.csv")
    parser.add_argument("--output_directory", type=str, default="data/grading/orchestration/hybrid_orchestration/spl")
    parser.add_argument("--function_template_file_path", type=str, default="/home/ec2-user/shufan/saia-finetuning/tooling/agents/orchestrator/orchestrator_data/function_metadata.json")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--function_count", type=int, default=-1)
    return parser.parse_args()


def load_data(file_path):
    """
    Loads data from the specified file path.
    """
    return load_csv_file(file_path)



def extract_llm_graph(
    model,
    input_ids,
    outputs,
    prompt:str,
):
    # specify token ids for relevant subportion of the prompt
    template_start_ids = model.search_for_target_sequence("### Instruction: <<function>>", prompt, input_ids)
    template_end_ids = model.search_for_target_sequence("<<question>>", prompt, input_ids)
    input_end_ids = model.search_for_target_sequence("### Response: ", prompt, input_ids)
    return extract_latent_feature_graph(
        input_ids=input_ids,
        outputs=outputs,
        position_index_a_endpoints=(template_start_ids+1, template_end_ids-2),
        position_index_b_endpoints=(template_end_ids+2, input_end_ids-4)
    )


def get_function_token_offset(
    model,
    input_ids,
    outputs,
    prompt:str,
    num_function
):
    function_name_start_ids = model.find_keywords(keywords=' {"name": ', prompt=prompt, input_ids=input_ids)
    function_name_end_ids = model.find_keywords(keywords='"parameters": {"type": "object", ', prompt=prompt, input_ids=input_ids)
    function_argument_start_ids = model.find_keywords(keywords='"properties": {"', prompt=prompt, input_ids=input_ids)
    function_argument_end_ids = model.find_keywords(keywords=', "required": [', prompt=prompt, input_ids=input_ids)

    # print(len(function_name_start_ids), len(function_name_end_ids), len(function_argument_start_ids), len(function_argument_end_ids))
    assert len(function_name_start_ids) == len(function_name_end_ids) == len(function_argument_start_ids) == len(function_argument_end_ids) == num_function
    try:
        template_end_ids = model.search_for_target_sequence("<<question>>", prompt, input_ids)
    except:
        template_end_ids = model.search_for_target_sequence("<<question", prompt, input_ids) # match for 
    input_end_ids = model.search_for_target_sequence("### Response: ", prompt, input_ids)
    
    return function_name_start_ids, function_name_end_ids, function_argument_start_ids, function_argument_end_ids, template_end_ids, input_end_ids
    
def construct_graph(
    input_ids,
    outputs,
    function_name_start_ids, 
    function_name_end_ids, 
    function_argument_start_ids, 
    function_argument_end_ids, 
    template_end_ids, 
    input_end_ids,
    num_function
):
    #  get all the input ids for input template
    tokenids = []
    for i in range(num_function):
        each_function = (function_name_start_ids[i],function_name_end_ids[i]) # each pair of start and end positions
        each_argument = (function_argument_start_ids[i],function_argument_end_ids[i])

        tokenids.append(arange(each_function[0], each_function[1]))
        tokenids.append(arange(each_argument[0], each_argument[1]))
    
    tokenids.append(arange(template_end_ids+2, input_end_ids-3))

    assert len(tokenids) == num_function * 2 + 1

    # all_functions is a list of list for input ids
    return extract_fine_grained_latent_feature_graph(
        input_ids=input_ids,
        outputs=outputs,
        target_ids=tokenids
    )

    

def extract_llm_graphs(
    input_file:str,
    output_directory:str, 
    function_template_file_path:str,
    limit:int=3,
    num_function:int=-1,
    model = llm()
) -> None:
    if num_function == -1:
        assert "please choose a reasonable number of functions"
    data = load_data(input_file)
    labels = data['label']
    inputs = data['text']
    # responses = []
    graph_latency = []
    response_latency = []
    label_mapping = {value: idx for idx, value in enumerate(list(set(labels)))}
    
    attention_graphs = []
    function_templates = load_json_file(function_template_file_path)
    if limit == -1: # map default to full examples for each skill
        limit = len(data) # the length of all examples
    
    # refactor the code to have two columns, one for examples, one for reference skills
    for idx, (input, label) in tqdm(enumerate(zip(inputs, labels)), total=limit):
        torch.cuda.empty_cache()
        if idx < limit:
            start = time()
            prompt = model.format_conversations(
                messages=[{"role": "user", "content": input}],
                functions = function_templates
            )
            input_ids, response = model._extract_latent_features(prompt=prompt, max_tokens=1)
            # first get start and end ids and then construct graph
            function_name_start_ids, function_name_end_ids, function_argument_start_ids, function_argument_end_ids, template_end_ids, input_end_ids = get_function_token_offset(model, input_ids, response, prompt, num_function)
            attention_graph = construct_graph(input_ids, response, function_name_start_ids, function_name_end_ids, function_argument_start_ids, function_argument_end_ids, template_end_ids, input_end_ids, num_function)
            attention_graph.y = label_mapping[label]
            # attention_graph.to(device("cpu"))
            attention_graphs.append(attention_graph)
            
            torch.cuda.empty_cache()
            end = time()
            graph_latency.append(end-start)
            
            # start = time()
            # response = model._generate_response(
            #     prompt=prompt,
            #     max_tokens=128
            # )
            # responses.append(response)
            # end = time()
            
            # response_latency.append(end-start)
        # else:
            # attention_graphs.append([])
            # responses.append("")
            # graph_latency.append(-1)
            # response_latency.append(-1)
            
    # save data
    # DataFrame({
    #     "inputs": inputs,
    #     "label": labels,
    #     "responses": responses,
    #     "graph_latency": graph_latency,
    #     "response_latency": response_latency
    # }).to_csv(Path(output_directory, "responses.csv"), index=False, header=True)
    # with open(Path(output_directory, "attention_graphs.pkl"), "wb") as f:
    #     pickle.dump(attention_graphs, f)
    # with open(Path(output_directory, "label_mapping.json"), "w") as f:
    #     json.dump(label_mapping, f)

    return attention_graphs, label_mapping

if __name__ == "__main__":
    input_args = extract_arguments()
    attention_graphs, label_mapping = extract_llm_graphs(
        input_file=input_args.input_file, 
        output_directory=input_args.output_directory, 
        function_template_file_path=input_args.function_template_file_path,
        limit=input_args.limit,
        num_function=input_args.function_count,
        model=llm()
    )
    
    with open(Path(input_args.output_directory, "attention_graphs.pkl"), "wb") as f:
        pickle.dump(attention_graphs, f)
    with open(Path(input_args.output_directory, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)
# python -m experiments.orchestration.latent_graph_extraction --input_file="/home/azureuser/shufan/saia-finetuning-pre/data_processing/orchestrator/testdata_emma.csv" --output_dir="/home/azureuser/shufan/saia-finetuning/tooling/orchestrator_output" --model_name="gorilla" --train_data_percentage=0.8