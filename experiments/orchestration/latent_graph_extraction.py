from argparse import ArgumentParser
import json
from pathlib import Path
from tooling.llm_engine import GenerativeModel
from tooling.llm_engine.llama3_1instruct import Llama3_1Instruct as llm
from tooling.huggingface_latent_representations.transformers.attention_featurization import extract_latent_feature_graph, extract_fine_grained_latent_feature_graph
 
from torch import arange, device
from torch.cuda import empty_cache
from tqdm import tqdm
from tooling.agents.data import load_csv_file, load_json_file
from time import time
from pandas import DataFrame
import pickle

def extract_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/home/azureuser/lbetthauser/data/hybrid_orchestration/security/saia_intent.csv")
    parser.add_argument("--output_directory", type=str, default="data/grading/orchestration/hybrid_orchestration/spl")
    parser.add_argument("--function_template_file_path", type=str, default="/data/orchestration/hybrid_orchestration/spl/v1/function_metadata.json")
    parser.add_argument("--label_map_file_path", type=str, default="")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--function_start_str", type=str, default="###<<functions>>")
    parser.add_argument("--user_question_start_str", type=str, default="question")
    parser.add_argument("--assistant_start_str", type=str, default="assistant")    
    
    return parser.parse_args()


def load_data(file_path):
    """
    Loads data from the specified file path.
    """
    return load_csv_file(file_path)


def load_function_templates(file_path):
    return {
        template["name"]: template
    for template in load_json_file(file_path)}


def find_keywords(tokenizer, keywords, prompt, input_ids):
    tokenized_text = tokenizer.tokenize(keywords)[1:-1] # for matching purpose excluding the start and end token of provided matching tokens
    all_tokens = tokenizer.tokenize(prompt)
    assert len(all_tokens) == input_ids.shape[1] -1
    indices = []
    i = 0
    while i <= len(all_tokens) - len(tokenized_text):
        if all_tokens[i:i+len(tokenized_text)] == tokenized_text:
            indices.append(i)
            i += len(tokenized_text)
        else:
            i += 1
    return indices


def search_for_target_sequence(tokenizer, target_text, prompt, input_ids):
    '''
    grab the template part from input
    '''
    tokenized_text = tokenizer.tokenize(target_text)
    reference_tokens = tokenizer.tokenize(prompt)
    assert len(reference_tokens) == input_ids.shape[1] -1 # this model will automatically add one token at the beginning
    assert set(tokenized_text).issubset(set(reference_tokens)), "target tokens are not subset of prompt"
    last_index = find_subarray(reference_tokens, tokenized_text)
    assert last_index != -1, "token not found"
    # last_index = reference_tokens.index(tokenized_text[-1]) # find the position of last token in target text
    return last_index + len(tokenized_text) - 1


def find_subarray(large_list, small_list):
    large_len = len(large_list)
    small_len = len(small_list)
    
    for i in range(large_len - small_len + 1):
        if large_list[i:i+small_len] == small_list:
            return i
    
    return -1  # Return -1 if no matching subarray is found


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
    
    if input_end_ids-2 == template_end_ids+2 + 1: # this is for the shortest user question, when there is only one token
        tokenids.append(arange(template_end_ids+2, input_end_ids-1))
    else:
        tokenids.append(arange(template_end_ids+2, input_end_ids-2))

    assert len(tokenids) == num_function * 2 + 1

    # all_functions is a list of list for input ids
    return extract_fine_grained_latent_feature_graph(
        input_ids=input_ids,
        outputs=outputs,
        target_ids=tokenids
    )
    
def extract_llm_graph(
    tokenizer,
    input_ids,
    outputs,
    prompt:str,
    function_start_str,
    user_question_start_str,
    assistant_start_str
):
    template_start_ids = search_for_target_sequence(tokenizer, function_start_str, prompt, input_ids)
    template_end_ids = search_for_target_sequence(tokenizer, user_question_start_str, prompt, input_ids)
    input_end_ids = search_for_target_sequence(tokenizer, assistant_start_str, prompt, input_ids)
    
    return extract_latent_feature_graph(
        input_ids=input_ids,
        outputs=outputs,
        position_index_a_endpoints=(template_start_ids+1, template_end_ids-2),
        position_index_b_endpoints=(template_end_ids+2, input_end_ids-4)
    )


def get_function_token_offset(
    tokenizer,
    input_ids,
    prompt:str,
    num_function:int,
    user_question_start_str:str="<<question>>",
    assistant_start_str:str="### Response: ",
):
    function_name_start_ids = find_keywords(tokenizer, keywords=' {"name": ', prompt=prompt, input_ids=input_ids)
    function_name_end_ids = find_keywords(tokenizer, keywords='"parameters": {"type": "object", ', prompt=prompt, input_ids=input_ids)
    function_argument_start_ids = find_keywords(tokenizer, keywords='"properties": {"', prompt=prompt, input_ids=input_ids)
    function_argument_end_ids = find_keywords(tokenizer, keywords=', "required": [', prompt=prompt, input_ids=input_ids)

    assert len(function_name_start_ids) == len(function_name_end_ids) == len(function_argument_start_ids) == len(function_argument_end_ids) == num_function
    try:
        template_end_ids = search_for_target_sequence(tokenizer, user_question_start_str, prompt, input_ids)
    except:
        template_end_ids = search_for_target_sequence(tokenizer, user_question_start_str[:-2], prompt, input_ids) # match for 
    input_end_ids = search_for_target_sequence(tokenizer, assistant_start_str, prompt, input_ids)
    
    return function_name_start_ids, function_name_end_ids, function_argument_start_ids, function_argument_end_ids, template_end_ids, input_end_ids


def extract_llm_graphs(
    input_file:str,
    output_directory:str, 
    function_template_file_path:str,
    limit:int=3,
    model=llm(),
    function_start_str:str="###<<functions>>",
    user_question_start_str:str="question",
    assistant_start_str:str="assistant",
    label_map_file_path:str=""
) -> None:
    data = load_data(input_file)
    labels = data['label'] 
    inputs = data['input'] 
    responses = []
    graph_latency = []
    response_latency = []
    if not label_map_file_path:
        label_mapping = {value: idx for idx, value in enumerate(list(set(labels)))}
    else:
        with open(label_map_file_path, "r") as f:
            label_mapping=json.load(f)
    
    attention_graphs = []
    function_templates = load_function_templates(function_template_file_path)
    
    if limit == -1: # map default to full examples for each skill
        limit = len(data) # the length of all examples
    
    # refactor the code to have two columns, one for examples, one for reference skills
    for idx, (input, label) in tqdm(enumerate(zip(inputs, labels)), total=limit):
        # for idx, (each_exp, correct_skill, cot1, cot2) in enumerate(data.values.tolist()): for RAG+CoT
        if idx < limit:
            prompt = model.format_conversation(
                messages=[{"role": "user", "content": input}],
                functions = function_templates
            )
            try:
                start = time()
                input_ids, outputs  = model._extract_latent_features(prompt=prompt, max_tokens=1)
                # # This is more efficient than calling the LLM again however; we need to measure the time it takes to generate the graph
                # print(input_ids, response)
                # response = model.tokenizer.decode(outputs)
                # print(response)
                
                # first get start and end ids and then construct graph
                function_name_start_ids, function_name_end_ids, function_argument_start_ids, function_argument_end_ids, template_end_ids, input_end_ids = get_function_token_offset(
                    model.tokenizer,
                    input_ids,
                    prompt,
                    num_function=len(function_templates),
                    user_question_start_str=user_question_start_str,
                    assistant_start_str=assistant_start_str,
                )

                # Build a graph with a graph contraction grouping each function and user token ids
                attention_graph = construct_graph(
                    input_ids,
                    outputs,
                    function_name_start_ids,
                    function_name_end_ids,
                    function_argument_start_ids,
                    function_argument_end_ids,
                    template_end_ids,
                    input_end_ids,
                    num_function=len(function_templates)
                )
                
                # Extract entire attention submatrix over a contiguous range including all function tokens and all user tokens [without any quotienting]; this function below is for extracting the coarse-grained graph
                # attention_graph = extract_llm_graph(
                #     model.tokenizer,
                #     input_ids,
                #     outputs,
                #     prompt,
                #     function_start_str,
                #     user_question_start_str,
                #     assistant_start_str
                # )
                attention_graph.y = label_mapping[label]
                attention_graph.detach().to(device("cpu"))
                attention_graphs.append(attention_graph)
                empty_cache()
                end = time()
                graph_latency.append(end-start)
            
            except Exception as e:
                print(e)
                attention_graphs.append([])
                graph_latency.append((-1))
            
            try:
                start = time()
                response = model._generate_response(
                    prompt=prompt,
                    max_tokens=128
                )
                responses.append(response)
                end = time()
                
                response_latency.append(end-start)
            
            except Exception as e:
                print(e)
                responses.append("")
                response_latency.append(-1)
        else:
            attention_graphs.append([])
            responses.append("")
            graph_latency.append(-1)
            response_latency.append(-1)

    # save data
    DataFrame({
        "inputs": inputs,
        "label": labels,
        "responses": responses,
        "graph_latency": graph_latency,
        "response_latency": response_latency
    }).to_csv(Path(output_directory, "responses.csv"), index=False, header=True)
    with open(Path(output_directory, "attention_graphs.pkl"), "wb") as f:
        pickle.dump(attention_graphs, f)
    with open(Path(output_directory, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)

    return attention_graphs, label_mapping

if __name__ == "__main__":
    input_args = extract_arguments()
    attention_graphs, label_mapping = extract_llm_graphs(
        input_file=input_args.input_file, 
        output_directory=input_args.output_directory, 
        function_template_file_path=input_args.function_template_file_path,
        limit=input_args.limit,
        model=llm(),
        label_map_file_path=input_args.label_map_file_path,
        function_start_str=input_args.function_start_str,
        user_question_start_str=input_args.user_question_start_str,
        assistant_start_str=input_args.assistant_start_str,
    )

    with open(Path(input_args.output_directory, "attention_graphs.pkl"), "wb") as f:
        pickle.dump(attention_graphs, f)
    with open(Path(input_args.output_directory, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)

# Example usage for Llama3.1 Instruct
# python -m experiments.orchestration.latent_graph_extraction --input_file="/mount/splunka100groupstorage/a100-fs-share1/lbetthauser/data/spl/splits/test_split_15.csv" --function_template_file_path="/mount/splunka100groupstorage/a100-fs-share1/lbetthauser/orchestration/hybrid_orchestration/spl/v1/function_metadata.json" --output_directory=/mount/splunka100groupstorage/a100-fs-share1/lbetthauser/model_data/spl/graph_contraction/ming_split_test --limit=-1 --label_map_file_path=/mount/splunka100groupstorage/a100-fs-share1/lbetthauser/data/spl/splits/label_mapping.json --function_start_str="### Functions" --user_question_start_str="<|eot_id|><|start_header_id|>user<|end_header_id|>" --assistant_start_str="<|start_header_id|>assistant<|end_header_id|>"