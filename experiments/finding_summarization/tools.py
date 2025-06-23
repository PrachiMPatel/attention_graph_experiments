from collections import defaultdict
import json
import re
from transformers import AutoTokenizer
from typing import Any

from tooling.agents.skill import Skill
from tooling.llm_engine import GenerativeModel

def generate_subcontexts(context:dict[str,Any], subcontext_keys:dict[str,list[str]]) -> list[dict[str,Any]]:
    return { 
        subcontext_name: {key: context[key] for key in keys}
    for subcontext_name, keys in subcontext_keys.items()}


def call_llm(model:GenerativeModel, messages:list[dict[str,str]], **kwargs)-> tuple[str,int]:
    prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model.tokenizer.tokenize(prompt)
    model_output = model._generate_response(inputs, **kwargs)
    return model_output, len(inputs)


def compose_actions(context:dict[str,Any]) -> bool:
    return True


def merge_dictionaries(output_key:str, **kwargs:dict[str,Any]) -> dict[str,Any]:
    return {
        output_key: list(kwargs.values())
    }
    
def extract_mitre_data(investigation:dict[str,Any]) -> dict[str,Any]:
    return investigation["findings"]
    # mitre_keys = set([])
    # mitre_dictionary = defaultdict(list[str])
    # if "consolidated_summary" in investigation:
    #     mitre_dictionary["consolidated_summary"]=investigation["system_data"].pop("consolidated_summary")
    # for finding in investigation["findings"]:
    #     if isinstance(finding,dict):
    #         for k in finding.keys():
    #             normalized_key = k.lower()
    #             if "mitre" in normalized_key or "summary" in normalized_key:
    #                 mitre_dictionary[k].append(finding[k])
    #                 mitre_keys.add(k)
    # return mitre_dictionary
    

def parse_investigation(investigation:dict[str,Any], system_keys:list[str]=[]) -> dict[str,Any]:
    return {
        "system_subcontext":investigation,
        # "system_subcontext":extract_splunk_system_data(investigation),
        "mitre_subcontext":extract_mitre_data(investigation)
    }


def search_nested_dict(nested_dict, query:str):
    """
    Search a nested dictionary by using dot notation.

    Args:
        nested_dict (dict): The dictionary to search in.
        query (str): The dot notation string to search for. e.g., "a.b.c"

    Returns:
        Any: The value found at the specified key path, or None if not found.
    """
    keys = re.split(r'\.', query)
    current_dict = nested_dict

    # Iterate over each key in the query
    for key in keys:
        if isinstance(current_dict, dict) and key in current_dict:
            # If the current dictionary has the next key, move to it
            current_dict = current_dict[key]
        else:
            # If we can't find the next key, stop searching
            return None

    # At this point, we've found all keys and are at the final value
    return current_dict

def extract_splunk_system_data(investigation_data:dict[str,Any], system_data_map:dict[str,str]=METADATA_MAPPING, output_key:str=None) -> dict[str,Any]:
    system_data = defaultdict(list)
    
    # parse metadata
    investigation = investigation_data["investigation"]
    findings = investigation_data["findings"]

    # extract investigation name
    system_data["Investigation Name"] = "{display_id}:{investigation_name}".format(display_id=investigation["display_id"], investigation_name=investigation["name"])
    
    # find time frame from finding
    def parse_finding(finding:dict[str,Any], field_mapping:dict[str,str]) -> dict[str,Any]:
        finding_data = {}
        for key, value in field_mapping.items():
            finding_data[key] = search_nested_dict(finding, value)
        return finding_data
        
    if findings:
        parsed_findings = [parse_finding(finding, system_data_map) for finding in findings]
        for parsed_finding in parsed_findings:
            for key in system_data_map.keys():
                if key in parsed_finding:
                    system_data[key].append(parsed_finding[key])
    
    system_data = dict(system_data)
    return {output_key: system_data} if output_key else system_data


def merge_finding(findings:dict[str,Any], keys:list[str]):
    merged_findings = {k:[] for k in keys}
    for finding in findings:
        if isinstance(finding,dict):
            for k in keys:
                if k in finding.keys():
                    merged_findings[k].append(finding[k])
    return merged_findings

# assess if necessary information is contained in the input string
# used as a condition function on edges to determine if additional processing is required
class Evaluator(Skill):
    # static skill definitions
    skill_name = "evaluate_condition"
    description: str = "Verify a given input request with respects to a given context and returns a value."
    required_contexts: list[str] = []
    skill_metadata: dict = {}
    # oss_function: ClassVar[dict] = None
    # oai_function: ClassVar[dict] = None

    prompt: str = "### Instructions\n\n{instructions}\n\n### Context\n\n{context}"
    system_message: str = "You are a helpful assistant. Use the following instructions to respond to the users with a boolean. Please respond with a JSON of the form {'response': [True/False]}."
    contexts: dict = {}
    stream: bool = False
    use_oss: bool = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.skill_metadata = self.evaluate.__annotations__

    def get_required_contexts(self):
        return self.required_contexts

    def get_prompt(self):
        return self.prompt
    
    def build_prompt(self, tokenizer:AutoTokenizer, **kwargs):
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.get_prompt().format(**kwargs)}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def parse_response(self, output:str):
        # Regular expression pattern to extract the JSON-like string
        pattern = r'{.*"decision":(true|false),.*}'
        
        # Use the regular expression to search for the pattern in the input string
        match = re.search(pattern, output)
        
        if match:
            # Extract the matched JSON-like string
            json_like_string = match.group()
            
            try:
                # Parse the extracted JSON-like string using json.loads()
                data = json.loads(json_like_string)
                
                # Return the value of the "decision" key
                decision = data['decision']
                return decision
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return None
        else:
            # If no match is found, return None or raise an exception depending on your requirements
            return None
    
    def evaluate(self, model:GenerativeModel, instructions:str, context:str, **kwargs) -> bool:
        try:
            # construct prompt
            prompt = self.build_prompt(tokenizer=model.tokenizer, instructions=instructions, context=context)
            # generate model response
            output = model._generate_response(prompt, **kwargs)
            # parse response and if the value is not True or no decision can be parsed
            decision = self.parsed_response(output) # this value will be None rather than False if the decision cannot be parsed properly
            decision = decision if decision else False
        except:
            decision = False
        
        return decision

    def execute_skill(self, model:GenerativeModel, instructions:str, context:str, **kwargs) -> bool:
        return self.evaluate(model=model, instructions=instructions, context=context, **kwargs)


# create skills
class InformationExtractor(Skill):
    # static skill definitions
    skill_name = "information_extractor"
    description: str = "select an available action"
    required_contexts: list[str] = []
    skill_metadata: dict = {}
    # oss_function: ClassVar[dict] = None
    # oai_function: ClassVar[dict] = None

    prompt: str = """### Instructions
    {instructions}
    Format the following 'Context' JSON(s).
    
    ### Context
    {context}"""
    system_message: str = """You are a helpful assistant. Use the following instructions to retrieve the relevant portions of the context and format the response. Your response should only be a JSON with the necessary keys specified in the instructions."""
    contexts: dict = {}
    stream: bool = False
    use_oss: bool = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.skill_metadata = self.extract_data.__annotations__

    def get_required_contexts(self):
        return self.required_contexts

    def get_prompt(self):
        return self.prompt

    def parse_response(self, output:str):
        try:
            # Parse the extracted JSON-like string using json.loads()
            return json.loads(output)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None

    def build_messages(self, **kwargs):
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.get_prompt().format(**kwargs)}
        ]

    def build_prompt(self, tokenizer:AutoTokenizer, **kwargs):
        messages = self.build_messages(**kwargs)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def extract_data_local_model(self, model:GenerativeModel, instructions:str, context:str, **kwargs) -> dict[str,Any]:
        try:
            # construct prompt
            prompt = self.build_prompt(tokenizer=model.tokenizer, instructions=instructions, context=context)
            # generate model response
            import pdb; pdb.set_trace()
            output = model._generate_response(prompt, **kwargs)
            return output
            # parse response and if the value is not True or no decision can be parsed
            output_dictionary = self.parsed_response(output) # this value will be None rather than {} if the decision cannot be parsed properly
            output_dictionary = output_dictionary if output_dictionary else {}
        except:
            output_dictionary = {}
    
    def extract_data(self, model:GenerativeModel, instructions:str, context:str, **kwargs) -> dict[str,Any]:
        try:
            # construct prompt
            messages = self.build_messages(instructions=instructions, context=context)
            # generate model response
            response = model.call_llm_chat(messages, **kwargs)
            output = model.parse_response(response)
            return output
        except Exception as e: 
            output_dictionary = {"python_error": e}
        
        return output_dictionary

    def execute_skill(self, model:GenerativeModel, instructions:str, context:str, output_key:str, **kwargs) -> dict[str,Any]:
        return {
            output_key: self.extract_data(model=model, instructions=instructions, context=context, **kwargs)
        }

    
# merges retrieved information dictionaries
class Formatter(Skill):
    # static skill definitions
    skill_name = "response_formatter"
    description: str = "Gather multiple summaries and a prompt containing instructions and aggregate into a single output."
    required_contexts: list[str] = []
    skill_metadata: dict = {}
    # oss_function: ClassVar[dict] = None
    # oai_function: ClassVar[dict] = None

    prompt: str = """### Instructions
    {instructions}
    Format the following 'Context' JSON(s).
    
    ### Context
    {context}"""
    system_message: str = """You are a helpful assistant. Use the following instructions to retrieve the relevant portions of the context and format the response. Your response should be concise and precisely follow the format specified in the instructions."""
    failure_message: str = "Sorry I was not able aggregate the structured summaries."
    ai_disclaimer_message: str = "\n\nThis is an AI generated summary of the available information, and further investigation may be required to fully understand the finding."
    contexts: dict = {}
    stream: bool = False
    use_oss: bool = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.skill_metadata = self.merge_responses.__annotations__

    def get_required_contexts(self):
        return self.required_contexts

    def get_prompt(self):
        return self.prompt

    def build_messages(self, **kwargs):
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.get_prompt().format(**kwargs)}
        ]

    def build_prompt(self, tokenizer:AutoTokenizer, **kwargs):
        messages = self.build_messages(**kwargs)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # def merge_responses_local_model(self, model:GenerativeModel, instructions:str, context:str, **kwargs) -> str:
    #     try:
    #         # construct prompt
    #         prompt = self.build_prompt(tokenizer=model.tokenizer, instructions=instructions, context=context)
    #         # generate model response
    #         output = model._generate_response(prompt, **kwargs)
    #         truncated_output = ".".join(output.split(".")[:-1]) # cut response to complete sentences
    #         formatted_output = truncated_output + self.ai_disclaimer_message
            
    #     except:
    #         formatted_output = self.failure_message
        
    #     return formatted_output
    
    def merge_responses(self, model:GenerativeModel, instructions:str, context:str, **kwargs) -> str:
        try:
            # construct prompt
            messages = self.build_messages(instructions=instructions, context=context)
            # generate model response
            response = model.call_llm_chat(messages, **kwargs)
            output = model.parse_response(response)
            truncated_output = ".".join(output.split(".")[:-1]) # cut response to complete sentences
            formatted_output = truncated_output + self.ai_disclaimer_message
            
        except:
            formatted_output = self.failure_message
        
        return formatted_output

    def execute_skill(self, model:GenerativeModel, instructions:str, context:str, output_key:str, **kwargs) -> str:
        return {
            "finding_ai_summary": self.merge_responses(model=model, instructions=instructions, context=context, **kwargs)
        }