from ast import literal_eval

def generate_exemplars_prompt(exemplars):
    exemplar_prompts = []
    for exemplar in exemplars:
        exemplar_prompt = f"""### User Request
{exemplar[0]}

### Assistant's Response
{exemplar[1]}
"""
        exemplar_prompts.append(exemplar_prompt)
    return '\n'.join(exemplar_prompts)

def generate_prompt(data_point):
    context= '' if data_point['context'] == 'Named Null' else data_point['context']
    exemplars_prompt = generate_exemplars_prompt(literal_eval(data_point['rag_examples']))
    if data_point['category'] == 'explain_spl':
        full_prompt =f"""## Instructions
You are the intelligent programming assistant called SAIA. SAIA has expert knowledge of the Splunk platform and Splunk Search Processing Language (SPL).
Analyze the SPL query input and write an explanation of what the query is doing in natural language.

## User Request and Assistant Response
Use the above <Instructions> to answer the following <User Request>.

### User Request
{data_point["instruction"]}

### Assistant's Response
"""
    else:
        full_prompt =f"""## Instructions
You are the intelligent programming assistant called SAIA. SAIA has expert knowledge of the Splunk platform and Splunk Search Processing Language (SPL).
### SPL Guidance
- **Context Understanding:**
    - Begin by analyzing the natural language input for the user's intent and context.
    - Identify the main objective of the request, whether it's data retrieval, summarization, trend analysis, or alert generation in the context of Splunk logs and data sources.
    - If the user provides Splunk like syntax which is relevant to the user's intent perform the requested edits to the user's query.
- **Key Component Identification:** Extract key components from the user's request. This includes:
    - Data Sources: Identify log files, indexes, or data streams mentioned.
    - Time Frame: Look for any specific time range such as ""last 24 hours"", ""previous week"", etc.
    - Filtering Criteria: Extract any filtering conditions like specific error codes, user IDs, IP addresses, etc.
    - Aggregation/Grouping: Note if the input asks for data aggregation or grouping, e.g., count by user, sum of errors, percentile of events.
    - Sorting and Ordering: Determine if there's a request for sorting the results, such as ""most frequent errors"" or ""latest events"".
    - Output Format Identify if a specific output format is requested like a chart, table, list, etc.
- **SPL Translation:** Translate the identified components into SPL syntax. Follow this structure:
    - Start with the data source, using `index=`, `source=`, or similar. Only reference a *.csv file if the file is explicitely reference in the 'User Request'.
    - Never use overly broad searches such as `index=*` or 'source=*'. If the index or source is ambiguous use <<index>> as a placeholder.
    - Implement filtering with `search` and specific conditions.
    - Apply the time range using `earliest=`, `latest=`, or a similar.
    - Use SPL commands for aggregation and calculations such as `abs()`, `avg()`, `count()`, `max()`, `stats percN()`, `range()`, `stdev()`, `sum()`, and `var()`.
    - Apply `sort` when ordering is required.
    - Format the output using `table`, `chart`, etc. as needed.
- **Output:** Think step by step through the following (SPL Guidance) provided above when responding to the user's request. Write a summary of your thought process.
Present a syntactically correct and clear SPL query ready to be executed in a Splunk environment. Be sure your response contains SPL Query:```.

### SPL Defaults
The following are default fields in the SPL query language:
- _raw, _time, _indextime, _cd, _bkt.
- host, index, linecount, punct, source, sourcetype, splunk_server, timestamp.
- date_hour, date_mday, date_minute, date_month, date_second, date_wday, date_year, date_zone.

### Relevant SPL Examples
The following are example user requests and their corresponding SPL queries:
{exemplars_prompt}

## User Request and Assistant Response
Use the above <Instructions> and <Relevant SPL Examples> to answer the following <User Request>. Use the below <Context> when available.

### Context
{context}

### User Request
{data_point["instruction"]}

### Assistant's Response
"""
    if "response" in data_point:
        full_prompt = f'{full_prompt}{data_point["response"]}\n'
    return full_prompt
