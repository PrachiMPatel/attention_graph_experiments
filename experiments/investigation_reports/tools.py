from collections import defaultdict
import json
import re
from transformers import AutoTokenizer
from typing import Any
from datetime import datetime, timezone
import pandas as pd
from io import StringIO
import traceback
from tooling.agents.skill import Skill
from tooling.llm_engine import GenerativeModel
from experiments.investigation_reports.constants import FINDINGS_PROMPT, FINDINGS_TEMPLATE, NOTES_TEMPLATE, NOTES_PROMPT, RESPONSE_MERGER_PROMPT


def extract_splunk_system_data_reports(investigation_data:dict):
    
    investigation = investigation_data["investigation"]
    findings = investigation_data["findings"]
    notes = investigation_data["notes"]
    metadata = investigation_data["metadata"]
    print(type(investigation), type(findings), type(notes), type(metadata))


    return {
         "investigation":investigation,
         "metadata":metadata,
         "notes":notes,
         "findings":findings
     }

def extract_data_reports(investigation: dict, findings: list, notes: list, metadata: dict)-> dict:
    system_data = {}

    report_status=int(investigation["status"])
    if report_status==5:
        report_title="Investigation Final Report"
    else:
        report_title="In-Progress Investigation Summary"
    system_data["Title"]=report_title
    timestamp=int(investigation["create_time"])
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    formatted_date=dt.strftime('%b %d, %Y %I:%M %p UTC')
    
    date_obj=datetime.strptime(formatted_date,"%b %d, %Y %I:%M %p %Z")
    # extract investigation information
    system_data["Investigation Title"]=investigation["name"]
    system_data["Investigation Commander"]=investigation["assignee"] + ' or ' +metadata["incident_metadata"]["assignee"][0]["realname"]
    system_data["Investigation Number"]=investigation["display_id"]
    system_data["Report Number"]=investigation["display_id"]+'_'+ date_obj.strftime("%Y_%m_%d_%H_%M_%S")

    system_data["Investigation Urgency"]=investigation["urgency"]
    system_data["Timeline of activity"]=""
    system_data["Investigation Create Time"]=formatted_date

    timestamp=int(investigation["update_time"])
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    formatted_date_update=dt.strftime('%b %d, %Y %I:%M %p UTC')
    system_data["Investigation Last Updated Time"]=formatted_date_update
    # print(system_data)
    output={}
    output["extract_system_data"]=system_data
    return output




def response_formatter_reports(extract_system_data:dict, findings_response_text:dict, notes_response_text:dict, model:GenerativeModel):
    
    def response_merger_api(extract_system_data:dict, findings_response_text:dict, notes_response_text:dict, model:GenerativeModel):

        PROMPT=RESPONSE_MERGER_PROMPT
        input_str=str(extract_system_data)+"\n"+ "findings_response_text:"+ str(findings_response_text) + "\n"+ "notes_response_text:"+ str(notes_response_text)
        query = "Input:" + input_str
        messages_in=[{"role":"system", "content":PROMPT},{"role":"user","content":query}]
        response = model.call_llm_chat(messages_in)
        output_text={}   
        output_text["output"]=response.choices[0].message.content
        return output_text
    

    def format_row(row,max_w):
        return "| " + " | ".join(f"{str(val).ljust(max_w[idx])}" for idx, val in enumerate(row)) + " |"
   
    
    def format_table_str(VAR:str):
        print(type(VAR))
        table_str="\n".join([line for line in VAR.splitlines() if "|" in line])

        df=pd.read_csv(StringIO(table_str),sep="|",engine="python").dropna(axis=1,how="all")

        df.columns=df.columns.str.strip()

        df=df[1:]
        if df.iloc[:,-1].isna().any():
            df=df.drop(df.columns[-1],axis=1)
        
        df=df.applymap(lambda x:x.strip() if isinstance(x,str) else x)


        na_idx=[]
        nonna_idx=[]

        df.reset_index(drop=True, inplace=True)

        for i in range(len(df)):
            if df.iloc[i][0]=="N/A" or df.iloc[i][1]=="N/A" or pd.isna(df.iloc[i][0]) or pd.isna(df.iloc[i][1]):
                na_idx.append(i)
            else:
                nonna_idx.append(i)

        if na_idx==[]: # if there are na vals in date or time

            non_na_rows=df # non_na is the main df
            non_na_rows['DateTime']=pd.to_datetime(non_na_rows['*Date:*']+' '+non_na_rows["*Time:*"], format='mixed' )#format='%b %d, %Y %I:%M %p %Z'
            df_sorted_nonna=non_na_rows.sort_values('DateTime').drop(columns='DateTime')
            df_sorted=df_sorted_nonna


        else:
            na_rows=df.loc[na_idx].reset_index(drop=True)
            non_na_rows=df.loc[nonna_idx].reset_index(drop=True)
            non_na_rows['DateTime']=pd.to_datetime(non_na_rows['*Date:*']+' '+non_na_rows["*Time:*"], format='mixed') #'%b %d, %Y %I:%M %p %Z'
            df_sorted_nonna=non_na_rows.sort_values('DateTime').drop(columns='DateTime')
            df_sorted=pd.concat([df_sorted_nonna, na_rows], ignore_index=True)

        max_w=df_sorted.applymap(len).max()




        sorted_str="| "+" | ".join(f"{col.ljust(max_w[col])}" for col in df_sorted.columns)+ " |\n"#titles

        formated_rows=[format_row(row,max_w) for _,row in df_sorted.iterrows()]
        sorted_str+="\n".join(formated_rows)


        return sorted_str
   
    def dict_to_formatted_str(d:dict):
        formatted_str=""
        for key,val in d.items():
            formatted_str += f"**{key}**\n{val}\n\n" if val else f"**{key}**\n"
        return formatted_str.strip()
   

    try:
        table1_rows=[line for line in findings_response_text.splitlines() if line.startswith('|')]
        table2_rows=[line for line in notes_response_text.splitlines() if line.startswith('|')]
        combined_table=table1_rows +table2_rows[2:]
        combined_table_str="\n".join(combined_table)
        combined_table_str_formatted=format_table_str(combined_table_str)
        combined_doc=re.sub(r'\|(?:[^|]+\|)+',combined_table_str_formatted,findings_response_text,count=1)
        report_title=extract_system_data.pop("Title")
        extract_system_data_str=report_title +"\n" + dict_to_formatted_str(extract_system_data)

        output_text={}
        print(f"{str(extract_system_data_str)}\n\n{combined_doc}")
        output_text["output"]=f"{extract_system_data_str}\n\n{combined_doc}"
    except Exception as e:
        print("ERROR in merging, making API call")
        output_text={}
        output_text["output"]=response_merger_api(extract_system_data, findings_response_text, notes_response_text, model=model)
    return output_text





def finding_fields_extractor(investigation: dict, findings: list, notes: list, metadata: dict, model:GenerativeModel):


    PROMPT=FINDINGS_PROMPT

            # get query
    investigation_data_str = json.dumps(findings, indent=4)
    query = f"Generate a report for the investigation:\n{investigation_data_str}"

            # generate model response
    messages_in=[{"role":"system", "content":PROMPT},{"role":"user","content":query}]
    response = model.call_llm_chat(messages_in)
  
    output={}
    # output["findings_response_text"]=f"{report_title}\n\n"+response.choices[0].message.content
   
    output["findings_response_text"]=response.choices[0].message.content
    return output

def notes_fields_extractor(investigation: dict, findings: list, notes: list, metadata: dict, model:GenerativeModel):
    PROMPT=NOTES_PROMPT

            # get query
    investigation_data_str = json.dumps(notes, indent=4)
    query = f"Generate a report for the investigation:\n{investigation_data_str}"
    messages_in=[{"role":"system", "content":PROMPT},{"role":"user","content":query}]
    response = model.call_llm_chat(messages_in)
    output={}
    output["notes_response_text"]=response.choices[0].message.content
    return output