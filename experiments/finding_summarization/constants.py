# 
METADATA_MAPPING = {
    "Timeframe": "_time",
    "Risk Object": "risk_object",
    "Severity": "severity",
    "MITRE ATT&CK Tactic": "mitre_categories",
    "MITRE ATT&CK Tactic ID": "mitre_tactic_ids",
    "MITRE ATT&CK Technique": "mitre_techniques",
    "MITRE ATT&CK Technique ID": "mitre_technique_ids",
    "Original Source": "source",
}

# define instructions for extracting MITRE information
MITRE_INSTRUCTIONS = """Task: Extract the information specified in the template from the context below and return a JSON dictionary. If none of the values are present return an empty JSON.

Contraints:
- Defang all url links in the output. For example, replace "https://" with "hXXps://"
- Do not duplicate MITRE tactics or techniques.
- Template text in square brackets are instructions, and should not be included in the output.
 
 Template:
| **Field**                  | **Value**                         |
|----------------------------|-----------------------------------|
| **MITRE ATT&CK Tactic**    | [List the name, ID, and URL of unique MITRE tactic across ALL findings.] |
|                            | [List ALL Mitre tactics. Put each tactic in it's own table row. Example:]  |
|                            | Command and Control (TA0003) (hXXps://attack.mitre.org/tactics/TA0003/) |
|                            | Credential Access (TA0006) (hXXps://attack.mitre.org/tactics/TA0006/) |
| **MITRE ATT&CK Technique** | [List the name, ID, and URL of unique MITRE technique across ALL findings.] |
|                            | [List ALL Mitre technique. Put each technique in it's own table row. Example:]  |
|                            | PowerShell (T1059.001) (hXXps://attack.mitre.org/techniques/T1059/001/) |"""

# define instructions for extracting system information
SPLUNK_SYSTEM_INSTRUCTIONS = """Task: Extract the information specified in the template from the context below and return a JSON dictionary. If none of the values are present return an empty JSON.

Constraints:
- Defang all url links in the output. For example, replace "https://" with "hXXps://"
- If the finding(s) are empty or not a Splunk Enterprise finding or Splunk RBA, respond that No Splunk Finding was provided.
- Use the following timestamp format throughout: "%b %d, %Y %I:%M %p UTC".
- Template text in square brackets are instructions, and should not be included in the output.

Template:
| **Field**                  | **Value**                         |
|----------------------------|-----------------------------------|
| **Investigation Name**     | [Name of this Investigation.]   |
| **Timeframe**              | [Min info_min_time until the max info_max_time across all findings.] |
|                            | [Example: "Jul 01, 2024 07:35 AM UTC - Jul 21, 2024 02:30 AM UTC"] |
| **Risk Object**            | [Identifier of risk object associated with the first finding.] |
| **Severity**               | [Severity level associated with the first finding.] |
| **Original Source**        | [The detection that started the investigation, which is the first finding's source.] |"""

# define instructions for merging summaries
FORMATTER_INSTRUCTIONS = """Task: Generate a concise summary of the provided JSON(s) conforming to the provided template. The output should be a single table without duplicated column headers.

Constraints:
- All summaries should be targeted at a tier-1 analyst.
- Limit summaries to at most 1 paragraph with 5 sentences.
- Template text in square brackets are instructions, and should not be included in the output.
- List information for ALL Mitre tactices and techniques in the input.
- If any of the values from the template are missing include the field name in the template with the value 'None'.

Template:
| **Field**                  | **Value**                         |
|----------------------------|-----------------------------------|
| **Investigation Name**     | [Name of this Investigation.]   |
| **Timeframe**              | [Min info_min_time until the max info_max_time across all findings.] |
|                            | [Example: "Jul 01, 2024 07:35 AM UTC - Jul 21, 2024 02:30 AM UTC"] |
| **Risk Object**            | [Identifier of risk object associated with the first finding.] |
| **Severity**               | [Severity level associated with the first finding.] |
| **MITRE ATT&CK Tactic**    | [List the name, ID, and URL of unique MITRE tactic across ALL findings.] |
|                            | [List ALL Mitre tactics. Put each tactic in it's own table row. Example:]  |
|                            | Command and Control (TA0003) (hXXps://attack.mitre.org/tactics/TA0003/) |
|                            | Credential Access (TA0006) (hXXps://attack.mitre.org/tactics/TA0006/) |
| **MITRE ATT&CK Technique** | [List the name, ID, and URL of unique MITRE technique across ALL findings.] |
|                            | [List ALL Mitre technique. Put each technique in it's own table row. Example:]  |
|                            | PowerShell (T1059.001) (hXXps://attack.mitre.org/techniques/T1059/001/) |
| **Original Source**        | [The detection that started the investigation, which is the first finding's source.] |

**Understanding the Findings:**
[Summarize the investigation, detailing the context and significance of the findings]
[The goal is to understand the potential impact and necessary action(s).]

** MITRE Analysis **
[Only include this section if MITRE info is present in the investigation.]
[Summarize the MITRE information in at most a few sentences.]
[In the summary, include actionable next step(s).]
[In the summary, suggest implication(s) of the MITRE(s). 
    Example: "These activities suggest someone is trying to delete sensitive data."]
    
[END OF TEMPLATE; DO NOT WRITE ANY MORE TEXT]"""