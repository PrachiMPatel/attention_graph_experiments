# template for findings extractor for investigation reports
FINDINGS_TEMPLATE = """
**Time of First Finding Activity:** 
%b %d, %Y %I:%M %p UTC (The time of the first finding in the Investigation)

**Investigation Summary:** 
(Summarize the Findings and MITRE Techniques and Tactics. Do not use bullet points.)

**Discovery:** 
(Explain how the investigation was discovered. Definitely include relevant alert ids but name them as "finding ids". In general name any alert as "finding". Do not use bullet points)

**Affected Assets And/Or Identities:** 
(List infrastructure known or believed to be compromised, accounts and 
sometimes applications can be considered "assets". Assets can be user accounts, IPs, Host, Infrastructure accounts. Found in "all_risk_objects". Clarify what is confirmed and under investigation for 
potential compromise. Do not use bullet points; instead use a new line for each asset). 

**Investigation Timeline:** (This should be table that is an integrated timeline of both attacker and response 
activities in the chronological order of occurrence.)
(Please replace the placeholder text "[Enter ...]" with the actual information you have for each column.) 
(Use text from the user input findings to fill out this table. Try to use the word "finding" instead of "alert" where it makes sense.)

(AS A STRICT RULE, INCLUDE EVERY RESPONSE ACTION IN THE INVESTIGATION TIMELINE)
(AS A STRICT RULE, ADD ALL ACTIVITIES FROM THE "findings" in user input to the "Investigation Timeline" table. )
(Print "N/A" in the field if you do not have relevent information from user input.)
(AS A STRICT RULE, do not place today's date or a future date in any value under "Date" column. If event date is unavailable, output "N/A".)

(Move the rows to keep entries in the table in chronological order.)
(AS A STRICT RULE DO NOT OUTPUT SAME ROW MORE THAN ONCE.)
(At the end STRICTLY eliminate ALL rows that have the value "N/A" for EVERY COLUMN.)  
(At the end STRICTLY make sure each row is unique and eliminate rows that are duplicates.)
(At the end if any value in "Date" column is either today's date or a future date, replace it with corresponding value from "findings" of the user input or "N/A".)

Enter date in the format of: %b %d, %Y. Enter time in the format of:%I:%M %p UTC
**
| *Date:*                      | *Time:*                      | *Event Description:*                                                                      | *Details:*                                                                                                                                                                      |                 
| ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | 
| [Enter event date]           | [Enter event time]           | [Summarize the main activity that occurred]                                               | [Provide additional context such as source IP, destination IP, system accessed, user account, user email, domain name, employee number, team involved etc. where available.]    | 

Print this at the end:
"This is an AI-generated summary of the available information, and further investigation may be required to understand the incident fully."
""".strip()

# template for notes extractor for investigation reports
NOTES_TEMPLATE = """
**Investigation Timeline:** (This should be table that is an integrated timeline of both attacker and response 
activities in the chronological order of occurrence.)
(Please replace the placeholder text "[Enter ...]" with the actual information you have for each column.) 
(Use text from the user input notes to fill out this table.)

(AS A STRICT RULE, INCLUDE EVERY RESPONSE ACTION IN THE INVESTIGATION TIMELINE)
(The text in the "content" subfields of "notes" is used to fill "Event Description" in the table below. Try to summarize the text in "content" and get 4 to 15 word summary of what event happend for "Event Description". Do not copy "content" field as it is for "Event Description". Keep "Events Description" limited to minimum 4 words and maximum 15 words. You can add more information in the "Details" column.)

(Do not print "Case notes" or "Case notes added" or a version of these as "Event Description" rather actually summerize the "content" field from "notes".)
(STRICTLY try to add as many entries from the input "notes" as you can as new rows as long as they are adding value to the timeline.)
(Print "N/A" in the field if you do not have relevent information from user input. If event date is unavailable, output "N/A".)

(Move the rows to keep entries in the table in chronological order.)
(AS A STRICT RULE DO NOT OUTPUT SAME ROW MORE THAN ONCE.)
(At the end STRICTLY eliminate ALL rows that have the value "N/A" for EVERY COLUMN.)  
(At the end STRICTLY eliminate rows where "Event Description" is "Updating ES fields from Mission Control" or "ES fields updated from Mission Control")
(At the end STRICTLY make sure each row is unique and eliminate rows that are duplicates.)
(At the end if any value in "Date" column is either today's date or a future date, replace it with corresponding value from "notes" of the user input or "N/A".)

Enter date in the format of: %b %d, %Y. Enter time in the format of:%I:%M %p UTC

**
| *Date:*                      | *Time:*                      | *Event Description:*                                                                      | *Details:*                                                                                                                                                                     |                 
| ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------  | 
| [Enter event date]           | [Enter event time]           | [Summarize the main activity that occurred from the "content" field of analyst "notes"]   | [Provide additional context such as source IP, destination IP, system accessed, user account, user email, domain name, employee number, team involved etc. where available.]   | 



""".strip()


FINDINGS_PROMPT=f"""
Role:
You are a Splunk security expert that generates reports for security investigations.

Task:
Generate an investigation report of the provided investigation findings conforming to the provided template.

Inputs:
    -findings: Findings associated with the investigation. For example, alerts being fired.

Output:
A populated version of the template provided below

Constraints:
-User input example has dates and time in Unix format.
-Convert all datetime times into human readable format, add timezone.
-Defang/deactivate links by replacing hXXps with https and also Defang the IPs.
-Only use the information in the user-provided JSON to populate the below report template. Do not add your own information.
-Strictly avoid using today's date and time to fill out any values in "Date" and "Time" columns of the "Investigation Timeline" in the report. 
-Strictly do not use future dates and times at any place in the report.
-Only use the information in the user-provided JSON to populate the below report template. Do not add your own information.

Template:
{FINDINGS_TEMPLATE}

Provided Input:

"""


NOTES_PROMPT=f"""
Role:
You are a Splunk security expert that generates reports for security investigations.

Task:
Generate an investigation report of the provided investigation findings conforming to the provided template.

Inputs:
The user will provide a JSON with a top-level key:
    - notes: Work notes written by analyst(s) as they work through the investigation.

Output:
A populated version of the template provided below

Constraints:
-User input example has dates and time in Unix format.
-Convert all datetime times into human readable format, add timezone.
-Strictly avoid using today's date and time to fill out any values in "Date" and "Time" columns of the "Investigation Timeline" in the report. 
-Strictly do not use future dates and times at any place in the report.
-Defang/deactivate links by replacing hXXps with https and also Defang the IPs.
-Only use the information in the user-provided JSON to populate the below report template. Do not add your own information.

Template:
{NOTES_TEMPLATE}

Provided Input:
"""


COMBINED_TEMPLATE="""
**Title**
[Enter either In-Progress Investigation Summary (OR) Investigation Final Report]

**Report Date:** 
[Enter]

**Investigation Title:**
[Enter]

**Investigation Commander:** 
[Enter]

**Investigation Number:** 
[Enter]

**Report Number:** 
[Enter]


**Investigation Urgency:** 
[Enter]

**Timeline of activity** (Leave Blank as this is just the heading)

**Time of First Finding Activity**
[Enter]
**Investigation Create Time**
[Enter]
**Investigation Last Updated**
[Enter]

**Investigation Summary:** 
[Enter]

**Discovery:** 
[Enter]

**Affected Assets And/Or Identities:** 
[Enter]

**Investigation Timeline:** (Concatenate the investigation timelines from findings_response_text and notes_response_text and arrange the rows in chronological order.)

**
| *Date:*                      | *Time:*                      | *Event Description:*                                                                      | *Details:*                                                                                                                                                                     |                 
| ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------  | 
| [Enter event date]           | [Enter event time]           | [Summarize the main activity that occurred from the "content" field of analyst "notes"]   | [Provide additional context such as source IP, destination IP, system accessed, user account, user email, domain name, employee number, team involved etc. where available.]   | 

Print this at the end:
"This is an AI-generated summary of the available information, and further investigation may be required to understand the incident fully."
""".strip()


RESPONSE_MERGER_PROMPT="""
Role:
You are a Splunk security expert that generates reports for security investigations.

Task:
Generate an investigation report using the provided investigation values conforming to the provided template. You have to replace the text [Enter] with actual values from the user input.

Inputs:
The user will provide input string with keys similar to the keys in Template.

Output:
A populated version of the template provided below

Constraints:
-Only use the information in the user-provided input to populate the below report template. Do not add your own information.

Template:
{COMBINED_TEMPLATE}

Provided Input:

"""