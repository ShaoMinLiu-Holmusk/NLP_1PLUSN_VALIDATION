"""
# JSON parsing

Each label for the sentence is assumed to have the following 
structure:
[label], negation, historical, assertion, annotationSubtype    
    # apparently there is a mistake in the raw data
    # not all in consistent format of `Type - sublabel`
    # some have no hypen
    # some have missing spaces infront or after the hypen

For now, its likely that assertion and annotationSubtype does not 
contain useful information, if we were to extract even just 
{label, negation, historical}:
    for N unique label options, and D unique documents,
    we will need to generate a 
    D * N * 3 matrix -> 3 because we need one layer for the flag
    of label, 1 layer for negation flag,last layer for historical flag
    This is too complex and not scalable. 
    
Thus the current solution is to parse all these JSON files into a 
single JSON files in the followig structure:

[
    {
        user: john,
        document: ajoanne-11344605-692014-309976-1696-11344605-.json,
        labels: [
            MSEtype - subLabel, negation, historical, assertion, annotationSubtype,
            violent bahviour, suicidal, negation, historical, assertion, annotationSubtype,
            ...
        ]
    },
    {
        user: jane,
        document: ajoanne-170088225-11212016-372590-1696-170088225-.json,
        labels: [
            ...
        ]
    }
    ...
]

This is so that the label status of each all the documents are aggregated into a single 
JSON, and when we need the information about only the `label + negation`

# Report
Given the JSON result in the structure above, the report aims to 
produce a summary of the data that is present in the documents.

Report includes:
    How many documents are there.
    
    How many distinct labels are present
    How many of each MSE category is present, 
    For each MSE category, how many sub-categories, each of how many
    
    Negation/Historical flag occurs in how many of the categories?
"""