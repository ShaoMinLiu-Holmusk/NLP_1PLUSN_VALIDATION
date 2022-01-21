'''
This module will take in a json output from `parsedJsonMSE`
which will have the following structure:
[
    {
        user: aescamil,
        document: ***,
        labels: [
            [label, negationFlag, historicalFlag, assertionType, annotationFlag],
            [label, negationFlag, historicalFlag, assertionType, annotationFlag],
            [label, negationFlag, historicalFlag, assertionType, annotationFlag],
            ...
        ]
        
    },
    {
        ...
    }
]

This module intent to take the above structure and create a csv file,
that has the following structure:
user, document, `specifiedLabel`, negationFlag, HistoricalFlag, assertionType, annotationFlag

The `specifiedLabel` shall be set in the module config.
Suppose you want to convert the "Suicidality" label:
{
    "Suicidality - attempt": 267,
    "Suicidality - ideation present": 493,
    "Suicidality - ideation present with intent": 54,
    "Suicidality - ideation present with means": 15,
    "Suicidality - ideation present with plan": 91,
    "Suicidality - no issue": 419,
}
You want `specifiedLabel` == 'attemptedSuicide':
    such that "Suicidality - attempt" to be considered True, otherwise False
    Specifiy in the config like this:
    convertTask: [
        {
            customLabel: attemptedSuicide,
            positive: Suicidality - attempt
        }
    ]

    such that "Suicidality - attempt" and "Suicidality - ideation present with intent"
    to be considered True, otherwise False
    Specifiy in the config like this:
    convertTask: [
        {
            customLabel: attemptedSuicide,
            positive: [
                Suicidality - attempt,
                Suicidality - ideation present with intent
                ]
        }
    ]
    
Multiple tasks can be appended to `convertTask` config to run at once. 
Each task will generate a csv file named as the customLabel, stored under
../data/intermediate/moduleName/runInstanceID_name
'''