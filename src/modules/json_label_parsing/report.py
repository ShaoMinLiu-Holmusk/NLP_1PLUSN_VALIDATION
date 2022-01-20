from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from pathlib import Path, PosixPath

script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from re import split

@lD.log(logBase + '.report')
def report(logger, writeDirectory:PosixPath)->dict:
    print('reading document')
    labelFile = readYASON(writeDirectory)
    
    docCount = documentCount(labelFile)
    labelCounter, MSETopicCounter, subTopicCount, negationCount,historicalCount = analyseLabels(labelFile)
    
    report = {
        'documentCount': docCount,
        'distinctLabels': {
            'labelCount' : len(labelCounter),
            'tally': labelCounter
            },
        'MSEtopics':{
            'topicCount': len(MSETopicCounter),
            'tally': MSETopicCounter,
            'subTopics': subTopicCount
        },
        'negationCount': negationCount,
        'historicalCount':historicalCount
    }
    return report

@lD.log(logBase + '.documentCount')
def documentCount(logger, labelFile:list)->int:
    return len(labelFile)

@lD.log(logBase + '.analyseLabels')
def analyseLabels(logger, labelFile:list)->int:
    labelCounter = Counter() # how many distinct labels are there
    MSETopicCounter = Counter() # how many types of MSE labels?
    subTopicCount = defaultdict(Counter) # how many subLabels for each MSE type
    
    negationCount = Counter()
    historicalCount = Counter()
    
    for document in tqdm(labelFile, 
                         desc='Analysis', 
                         total=len(labelFile)):
        for eachLabel in document['labels']:
            # MSEtype - subLabel, negation, historical, assertion, annotationSubtype,
            
            # apparently there is a mistake in the raw data
            # not all in consistent format of `Type - sublabel`
            # some have no hypen
            # some have missing spaces infront or after the hypen
            labelCounter.update([eachLabel[0]]) # the MSE labels
            try:
                mseTopic, subTopic = split(' - |- | -', eachLabel[0])
            except ValueError:
                # there is no hyper present
                mseTopic, subTopic = "|||", eachLabel[0]
            
            MSETopicCounter.update([mseTopic])
            subTopicCount[mseTopic].update([subTopic])
            
            if eachLabel[1]:
                negationCount.update([eachLabel[0]])
            if eachLabel[2]:
                historicalCount.update([eachLabel[0]])
                
    labelCounter = OrderedDict(sorted(labelCounter.items()))
    MSETopicCounter = OrderedDict(sorted(MSETopicCounter.items()))
    subTopicCount = OrderedDict(sorted(subTopicCount.items()))
    for key,valueDict in subTopicCount.items():
        valueDict = OrderedDict(sorted(valueDict.items()))
    negationCount = OrderedDict(sorted(negationCount.items()))
    historicalCount = OrderedDict(sorted(historicalCount.items()))
    
    return labelCounter, MSETopicCounter, subTopicCount, negationCount, historicalCount
            