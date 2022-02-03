import sys
sys.path.insert(0,'../src/')

from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from lib.Toolbox.writerWrap import writeYASON
from pathlib import Path
from datetime import datetime

script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

moduleConfig = config['configVersions'][moduleName]
moduleConfig = readYASON(moduleConfig)
runID = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
tag = moduleConfig['output']['posfix'] # each run has an timebased unique ID 
runID = '_'.join((runID,tag)) if tag else runID # ID appened with tag(optional)
outputLocation = Path(moduleConfig['output']['location']) / runID

import pandas as pd
import numpy as np
from lib.Toolbox.loggerWrap import CSVwriter, SimplyLog
from lib.Toolbox.readerWrap import readYASON
from itertools import repeat
from tqdm import tqdm
from holbert import holbert
from modules.feature_competition.competition import evaluate

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    myLogger = SimplyLog(outputLocation)
    myCSVWriter = CSVwriter(outputLocation)
    
    outputLocation.mkdir(parents=True, exist_ok=True)
    myLogger('run').info('Module booted, should not appear on console')
    myLogger('run').warning('warning trial, should appear on console, and print to log')
    
    if not moduleConfig['validationDataSet']:
        modelDataUsage = pd.read_pickle(moduleConfig['modelDataUsage'])
        modelDataUsage = modelDataUsage[['id', 'flag']]
        modelDataUsage = modelDataUsage[modelDataUsage.flag == 'val']
        myLogger('run').info(f'{modelDataUsage.shape[0]} validation sentences found')
        
        masterData = pd.read_pickle(moduleConfig['masterData'])
        masterData = masterData[['id', 'sentence', 'mseLab']]
        
        dataMatches = pd.merge(modelDataUsage, masterData,
                            how='left',
                            on='id')
        if dataMatches.shape[0] != modelDataUsage.shape[0]:
            myLogger('run').warning('data does not match, please verify data sources')
        if dataMatches.sentence.isna().sum() > 0:
            myLogger('run').warning('some sentences not found, please verify data sources')
            
        dataMatches.to_pickle(outputLocation / 'validationSentences.p')
    else:
        dataMatches = pd.read_pickle(moduleConfig['validationDataSet'])
    
    # N sentences * 86 labels
    dataMatchesGroundTruth = np.array(dataMatches.mseLab.tolist())
    
    problemSet = readYASON(moduleConfig['testProblemSet'])
    for problemCSV, problemIndex in problemSet.items():
        positiveCases = dataMatchesGroundTruth[:,problemIndex].any(axis=1).sum()
        myLogger('run').info(f'{problemCSV}  {positiveCases} positive cases')
    
    thisCSV = myCSVWriter('validationDataLabelCount', 
                          ['label', 'count'],
                          keepacopy=True)
    
    labelKeys = readYASON(moduleConfig['labelKeys'])
    for index, labelText in labelKeys.items():
        index = int(index)
        positiveCases = dataMatchesGroundTruth[:,index].sum()
        thisCSV.writeRow([labelText, positiveCases])
    thisCSV = thisCSV.getDataFrame()
    
    if not moduleConfig['holbertFeatures']:
        columns = zip(repeat('feature_'), range(86))
        columns = list(name+str(num) for name, num in columns)
        columns.append('sentenceID')
        thisCSV = myCSVWriter('holBertFeaturesLive', columns)
        # convertAll the
        myLogger('run').info('Instantiating Holbert')
        model = holbert(moduleConfig['holBertConfig'])
        myLogger('run').info('Instantiated Holbert')
        
        dataFrame = []
        for index, row in tqdm(dataMatches.iterrows(), 
                            total=dataMatches.shape[0],
                            ncols=50,
                            mininterval=2):
            result = model.processDocument(row['sentence'])
            mseLab = result.MseClassifier.mseLab.sparseProb > 0
            mseLab = mseLab.any(axis=0).astype(int).tolist() # a single list
            mseLab.append(row['id']) # add the sentence id
            thisCSV.writeRow(mseLab)
            dataFrame.append(mseLab)
            
        holbertFeatures = pd.DataFrame(dataFrame, 
                                       columns=columns)
        holbertFeatures.to_csv(outputLocation/'holBertFeatures.csv',
                               index=False)
    else:
        holbertFeatures = pd.read_csv(moduleConfig['holbertFeatures'])
    
    # exclude the sentenceId in array
    holbertPrediction = holbertFeatures[holbertFeatures.columns[:-1]].values
    assert dataMatchesGroundTruth.shape == holbertPrediction.shape, 'data not the same'

    resultCSV = myCSVWriter('metricForEachFeature', 
                        ['label', 'count', 
                         'f1', 'ROCauc', 'PRauc', 
                         'precision','recall', 'accuracy'
                         ],
                        keepacopy=True)
    labelKeys = readYASON(moduleConfig['labelKeys'])
    for index, labelText in labelKeys.items():
        index = int(index)
        positiveCases = dataMatchesGroundTruth[:,index].sum()
        metrics = evaluate(dataMatchesGroundTruth[:,index],
                           holbertPrediction[:,index])
        resultCSV.writeRow([
            labelText, positiveCases,
            metrics['f1'],
            metrics['ROCauc'],
            metrics['PRauc'],
            metrics['precision'],
            metrics['recall'],
            metrics['accuracy']
        ])
    # keep a copy of the config file for regeneration
    writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)
    
if __name__ == '__main__':
    main('resultsDict')
    