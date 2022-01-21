from unittest import result
from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from pathlib import Path


script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

import pandas as pd
from tqdm import tqdm

@lD.log(logBase + '.executeTask')
def executeTask(logger, 
                task:dict,
                labels:dict)->pd.DataFrame:
    """The logic is to go through each document, check if any of the document's
    labels match the positive labels.
    
    If multiple matches are found, still return as true.
    If there are multiple labels, one of the label is negation, and some others
    are not negated, then the overall negation flag is still negative. 
    Negation flag will only be true if it is uninamous. 
    Same goes for historical and other flags. 
    
    Assertion type will only take the last one, not the best strategy but will
    do for now.

    Parameters
    ----------
    logger : [type]
        [description]
    task : dict
        [description]
    labels : dict
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """
    
    # task['positive'] could be a list of string or single string alone
    positiveLabels = set(task['positive'] \
        if isinstance(task['positive'], list) else (task['positive'],))
    
    results = []
    for document in tqdm(labels, 
                         total=len(labels), 
                         desc='Document Progress',
                         leave=False):
        user, documentName = document['user'], document['document']
        label, neg, hist, ass, ann = 0,1,1,'',1
        for _label, _neg, _hist, _ass, _ann in document['labels']:
            if _label not in positiveLabels:
                # not positve
                continue
            label = 1
            neg = 0 if not _neg else neg
            hist = 0 if not _hist else hist
            ass = _ass if  _ass else ass
            ann = 0 if not _ann else ann
            
        if label == 0:
            label, neg, hist, ass, ann = 0,0,0,'',0
        results.append((user, documentName, label, neg, hist, ass, ann))
    results = pd.DataFrame(results,
                           columns=['user', 'document', 'label', 
                                    'negationFlag', 'historicalFlag', 
                                    'assertionType', 'annotationFlag'])
    return results
    
    
    
    