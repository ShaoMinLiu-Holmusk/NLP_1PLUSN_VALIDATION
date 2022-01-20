import imp
import pathlib
import sys
sys.path.insert(0,'../src/')

from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from pathlib import Path

script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

moduleConfig = config['configVersions'][moduleName]
moduleConfig = readYASON(moduleConfig)

# the above should be the same for most modules, do not need to change
# the following is where your own code starts
from typing import List
from tqdm import tqdm
from lib.Toolbox.readerWrap import readYASON
from lib.Toolbox.writerWrap import writeJSON
from lib.Toolbox.pathWrap import nextName

@lD.log(logBase + '.parseMSEJSON')
def parseMSEJSON(logger, JSONfile:pathlib.PosixPath)->dict:
    """Each label for the sentence is assumed to have the following 
    structure:
    [label], negation, historical, assertion, annotationSubtype
    For now, its likely that assertion and annotationSubtype does not 
    contain useful information, if we were to extract 

    Parameters
    ----------
    JSONfile : pathlib.PosixPath
        [description]

    Returns
    -------
    dict
        [description]
    """
    
    fileName = JSONfile.name
    JSONfile = readYASON(JSONfile)
    user = JSONfile['user']
    labels = set()
    
    for eachLabel in JSONfile['labels']:
        if len(eachLabel[1][0]) != 1:
            continue
        try:
            (label,), negation, historical, assertion, annotationSubtype = eachLabel[1]
        except ValueError:
            raise ValueError
            
        labels.add((label, negation, historical, assertion, annotationSubtype))
    labels = list(labels)
    labels.sort()
    result = {
        'user': user,
        'document': fileName,
        'labels':labels
    }
    return result

@lD.log(logBase + '.parseAllJson')
def parseAllJson(logger, givenUsers:List[pathlib.PosixPath])->None:
    """Each json file represent a document, within which, each
    element in the list will be a sentence of the document.
    for each json, convert it to a row in csv, record the username, 
    fileName, and the aggregated labels of the csv
    
    Parameters
    ----------
    logger : [type]
        [description]
    givenUsers : List[pathlib.PosixPath]
        [description]
    """
    
    results = []
    for eachUser in tqdm(givenUsers, desc="Documents"):
        eachUser = parseMSEJSON(eachUser)
        results.append(eachUser)
    return results 
    
    
    