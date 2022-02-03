

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

from modules.ParaHolbert.somework import doingWork, processDocuments
import multiprocessing as mp
from tqdm import tqdm
from time import sleep
from lib.Toolbox.parallelWrap import ParallelWorker, TaskLogic
from lib.Toolbox.loggerWrap import SimplyLog, CSVwriter

class ReadSentence(TaskLogic):
    
    def workSingleTask(self, task):
        return task
    

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:


    outputLocation = Path(moduleConfig['output']['location']) / runID
    
    sentences = open('../data/intermediate/preprocessed/allBatchesParsed/2022-01-25_16-42-07/aadeladan-522741-8242012-0-1427-161410.txt',
                     'r').readlines()
    
    for each in sentences:
        logger.debug(each)
    # outputLocation.mkdir(parents=True, exist_ok=True)
    # keep a copy of the config file for regeneration
    # writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)
    pass

if __name__ == '__main__':
    main('resultsDict')
    