import sys
sys.path.insert(0,'../src/')

from itertools import count
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

import pandas as pd
from lib.Toolbox.writerWrap import writeYASON

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    
    sourceFolder = list(Path(moduleConfig['dataSource']).iterdir())
    outputLocation = Path(moduleConfig['output']['location']) / runID
    outputLocation.mkdir(parents=True, exist_ok=True)
    
    counterMaster = dict()
    for eachFile in sourceFolder:
        file = pd.read_csv(str(eachFile))
        counter = file.label.value_counts()
        counter = dict(zip((str(i) for i in counter.index), 
                           (str(i) for i in counter.values)))
        counterMaster[eachFile.name] = counter
        print(eachFile.name, counter)
        
    writeYASON(counterMaster, outputLocation/'labelCounts.yaml')
    # keep a copy of the config file for regeneration
    # writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)
    pass

if __name__ == '__main__':
    main('resultsDict')
    