
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

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    
    for 

    
    outputLocation.mkdir(parents=True, exist_ok=True)
    # keep a copy of the config file for regeneration
    writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)
    pass

if __name__ == '__main__':
    main('resultsDict')
    