
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

from glob import glob
from tqdm import tqdm 

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    '''
    Basically run through the bokeh label fiels
    concatenate the sentences together to get back
    the txt files
    '''
    
    
    outputLocation = Path(moduleConfig['output']['location'])/runID
    outputLocation.mkdir(parents=True, exist_ok=True)
    
    allLabels = list(glob(moduleConfig['dataSource']+'/**/*.json'))
    for eachLabelDoc in tqdm(allLabels, total=len(allLabels)):
        eachLabelDoc = readYASON(str(eachLabelDoc))
        fileName = eachLabelDoc['filename']
        
        eachLabelDoc = eachLabelDoc['labels']
        textPath = Path(outputLocation/fileName.replace('.json', '.txt'))
        if textPath.exists():
            logger.warning(f'already exist: {textPath}')
            continue
        with open(str(textPath), 'a') as file:
            for row in eachLabelDoc:
                file.write(row[-1])
    
    # keep a copy of the config file for regeneration
    # writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)

if __name__ == '__main__':
    main('resultsDict')
    