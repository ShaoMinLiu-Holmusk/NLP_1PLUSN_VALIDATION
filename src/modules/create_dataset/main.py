from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
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
tag = moduleConfig['output']['name']
runID = '_'.join((runID,tag)) if tag else runID

from modules.create_dataset.task_worker import executeTask
from tqdm import tqdm

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    
    labels = readYASON(moduleConfig['dataSource'])
    convertTask = moduleConfig['convertTask']
    outputLocation = Path(moduleConfig['output']['location']) / runID
    outputLocation.mkdir(parents=True, exist_ok=True)
    
    for task in tqdm(convertTask, disable=len(convertTask)==1):
        resultDataFrame = executeTask(task, labels)
        saveDir = outputLocation / (task['customLabel'] + '.csv')
        resultDataFrame.to_csv(saveDir)

if __name__ == '__main__':
    main('resultsDict')