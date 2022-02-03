
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

from pydoc import locate
from pandas import DataFrame, Series
from itertools import repeat

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    
    # get a list all document names
    datafromMSEParse = moduleConfig['dataSource']
    # the list of txt that needs translation
    datafromMSEParse = list(Path(datafromMSEParse).iterdir())
    
    outputLocation = Path(moduleConfig['output']['location']) / runID
    outputLocation.mkdir(parents=True, exist_ok=True)
    
    for eachFeature in moduleConfig['features']:
        if not eachFeature['participating']:
            continue
        Feature = locate(eachFeature['subModule'])
        if Feature == None:
            logger.warning(f"feature {eachFeature['name']} is not loaded")
            continue
        else:
            logger.info(f"feature {eachFeature['name']} is loaded")
            thisFeature = Feature(name = eachFeature['name'],
                                  subModule = eachFeature['subModule'],
                                  config = eachFeature['subModuleConfig'],
                                  txtLocation = None # no use
                                  )
        X_train = thisFeature.extract(datafromMSEParse)
        columns = zip(repeat('feature_'), range(X_train.shape[1]))
        columns = list(name+str(num) for name, num in columns)
        X_train = DataFrame(X_train,
                            columns=columns)
        X_train['filename'] = Series(name.name for name in datafromMSEParse)
        X_train.to_csv(str(outputLocation/(eachFeature['name']+'.csv') ))

    # keep a copy of the config file for regeneration
    writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)
    pass

if __name__ == '__main__':
    main('resultsDict')
    