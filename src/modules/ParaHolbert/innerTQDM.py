

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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:


    outputLocation = Path(moduleConfig['output']['location']) / runID
    longWork = [50,70,10,30,80,90,70,50,70]
    
    
    
    pool = mp.Pool(4) # use 4 processes

    funclist = []
    for df in longWork:
            # process each data frame
            f = pool.apply_async(doingWork,[df])
            funclist.append(f)

    result = 0
    # bar = tqdm(funclist, total=len(longWork))
    for f in funclist:
        output = f.get()
        print(output)
        # bar.update(output)
            # result += f.get(timeout=10) # timeout in 10 seconds
    # outputLocation.mkdir(parents=True, exist_ok=True)
    # keep a copy of the config file for regeneration
    # writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)
    pass

if __name__ == '__main__':
    main('resultsDict')
    