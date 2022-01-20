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
import glob
from pathlib import PosixPath
from typing import Union, List, Set

from lib.Toolbox.writerWrap import writeJSON
from lib.Toolbox.pathWrap import nextName

from modules.json_label_parsing.parsing import parseAllJson
from modules.json_label_parsing.preparation import filterUsers
from modules.json_label_parsing.preparation import outputPathName
from modules.json_label_parsing.report import report
            
@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    
    # generating a name for the output file
    # depends on if the default config is set to True
    writeDirectory = outputPathName(outputLocation = moduleConfig['output']['location'],
                                    outputName=moduleConfig['output']['name'],
                                    allowedUsers = moduleConfig['allowedUsers'],
                                    dataSource = moduleConfig['dataSource'])
    
    if writeDirectory.exists() and moduleConfig['report']['rerunParsing']:
        givenUsers = glob.glob(moduleConfig['dataSource']+'/**/*.json')
        givenUsers = filterUsers(givenUsers, moduleConfig['allowedUsers'])
        result = parseAllJson(givenUsers)
        if not moduleConfig['output']['overwrite']:
            writeDirectory = nextName(writeDirectory)
        writeJSON(result, writeDirectory)
        del result
    
    if moduleConfig['report']['activate']:
        result = report(writeDirectory)
        writeDirectory = writeDirectory.parent /\
            (writeDirectory.stem + '_report' + writeDirectory.suffix)
        writeJSON(result, writeDirectory)

if __name__ == '__main__':
    main('resultsDict')