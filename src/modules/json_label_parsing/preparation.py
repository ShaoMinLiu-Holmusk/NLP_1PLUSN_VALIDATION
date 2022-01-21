from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from pathlib import Path


script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

from pathlib import PosixPath
from typing import List, Union, Set

from warnings import warn


@lD.log(logBase + '.filterUsers')
def filterUsers(logger, 
                givenUsers:List[str],
                acceptedUsers:Union[str, Set[str]]) -> List[PosixPath]:
    """Given a set of folders from different labeller, process only 
    the acceptedUsers, unless specified as 'all'

    Parameters
    ----------
    givenUsers : List[pathlib.PosixPath]
        a list of str path to the json files
    acceptedUsers : Union[str, Set[str]]
        * a string "all" means no filter is applied
        * a set of white list users

    Returns
    -------
    Iterator[pathlib.PosixPath]
        filtered iterator of the json file path
    """
    logger.info('filterTriggered')
    
    if acceptedUsers == 'all':
        return (Path(each) for each in givenUsers)
    
    def userGenerator():
        
        for eachUser in givenUsers:
            eachUser = Path(eachUser)
            if eachUser.parent.stem in acceptedUsers:
                yield eachUser
                
    result = userGenerator()
    return result

@lD.log(logBase + '.outputPathName')
def outputPathName(logger, 
                   outputLocation:str,
                   outputName:str,
                   allowedUsers:Union[str, list],
                   dataSource:str) -> PosixPath:
                   
    # the following deals with naming and writing of files
    writeFolderPath = Path(outputLocation)
    writeFolderPath.mkdir(exist_ok=True, parents=True)
    writeName = outputName
    suffix = '.json' # correct suffix for the file
    if writeName == 'default':
        if allowedUsers == 'all':
            nameSuffix = '_all'
        elif isinstance(allowedUsers, list):
            nameSuffix = f'_{len(allowedUsers)}users'
        writeName = Path(dataSource).stem + nameSuffix + suffix
    else:
        if not writeName.endswith(suffix):
            warn(f'Filename has to suffix with `{suffix}`')
            writeName = Path(writeName).stem + suffix
            warn(f'Writing to `{writeFolderPath/writeName}`')
            
    writeDirectory = writeFolderPath/writeName
    # writeDirectory = nextName(writeDirectory)
    
    return writeDirectory