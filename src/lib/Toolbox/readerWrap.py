from logs import logDecorator as lD
from pathlib import Path
import json
import yaml

script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

@lD.log(logBase + '.readYASON')
def readYASON(logger, directory:str)->dict:
    """YASON stands for YAML or JSON

    Parameters
    ----------
    directory : str
        Path to the YAML or JSON file

    Returns
    -------
    dict
        dictionary representation of the configuration file
    """
    
    if directory.endswith('.json'):
        result = json.load(open(directory))
        return result
    else:
        with open(directory, 'r') as stream:
            result = yaml.safe_load(stream)
        return result
        
