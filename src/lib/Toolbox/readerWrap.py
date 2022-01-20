from logs import logDecorator as lD
from pathlib import Path
import jsonref
import yaml

script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = jsonref.load(open('../config/config.json'))
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
    
    with open(directory, 'r') as stream:
        try:
            result = yaml.safe_load(stream)
            return result
        except yaml.YAMLError as error:
            pass
        
        try:
            result = jsonref.load(open(directory))
            return result
        except Exception as anotherError:
            print(error)
            print(anotherError)
        
