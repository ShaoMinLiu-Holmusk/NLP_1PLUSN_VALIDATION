from email.policy import default
from pathlib import Path
import json
from tkinter import W
from argparse import ArgumentParser
from warnings import warn

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-n','--moduleName',required=True,type=str)
    args.add_argument('-o','--owner',type=str, default='')
    args.add_argument('-d','--description',type=str, default='')
    args = args.parse_args()
    moduleName = args.moduleName
    
    moduleFolder = Path(f'./modules/{moduleName}')
    while moduleFolder.exists():
        warning = f'Module `{moduleName}` already exist!'
        warn(warning)
        moduleName = input(f'Please rename: ')
        moduleFolder = Path(f'./modules/{moduleName}')
        
    moduleFolder.mkdir(parents=True)
    (moduleFolder/ '__init__.py').mkdir()
    
    mainTemplate = '''
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

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    pass

if __name__ == '__main__':
    main('resultsDict')
    '''
    with open(moduleFolder / 'main.py', 'w') as mainFile:
        mainFile.write(mainTemplate)
        
    # create config files
    configTemplate = f'''
# trying out using YAML as config file
# YAML supports commenting unlike JSON
# it is more user-friendly for documentation purposes
# other than this, yamlFile and JSON can both be read as
# a dictionary format, thus little changes are needed 
# to be made to the actual code

dataSource    : ../data/

output:
  location    : ../data/intermediate/{moduleName}
  name        : default # easier to just keep to deafult
  overwrite   : false # if overwrite is false, will write with incremental name
    '''
    configFolder = Path(f'../config/modules/{moduleName}')
    configFolder.mkdir(exist_ok=True, parents=True)
    with open(configFolder / 'sampleConfig(dev).yaml', 'w') as mainFile:
        mainFile.write(configTemplate)
        
    config = json.load(open('../config/config.json'))
    config['configVersions'][moduleName] = str(configFolder / 'sampleConfig(dev).yaml')
    with open('../config/config.json', 'w') as json_file:
        json.dump(config, json_file, indent=4)
        
    config = json.load(open('../config/modules.json'))
    config.append(
        {
        "moduleName" : moduleName,
        "path"       : f"modules/{moduleName}/main.py",
        "execute"    : False,
        "description": args.description,
        "owner"      : args.owner
    }
    )
    with open('../config/modules.json', 'w') as json_file:
        json.dump(config, json_file, indent=4)
    
        
    
    