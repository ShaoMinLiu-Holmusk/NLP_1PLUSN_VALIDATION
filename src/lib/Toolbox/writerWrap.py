import yaml
import json

def writeJSON(object, path, sort_keys=False):
    """[summary]

    Parameters
    ----------
    object : [type]
        [description]
    path : [type]
        [description]
    sort_keys : bool, optional
        by default False
        leave it to false if its not important to sort it, 
        otherwise it might take too much time to sort the values
    """
    with open(path, 'w') as json_file:
        json.dump(object, json_file, indent=4, sort_keys=sort_keys)
        
def writeYAML(object, path):
    with open(path, 'w') as f:
        yaml.safe_dump(object, f, sort_keys=False)
    
    