
from pathlib import Path


def nextName(name:str)->str:
    originalPath = Path(name)
    if not originalPath.exists():
        return name
    
    index = 1
    while Path(f"{originalPath.parent/originalPath.stem}_{index}{originalPath.suffix}").exists():
        index += 1
    return f"{originalPath.parent/originalPath.stem}_{index}{originalPath.suffix}"