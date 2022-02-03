
from modules.feature_competition.models.Interface import ModelInterface
from numpy import ndarray, array
from lib.Toolbox.readerWrap import readYASON

class Model(ModelInterface):
    
    def __init__(self, config: dict,
                 problemName:str) -> None:
        """[summary]
        

        Parameters
        ----------
        config : dict
        
        
        problemName : str
            must be one of the options in config['mselabelKeys']
            this will point to the index of the matrix that 
            will know indicate the position to look for
        """
        super().__init__(config)
        self.labelIndex = readYASON(config['mselabelKeys'])[problemName]
        
    def fit(self, X: ndarray, y:array)-> None:
        # self.model.fit(X, y)
        pass
        
    def predict_proba(self, X:ndarray)-> array:
        return X[:, self.labelIndex].any(axis=1).astype(int)