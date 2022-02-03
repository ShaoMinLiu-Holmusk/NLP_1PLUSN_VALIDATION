from modules.feature_competition.models.Interface import ModelInterface
from sklearn.linear_model import LogisticRegression
from numpy import ndarray, array, squeeze

class Model(ModelInterface):
    
    def __init__(self, config: dict) -> None:
        """[summary]

        Parameters
        ----------
        config : dict
            Configuration for the logistic regression, 
            the following default template:
        """
        
        super().__init__(config)
        self.model = LogisticRegression(**config)
        
    def fit(self, X: ndarray, y:array)-> None:
        self.model.fit(X, y)
        
    def predict_proba(self, X:ndarray)-> array:
        trueIndex = list(self.model.classes_).index(True)
        return squeeze(self.model.predict_proba(X)[:,trueIndex])