from modules.feature_competition.models.Interface import ModelInterface
from sklearn.neighbors import KNeighborsClassifier
from numpy import ndarray, array, squeeze

class Model(ModelInterface):
    
    def __init__(self, config: dict) -> None:
        """[summary]

        Parameters
        ----------
        config : dict
            Configuration for the KNN, the following default template:
            {
                n_neighbors     : 3,
                leaf_size:      : 30
            }
        """
        
        super().__init__(config)
        self.model = KNeighborsClassifier(**config)
        
    def fit(self, X: ndarray, y:array)-> None:
        self.model.fit(X, y)
        
    def predict_proba(self, X:ndarray)-> array:
        trueIndex = list(self.model.classes_).index(True)
        return squeeze(self.model.predict_proba(X)[:,trueIndex])