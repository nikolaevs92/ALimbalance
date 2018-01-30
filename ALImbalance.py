import numpy as np
from sklearn.linear_model import LogisticRegression

class TwoDataSets(object):
    def __init__(self, X, y):
        self.classes = np.unique(y)
        self.classes_dict = {self.classes[i] : i for i in range(len(self.classes))}
        self.untaken = [X[np.array(y == one_class)] for one_class in self.classes_dict.keys()]
        self.taken_X = np.zeros(shape = (0, X.shape[1]))
        self.taken_y = np.zeros(shape = (0))
    
    def random_start(self):
        for one_class in self.classes_dict.keys():
            self.take(one_class,\
                      np.random.randint(len(self.untaken[self.classes_dict[one_class]])))
    
    def takenable(self, one_class = None):
        if one_class == None:
            for one_class_X in self.untaken:
                if len(one_class_X) == 0:
                    return False
            return True
        return False
        
    def untaken_X(self, one_class):
        return self.untaken[self.classes_dict[one_class]]
        
    def untaken_y(self, one_class):
        return np.zeros(len(self.untaken[self.classes_dict[one_class]])) + one_class
        
    def take(self, one_class, index):
        self.taken_X = np.vstack((self.taken_X,\
                                 self.untaken[self.classes_dict[one_class]][index]))
        self.taken_y = np.hstack((self.taken_y, one_class))
        self.untaken[self.classes_dict[one_class]] = np.vstack((\
            self.untaken[self.classes_dict[one_class]][:index],\
            self.untaken[self.classes_dict[one_class]][index+1:]))
        #TODO исправить этот ужас
    
class ALImblance(object):
    #TODO подумать о стратегиях и видах неуверенности
    def __init__(self, model = LogisticRegression, confidence = 'entropy', strategy = 'max'):
        self.model = model()
        self.confidence = confidence
        self.strategy = strategy
    
    def fit(self, X, y):
        self.data_set = TwoDataSets(X, y)
    
    def sample(self): 
        self.data_set.random_start()
        while self.data_set.takenable(): #нужен обобщеный предикат
        #for i in range(200):
            self.sample_step()
        return (self.data_set.taken_X,\
                self.data_set.taken_y)
        
    def sample_step(self):
        self.model.fit(self.data_set.taken_X, self.data_set.taken_y)
        
        for one_class in self.data_set.classes:
            probabilities = self.model\
                .predict_proba(self.data_set.untaken_X(one_class))
            if self.confidence == 'entropy':
                model_confidence = np.sum((probabilities * np.log(probabilities)),axis=1)
            else:
                print("confidence mast be entropy")
            
            if self.strategy == 'max':
                self.data_set.take(one_class,\
                      np.random.randint(len(self.data_set.untaken[self.data_set.classes_dict[one_class]])))
                    #one_class, model_confidence.argmin())
            else:
                print("strategy mast be max or random")
        
    
    def fit_sample(self, X, y):
        self.fit(X, y)
        return self.sample()
    
def resample(X,y):
    return ALImblance().fit_sample(x_train,y_train)
