import numpy as np
from sklearn.linear_model import LogisticRegression

def log_progress(sequence, every=10):
    from ipywidgets import IntProgress
    from IPython.display import display

    progress = IntProgress(min=0, max=len(sequence), value=0)
    display(progress)
    
    for index, record in enumerate(sequence):
        if index % every == 0:
            progress.value = index
        yield record

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
        
    def min_untaken_len(self):
        return min([len(X) for X in  self.untaken])
    
    def untaken_X(self, one_class):
        return self.untaken[self.classes_dict[one_class]]
        
    def untaken_y(self, one_class):
        return np.zeros(len(self.untaken[self.classes_dict[one_class]])) + one_class
        
    def take(self, one_class, indexes):
        new_untaken_ind = np.array(np.ones(len(self.untaken_X(one_class))),dtype=bool)
        new_untaken_ind[indexes] = False
        self.taken_X = np.vstack((self.taken_X,\
                                 self.untaken[self.classes_dict[one_class]][indexes]))
        self.taken_y = np.hstack((self.taken_y, np.zeros(1 if type(indexes) == int else len(indexes))+ one_class))
        self.untaken[self.classes_dict[one_class]] = self.untaken[self.classes_dict[one_class]][new_untaken_ind]
    
class ALImbalance(object):
    #TODO подумать о стратегиях и видах неуверенности
    def __init__(self, model = LogisticRegression, confidence = 'entropy', strategy = 'max', step=10):
        self.model = model()
        self.confidence = confidence
        self.strategy = strategy
        self.step = step
    
    def fit(self, X, y):
        self.data_set = TwoDataSets(X, y)
    
    def sample(self): 
        self.data_set.random_start()
        #while self.data_set.takenable(): #нужен обобщеный предикат
        for i in log_progress(range(0, self.data_set.min_untaken_len(), self.step)):
            self.sample_step()
        if self.data_set.min_untaken_len() > 0:
            self.sample_step(self.data_set.min_untaken_len())
        
        return (self.data_set.taken_X,\
                self.data_set.taken_y)
        
    def sample_step(self, step = None):
        step = self.step if step == None else step
        self.model.fit(self.data_set.taken_X, self.data_set.taken_y)
        
        for one_class in self.data_set.classes:
            probabilities = self.model\
                .predict_proba(self.data_set.untaken_X(one_class))
            if self.confidence == 'entropy':
                model_confidence = -np.sum(probabilities * np.log(probabilities),axis=1)
            elif self.confidence == 'least_confident': 
                model_confidence = np.min(probabilities , axis=1)
            elif self.confidence == 'margin': 
                model_confidence = np.argsort(probabilities,axis=1)[:,-1] -\
                                   np.argsort(probabilities ,axis=1)[:,-2]
            else:
                print("confidence mast be entropy, margin or someF*")
            
            if self.strategy == 'max':
                chosen = model_confidence.argsort()[:step]    #TODO top_k
            else:
                print("strategy mast be max or random")
            self.data_set.take(one_class,chosen)
        
    
    def fit_sample(self, X, y):
        self.fit(X, y)
        return self.sample()
    
def resample(X,y, model = LogisticRegression, confidence = 'entropy', strategy = 'max', step=10):
    return ALImblance(model, confidence, strategy , step).fit_sample(x_train,y_train)
