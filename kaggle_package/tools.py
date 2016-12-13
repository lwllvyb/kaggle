#-*- coding:utf-8 -*-


class Ensemble(object):
    def __init__(self, n_folds, base_models, stacker):
        self.n_folds = n_folds
        self.base_models = base_models
        self.stacker = stacker
        
    def predict(self, X):
        test = np.zeros((X.shape[0], len(self.base_models)))
        
        for i , clf in enumerate(self.base_models):
            test[:, i] = clf.predict(X)
            
        return self.stacker.predict(test)
    
    def fit_predict(self, train_X, train_y, test_X):
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        
        folds = list(KFold(len(train_y), n_folds = self.n_folds, shuffle=True, random_state=2016))
        
        S_train = np.zeros((train_X.shape[0], len(self.base_models)))
        S_test = np.zeros((test_X.shape[0], len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((test_X.shape[0], len(folds)))
            
            for j, (train_idx, cv_idx) in enumerate(folds):
                X_train = train_X[train_idx]
                y_train = train_y[train_idx]
                X_cv = train_X[cv_idx]
                
                clf.fit(X_train, y_train)
                
                y_pred = clf.predict(X_cv)
                S_train[cv_idx, i] = y_pred
                
                S_test_i[:, j] = clf.predict(test_X)
            S_test[:, i] = S_test_i.mean(1)
        
        self.stacker.fit(S_train, train_y)
        
        y_pred = self.stacker.predict(S_test)
        
        return y_pred

# feature select
def FeatureSelect(X, y, feature_name):
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True)
