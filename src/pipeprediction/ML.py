from utils import UtilMethods as Utils
from pipeprediction import Extractor, DimensionHandler, Loader
import re, os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier as GP
from sklearn.ensemble import RandomForestClassifier as RF, IsolationForest as IF
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.covariance import EllipticEnvelope as RC  #RC = Robust Covariance
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as pfrs, classification_report
from sklearn.externals import joblib
from sklearn.utils.validation import check_is_fitted
from pyspark import SparkContext


###############
# Manages machine learning pipeline:
# train, validation, test, model loading
# and saving, classifiers, performance output
###############


class ML:

    def __init__(self):
        # read application configuration props
        self.config = Utils.loadConfig()
        self.path = self.config.get('prediction', 'source.path')
        self.path = Utils.normalizePath(self.path)
        self.trainPath = self.path + 'train/'
        self.validPath = self.path + 'validation/'
        self.gridCVPath = self.path + 'train_validation/'
        self.testPath = self.path + 'test/'
        self.outputPath = self.path + 'metrics/cv_gridsearchparams/'
        self.task = self.config.get('prediction', 'task')
        self.posPerc = int(self.config.get('prediction', 'pos.perc'))
        self.classif = self.config.get('prediction', 'classifier')
        os.makedirs(os.path.dirname(self.outputPath), exist_ok=True)
        self.extractor = Extractor.Extractor(self.config, self.outputPath)
        self.loader = Loader.Loader(self.config, self.outputPath)
        self.dimHandler = DimensionHandler.DimensionHandler(self.config, self.outputPath)
        self.outFile = ''
        self.useEmbeddings = self.config.getboolean('prediction', 'use.embeddings')
        self.cv = self.config.getboolean('prediction', 'use.crossvalid')
        if('cross' in self.task):
            self.cv = True
        if (not 'none' in self.dimHandler.name.lower()):
            self.outFile = self.dimHandler.getOutFile(self.classif)
            self.outFile = self.outFile + '_embeddings' if self.useEmbeddings else self.outFile
        else:
            self.outFile = self.outputPath + self.classif + '_' + self.extractor.featType
            if('kmers' in self.extractor.featType):
                kmerfeats = 'kmers' + str(self.extractor.size) + '_minOcc' + str(self.extractor.minOcc)
                self.outFile = self.outFile.replace('kmers', kmerfeats)
                #self.outFile +=  str(self.extractor.size) + '_minOcc' + str(self.extractor.minOcc)
        if('cross' in self.task or 'grid' in self.task or self.cv):
            self.extractor.featFile = self.extractor.featFile.replace('.feat', '.complete.feat') if 'grid' in self.task else self.extractor.featFile
            if('cross' in self.task or self.cv):
                self.outFile += '_cv05'

        self.modelFile = self.outFile + '.model.pkl'
        self.classifier = self.setUpClassifier()


    def main(self):
        sparkContext = SparkContext(conf=self.extractor.initSpark())

        # performs gridsearch or cross validation, extract features from entire train set
        if ('cross' in self.task or 'grid' in self.task):
            self.extractor.extractFeatures(self.gridCVPath, sparkContext, featPerInst=False)
            IDs, x_occ, y_labels, parentDir = self.extractor.countOccurrence(self.gridCVPath, sparkContext)

            if ('cross' in self.task):
                self.runCrossValid(x_occ, y_labels, IDs)

            elif ('grid' in self.task):
                self.runGridSearch(x_occ, y_labels)

        # performs training, extracts features from split train / valid
        elif ('train' in self.task and not os.path.isfile(self.modelFile)):
            if (not os.path.isfile(self.extractor.featFile)):
                self.extractor.extractFeatures(self.trainPath, sparkContext, featPerInst=False)
            print('Training...')
            IDs, x_occ, y_labels, parentDir = self.extractor.countOccurrence(self.trainPath, sparkContext)

            if(not 'none' in self.dimHandler.name.lower()):
                x_occ = self.dimHandler.trainMethod(x_occ, y_labels)

            self.classifier.fit(x_occ, y_labels)

            joblib.dump(self.classifier, self.modelFile)
            print('Model saved. \nPredicting...')

        else:
            try:
                self.classifier = joblib.load(self.modelFile)
                print('Model loaded. \nPredicting...')
            except FileNotFoundError:
                print('Model', self.modelFile, 'does not exist. Generate it by training (task = train).')


        if('cross' not in self.task and 'randomforest' in self.classif.lower() and not os.path.isfile(self.extractor.featFile + 'importance')):
            self.getRFImportance()

        # performs validation, loads features from split train /valid
        if('validation' in self.task):
            IDs, x_occ_val, y_labels_val, parentDir = self.extractor.countOccurrence(self.validPath, sparkContext)
            output, IDoutput = self.getPredictions(IDs, x_occ_val, y_labels_val)
            Utils.writeFile(self.outFile + '.valid', output)
            Utils.writeFile(self.outFile + '.IDs.valid', IDoutput)

        # performs validation, loads features from split train /valid
        if ('test' in self.task):
            IDs, x_occ_test, y_labels_test, parentDir = self.extractor.countOccurrence(self.testPath, sparkContext)

            if('none' not in self.dimHandler.name.lower()):
                x_occ_test = self.dimHandler.testMethod(x_occ_test)

            output, IDoutput = self.getPredictions(IDs, x_occ_test, y_labels_test)

            Utils.writeFile(self.outFile + '_' + parentDir + '.test', output)
            Utils.writeFile(self.outFile + '_' + parentDir + '.IDs.test', IDoutput)

        print('Done!')



    def getPredictions(self, IDs, occ, labels):
        if ('outlier' in self.classif.lower()):
            predLabels = self.classifier.fit_predict(occ)
        else:
            predLabels = self.classifier.predict(occ)

        if('one' or 'isolation' in self.classif.lower()):
            score = 'one class case'
        else:
            score = self.classifier.score(occ, labels)
        IDoutput = ''

        confidence = [None] * len(IDs)
        if ('svc' in self.classif.lower()):
            confidence = self.classifier.decision_function(occ)

        for i in range(0,len(predLabels)):
            IDoutput += str(IDs[i]) + '\t' + str(predLabels[i]) + '\n'

        output = self.getMetrics(score, predLabels, labels)

        return output, IDoutput


    ##############################
    # output RF feature importance
    ##############################
    def getRFImportance(self):
        pd.options.display.float_format = '{:,.8f}'.format
        importance = self.classifier.feature_importances_
        features = self.extractor.loadFeatures()

        feature_importances = pd.DataFrame(importance,
                                           index=features,
                                           columns=['importance']).sort_values('importance', ascending=False)

        Utils.writeFile(self.extractor.featFile + 'importance', feature_importances.to_string())


    def getMaxLen(self):
        if('mlp' in self.classif.lower()):
            return len(self.classifier.coefs_[0])
        elif('logit' in self.classif.lower() or 'linearsvc' in self.classif.lower()):
            return self.classifier.coef_.size
        elif('randomforest' in self.classif.lower()):
            return self.classifier.n_features_
        elif('nusvc' in self.classif.lower() or self.classif.lower() in 'svc'):
            return self.classifier.shape_fit_[1]


    def setUpClassifier(self):
        if('linearsvc' in self.classif.lower()):
            if (not 'grid' in self.task):
                return svm.LinearSVC(C=0.01, loss='squared_hinge', penalty='l2')
            else:
                return svm.LinearSVC()

        # in case positive perc is low,
        # NuSVC nu param has to be adjusted
        elif('nusvc' in self.classif.lower()):
            thisNu = self.posPerc / 100
            if(not 'grid' in self.task):
                return svm.NuSVC(nu=thisNu, coef0=0.01, gamma=0.01, kernel='sigmoid')
            else:
                return svm.NuSVC(nu=thisNu)

        elif(self.classif.lower() in 'svc'):
            # pass 'probability=True' if confidence values must be computed
            if (not 'grid' in self.task):
                return svm.SVC(C=100, gamma=0.001, kernel='rbf')
            else:
                return svm.SVC()

        elif ('svr' in self.classif.lower()):
            return svm.SVR()

        elif ('mlp' in self.classif.lower()):
            if (not 'grid' in self.task):
                return MLP(activation='relu', batch_size=256, hidden_layer_sizes=256, learning_rate='adaptive', solver='adam')
            else:
                return MLP()

        elif ('gaussian' in self.classif.lower()):
            return GP()

        elif('randomforest' in self.classif.lower()):
            if (not 'grid' in self.task):
                return RF(bootstrap=False, criterion='entropy', max_features='log2', n_estimators=1000)
            else:
                return RF()

        elif('logit' in self.classif.lower()):
            if (not 'grid' in self.task):
                return Logit(penalty='l1', C=10, solver='saga')
            else:
                return Logit()

        elif('one' in self.classif.lower()):
            return OCSVM(kernel='rbf')

        elif('isolation' in self.classif.lower()):
            return IF()

        elif('covariance' in self.classif.lower()):
            return RC()

        elif('outlier' in self.classif.lower()):
            return LOF()


    def getMetrics(self, score, pLabels, rLabels):
        output = 'Mean accuracy:\t' + str(score) + '\n'
        prfscore = pfrs(rLabels, pLabels)

        precision = np.array2string(prfscore[0]).replace('[', '').replace(']', '')
        recall = np.array2string(prfscore[1]).replace('[', '').replace(']', '')
        fbeta = np.array2string(prfscore[2]).replace('[', '').replace(']', '')

        precision = re.sub('\s+', '\t', precision)
        recall = re.sub('\s+', '\t', recall)
        fbeta = re.sub('\s+', '\t', fbeta)

        output += '\tneg\tpos\n'
        output += 'P:' + precision + '\n'
        output += 'R:' + recall + '\n'
        output += 'F1:' + fbeta + '\n'

        cf = np.array2string(confusion_matrix(rLabels, pLabels))
        cf = 'neg\t' + cf.replace('[[', '').replace(']]', '')
        cf = cf.replace(']\n [', '\npos\t')

        output += 'Confusion matrix: ' + '\n'
        output += '\tpNeg\tpPos' + '\n'
        output += cf

        return output


    def getConfidencePerID(self, IDs, pLabels, rLabels, confidence):
        print('ID\tPredicted\tTrue class\tConfidence')
        for i in range(len(IDs)):
            print(str(IDs[i]) + '\t'
                  + str(pLabels[i]) + '\t'
                  + str(rLabels[i]) + '\t'
                  + str(confidence[i]))


    # perform cross validation on dataset
    def runCrossValid(self, x_occ, y_labels, IDs):
        seed = 5
        np.random.seed(seed)
        i = 1

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for train, valid in kfold.split(x_occ, y_labels):
            # get position of features inexistent on train, remove such feats from valid

            # gives which indexes are greater than 0
            filter = np.where(np.sum(x_occ[train], axis=0) > 0)[0]

            # takes only column indices from *filter*
            x_occT = np.take(x_occ[train], filter, axis=1)
            x_occV = np.take(x_occ[valid], filter, axis=1)

            self.classifier.fit(x_occT, y_labels[train])

            output, IDoutput = self.getPredictions(IDs[valid], x_occV, y_labels[valid])
            Utils.writeFile(self.outFile + 'f'+ str(i) + '.valid', output)
            Utils.writeFile(self.outFile + 'f'+ str(i) + '.IDs.valid', IDoutput)
            i += 1

        self.classifier = self.setUpClassifier()
        self.classifier.fit(x_occ, y_labels)

        joblib.dump(self.classifier, self.modelFile)
        print('Model saved.')


    # performs grid search on training and validation data to
    def runGridSearch(self, x_occ, y_labels):
        output = 'Running grid search for ' + self.classif + ' in ' + str(len(x_occ)) + ' instances ...\n'
        print('Running grid search for', self.classif, 'in', str(len(x_occ)), 'instances ...\n')
        scores = ['f1', 'precision', 'recall']

        for score in scores:
            output += 'Grid search for score: ---> ' + score + ' <---\n'

            classif = GridSearchCV(estimator=self.setUpClassifier(), param_grid=self.getGridParams(), scoring=score,
                                   cv=5, n_jobs=60)
            classif.fit(x_occ, y_labels)
            output += 'Best parameters in train set:\n'
            output += str(classif.best_params_) + '\n'
            output += 'Grid scores in train set:\n'
            means = classif.cv_results_['mean_test_score']
            stds = classif.cv_results_['std_test_score']

            for mean, std, params in zip(means, stds, classif.cv_results_['params']):
                params = str(params).replace('{', '').replace('}', '')
                output += ("%0.3f (+/-%0.03f) |\t params %r" % (mean, std * 2, params)) + '\n'

            output += "\n--------------------------------------------------\n"
            print('Done with', score, '.')

        Utils.writeFile(self.outputPath + self.classif + '.gridSearch', output)
        print(output)

    def getGridParams(self):
        Cvalues = [0.01,0.1, 1,10,100]
        gammaValues = [0.01,0.001,0.0001]
        lossValues = ['hinge', 'squared_hinge']
        estimatorValues = [10,100,1000]
        maxFeatValues = ['', 'auto', 'sqrt', 'log2']
        penalties = ['l1', 'l2']
        hiddenLayers = [(64,), (128,), (256,), (512,), (1024,)]
        activationValues = ['identity', 'logistic', 'tanh', 'relu']
        learningRate = ['constant', 'invscaling', 'adaptive']
        batchSizes = [32,64,128,256]
        nu = self.posPerc / 100

        if ('linearsvc' in self.classif.lower()):
            return [{'penalty': ['l1'], 'loss': ['squared_hinge'],'C': Cvalues, 'dual': [False]},
                    {'penalty': ['l2'], 'loss': lossValues,'C': Cvalues}]

        elif ('nusvc' in self.classif.lower()):
            return [{'nu': [nu], 'kernel': ['rbf'],     'gamma': gammaValues},
                    {'nu': [nu], 'kernel': ['linear']},
                    {'nu': [nu], 'kernel': ['sigmoid'], 'gamma': gammaValues},
                    {'nu': [nu], 'kernel': ['poly'],    'gamma': gammaValues}]

        elif (self.classif.lower() in 'svc'):
            return [{'kernel': ['rbf'],     'gamma': gammaValues, 'C': Cvalues},
                    {'kernel': ['linear'],  'C': Cvalues},
                    {'kernel': ['sigmoid'], 'gamma': gammaValues, 'C': Cvalues},
                    {'kernel': ['poly'],    'gamma': gammaValues, 'C': Cvalues}]

        elif ('mlp' in self.classif.lower()):
            return [{'solver': ['lbfgs'], 'activation': activationValues, 'hidden_layer_sizes': hiddenLayers, 'learning_rate': learningRate},
                    {'solver': ['sgd'], 'activation': activationValues, 'hidden_layer_sizes': hiddenLayers, 'learning_rate': learningRate, 'batch_sizes': batchSizes},
                    {'solver': ['adam'], 'activation': activationValues, 'hidden_layer_sizes': hiddenLayers, 'learning_rate': learningRate, 'batch_sizes': batchSizes}]

        elif ('randomforest' in self.classif.lower()):
            return [{'criterion': ['gini'], 'n_estimators': estimatorValues, 'max_features': maxFeatValues, 'bootstrap': [True, False]},
                    {'criterion': ['entropy'], 'n_estimators': estimatorValues, 'max_features': maxFeatValues, 'bootstrap': [True, False]}]

        elif ('logit' in self.classif.lower()):
            return [{'solver': ['newton-cg'], 'penalty': ['l2'], 'C': Cvalues},
                    {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': Cvalues},
                    {'solver': ['liblinear'], 'penalty': penalties, 'C': Cvalues},
                    {'solver': ['saga'], 'penalty': penalties, 'C': Cvalues}]


if __name__ == '__main__':
    ML().main()
