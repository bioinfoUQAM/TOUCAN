from utils import UtilMethods as Utils
from pipeprediction import Extractor, DimensionHandler, Loader
import re, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier as GP
from sklearn.ensemble import RandomForestClassifier as RF, IsolationForest as IF
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.covariance import EllipticEnvelope as RC  #RC = Robust Covariance
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as pfrs
from sklearn.externals import joblib
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
        self.cvPath = self.path + 'crossvalid/'
        self.testPath = self.path + 'test/'
        self.outputPath = self.path + 'metrics/'
        self.task = self.config.get('prediction', 'task')
        self.posPerc = int(self.config.get('prediction', 'pos.perc'))
        self.classif = self.config.get('prediction', 'classifier')
        self.cv = self.config.getboolean('prediction', 'use.crossvalid')
        os.makedirs(os.path.dirname(self.outputPath), exist_ok=True)
        self.extractor = Extractor.Extractor(self.config, self.outputPath)
        self.loader = Loader.Loader(self.config, self.outputPath)
        self.dimHandler = DimensionHandler.DimensionHandler(self.config, self.outputPath)
        self.outFile = ''
        if (not 'none' in self.dimHandler.name.lower()):
            self.outFile = self.dimHandler.getOutFile(self.classif)
        else:
            self.outFile = self.outputPath + self.classif + '_' + self.extractor.featType
            if('kmers' in self.extractor.featType):
                kmerfeats = 'kmers' + str(self.extractor.size) + '_minOcc' + str(self.extractor.minOcc)
                self.outFile = self.outFile.replace('kmers', kmerfeats)
                #self.outFile +=  str(self.extractor.size) + '_minOcc' + str(self.extractor.minOcc)
        if(self.cv):
            self.trainPath = self.cvPath
            self.outFile += '_cv10'

        self.modelFile = self.outFile + '.model.pkl'
        self.classifier = self.setUpClassifier()


    def main(self):
        sparkContext = SparkContext(conf=self.extractor.initSpark())
        if (not os.path.isfile(self.extractor.featFile)):
            self.extractor.extractFeatures(self.trainPath, sparkContext, featPerInst=False)
        if (not os.path.isfile(self.modelFile)):
            print('Training...')
            IDs, x_occ, y_labels, parentDir = self.extractor.countOccurrence(self.trainPath, sparkContext)
            if(not 'none' in self.dimHandler.name.lower()):
                x_occ = self.dimHandler.trainMethod(x_occ, y_labels)

            if (self.cv):
                seed = 5
                np.random.seed(seed)
                i = 1
                kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

                for train, valid in kfold.split(x_occ, y_labels):
                    self.classifier.fit(x_occ[train], y_labels[train])
                    output, IDoutput = self.getPredictions(IDs[valid], x_occ[valid], y_labels[valid])
                    Utils.writeFile(self.outFile + str(i) + '.valid', output)
                    Utils.writeFile(self.outFile + str(i) + '.IDs.valid', IDoutput)

            else:
                self.classifier.fit(x_occ, y_labels)

            joblib.dump(self.classifier, self.modelFile)
            print('Model saved. \nPredicting...')

        else:
            self.classifier = joblib.load(self.modelFile)
            print('Model loaded. \nPredicting...')

        if('randomforest' in self.classif.lower() and not os.path.isfile(self.extractor.featFile + 'importance')):
            self.getRFImportance()

        if('validation' in self.task):
            IDs, x_occ_val, y_labels_val, parentDir = self.extractor.countOccurrence(self.validPath, sparkContext)

            output, IDoutput = self.getPredictions(IDs, x_occ_val, y_labels_val)
            Utils.writeFile(self.outFile + '.valid', output)
            Utils.writeFile(self.outFile + '.IDs.valid', IDoutput)

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
            return svm.LinearSVC()

        # in case positive perc is low,
        # NuSVC nu param has to be adjusted
        elif('nusvc' in self.classif.lower()):
            if(self.posPerc < 30):
                thisNu = self.posPerc / 100
                return svm.NuSVC(nu=thisNu)
            else:
                return svm.NuSVC()

        elif(self.classif.lower() in 'svc'):
            # pass 'probability=True' if confidence values must be computed
            return svm.SVC()
        elif ('svr' in self.classif.lower()):
            return svm.SVR()
        elif ('mlp' in self.classif.lower()):
            return MLP()
        elif ('gaussian' in self.classif.lower()):
            return GP()
        elif('randomforest' in self.classif.lower()):
            return RF()
        elif('logit' in self.classif.lower()):
            return Logit()
        elif('mlp' in self.classif.lower()):
            return MLP()
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


if __name__ == '__main__':
    ML().main()