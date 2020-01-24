from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from pipeprediction import Visuals
from sklearn.externals import joblib
import os

###############
# Processes dimensionality reduction
# for instance visualization
###############


class DimensionHandler:


    def __init__(self, config, outputPath):
        self.config = config
        self.outputPath = outputPath
        self.plotter = Visuals.Visuals()
        self.name = config.get('prediction', 'feat.reduc')
        self.components = 2 #200 if len(features) > 200 else len(features)
        self.featType = config.get('prediction', 'feat.type')
        self.size = int(config.get('prediction', 'feat.size'))
        self.minOcc = int(config.get('prediction', 'feat.minOcc'))
        self.dimFile = outputPath + str(self.name).lower() + str(self.components) + '_' + self.featType + str(self.size) + '_minOcc' + str(
        self.minOcc) + '.' + str(self.name).lower()
        self.graphFile = self.dimFile + '.png'


    def trainMethod(self, trainOcc, trainLabels, ):
        method = self.getMethod()
        if(not os.path.isfile(self.dimFile)):
            print('Computing', self.name, '...')
            trainOcc = method.fit_transform(trainOcc)
            joblib.dump(method, self.dimFile)
            self.plotMethod(trainOcc, trainLabels)
        else:
            print(str(self.name), 'already computed.')
        return trainOcc


    def testMethod(self, testOcc):
        method = joblib.load(self.dimFile)
        testOcc = method.transform(testOcc)
        return testOcc


    def getMethod(self):
        if ('pca' in str(self.name).lower()):
            return PCA(n_components=self.components)
        elif ('mds' in str(self.name).lower()):
            return MDS(n_components=self.components)
        elif ('tsne' in str(self.name).lower()):
            return TSNE(n_components=self.components)


    def plotMethod(self, trainOcc, trainLabels):
        self.plotter.plotDimReduction(self.name, trainOcc, trainLabels, self.graphFile, self.featType, self.size)


    def getName(self):
        return str(self.name).lower()


    def getOutFile(self, classif):
        return self.outputPath + classif + '_' + self.name.lower() + str(self.components) + '_' + self.featType + str(
            self.size) + '_minOcc' + str(self.minOcc)