from utils import Parsers, UtilMethods as Utils
import os
from pyspark import SparkConf
from operator import add
from nltk.util import ngrams, everygrams
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np


###############
# Processes feature extraction from dataset.
###############


class Extractor:

    def __init__(self, config, outputPath):
         self.config = config
         self.dictPath = config.get('prediction', 'dict.path')
         self.featType = config.get('prediction', 'feat.type')
         self.nbFeatType = self.featType.count('-') + 1
         self.sourceType = config.get('prediction', 'source.type')
         self.size = config.get('prediction', 'feat.size')
         self.minOcc = config.get('prediction', 'feat.minOcc')
         outputPath = Utils.normalizePath(outputPath)
         self.featFile = outputPath + self.featType
         self.cv = self.config.getboolean('prediction', 'use.crossvalid')
         self.task = self.config.get('prediction', 'task')
         if ('cross' in self.task):
             self.cv = True
         if self.cv:
             self.featFile += '.cv'
         if ('kmers' in self.featType):
             kmerfeats = 'kmers' + str(self.size) + '_minOcc' + str(self.minOcc)
             self.featFile = self.featFile.replace('kmers', kmerfeats)
         self.featFile += '.feat'


    def extractFeatures(self, dataPath, sparkContext, featPerInst):
        files, feats, kmerCounts, featPerInstance = [], [], [], []
        useKmer = True if 'kmers' in self.featType else False
        useProt = True if 'prot' in self.featType else False
        useDistinct = True if 'dist' in self.featType else False
        listContents = Parsers.parseDatasetContents(dataPath, self.featType, self.sourceType)

        if('dictionary' in self.featType):
            feats += sorted(self.loadDictionary(self.dictPath))
        else:
        #if('domains' in self.featType or 'kmers' in self.featType):
            featRDD = sparkContext.parallelize(listContents, numSlices=1000)
            featuresRDD = featRDD.map(lambda x: (x[1], self.getFeatures(x)))

            if(featPerInst):
                # get a list of features per instance for embeddings
                featPerInstance = featuresRDD.values().collect()
                print(len(featPerInstance), 'instances processed.')

            if (not os.path.isfile(self.featFile)):
                if(useKmer):
                    # filter RDD and return only kmers, "flatten" arrays to single list of kmers
                    kmerRDD = featuresRDD.filter(lambda x: "kmer" in x[0]).flatMap(lambda x: x[1])

                    # change each element to (k, v), reduce list by keys to group
                    # + count features, filter features by minOcc
                    minOcc = int(self.minOcc)
                    countAndFilter = kmerRDD.map(lambda x: (x, 1)).reduceByKey(add).filter(lambda x: x[1] >= minOcc)

                    # remove counts and collect only keys
                    kmerCounts = sorted(countAndFilter.collect())
                    feats += sorted(countAndFilter.keys().collect())

                    # filter out kmers already processed
                    featuresRDD = featuresRDD.filter(lambda x: "kmer" not in x[0])

                if(useProt):
                    # filter RDD and return only prot properties
                    protRDD = featuresRDD.filter(lambda x: "protanalys" in x[0])
                    # select (unique) feature names
                    feats += sorted(protRDD.flatMap(lambda x: x[1]).distinct().collect())
                    featuresRDD = featuresRDD.filter(lambda x: "protanalys" not in x[0])

                # get a flat list of all features
                #if(useDistinct):
                 #   completeFeatures = featuresRDD.flatMap(lambda x: x[1]).distinct()
                #else:
                    #completeFeatures = featuresRDD.flatMap(lambda x: x[1])
                completeFeatures = featuresRDD.flatMap(lambda x: x[1]).distinct()
                feats += completeFeatures.collect()

                if (len(feats) > 1):
                    allFeatures = ''.join(str(i) + '\n' for i in feats)
                    Utils.writeFile(self.featFile, allFeatures)

                if (len(kmerCounts) > 1):
                    kmerCounts = ''.join(str(i).replace('(\'', '').replace('\',', '\t').replace(')', '') + '\n' for i in kmerCounts)
                    Utils.writeFile(self.featFile + 'count', kmerCounts)

                print(len(feats), 'features extracted.')

            else:
             feats = self.loadFeatures()

        return feats, featPerInstance, kmerCounts


    def loadFeatures(self):
        return [line.rstrip() for line in open(self.featFile)]


    def getFeatures(self, info):
        # receives tuple in format:
        # ((fileName, content, sequenceID), featureType)
        type = info[1]
        content = info[0][1]

        if("kmer" in type):
            return self.kmers(content)
        elif("domain" in type or "go" in type):
            return self.ngrams(content, 1)
        elif("prot" in type):
            result, flattened, flattenedFlexDic = self.protAnalysis(content)
            # return only keys: '*' unpacking generalization for an iterable,
            # which in a dict is supposed to return its keys
            return [*flattenedFlexDic]


    def kmers(self, content):
        try:
            size = int(self.size)
            ngramG = ngrams(content, size)
            return [''.join(i) for i in list(ngramG)]
        except:
            size = self.size.replace('to',' ').split(' ')
            minsize = size[0]
            maxsize = size[1]
            everygrams(content, minsize, maxsize)


        # generate ngrams (k-mers) from sequence
    def ngrams(self, content, gramSize):
        feats = []
        contentList = content.split('\n')
        size = gramSize if gramSize > 0 else int(self.size)
        for i in range(len(contentList) - size + 1):
            feature = contentList[i:i + size]
            if(size == 1):
                feature = feature[0]
            else:
                feature = " ".join(feature)
            if(len(feature) > 1):
            # Comment out next line to keep domain name:
                if('.' in feature):
                    feature = feature.split('.')[0]
                feats.append(feature)
        return feats


    #generate everygrams (k-mers) from sequence
    def everygrams(self, content, minsize, maxsize):
        everygramG = everygrams(content, min_len=minsize, max_len=maxsize)
        return [''.join(i) for i in list(everygramG)]


    # load a dictionary of features
    def loadDictionary(self, dictPath):
        dict = Utils.readFileLines(dictPath)
        result = []
        if(dict):
            for word in dict:
                    if('\t') in word:
                        temp = word.split('\t')[1]
                        if(len(temp) > 1):
                            result.append(temp)
                    else:
                        result.append(word)
        else:
            print("Dictionary not loaded from dictionary path: ", dictPath)
            exit()
        return result


    def countOccurrence(self, dataPath, sparkContext):
        feats = self.loadFeatures()
        contentIds = []

        listContents = Parsers.parseDatasetContents(dataPath, self.featType, self.sourceType)
        parentDir = os.path.split(os.path.dirname(listContents[0][0][0]))[1]

        for info in listContents:
            filename = info[0][0]
            content = info[0][1]
            type = info[1]
            firstLine = Utils.readFileLines(filename)[0]
            id = firstLine.replace('>', '') if '|' in firstLine else firstLine.split('.')[0].replace('>','')
            label = Utils.getLabel(filename)

            # avoid cases in which test synthetic genes are long and
            # in the split different clusters share same (gene) id
            for item in contentIds:
                if(id in item[0] and type in item[1]):
                    id = id + '|'

            contentIds.append(tuple([id, type, content, label]))

        sourceRDD = sparkContext.parallelize(contentIds, numSlices=1000)
        occRDD = sourceRDD.map(lambda x: self.occurrence(x, feats))

        # combine features with same ID and filter out instances with not enough features
        reducedRDD = occRDD.reduceByKey(lambda x, y: self.mergeFeatsSameId(x, y))

        ids = reducedRDD.map(lambda x: x[0]).collect()
        occ = reducedRDD.map(lambda x: x[1][0]).collect()
        labels = reducedRDD.map(lambda x: x[1][1]).collect()

        print('Features loaded.')
        return np.array(ids), np.array(occ), np.array(labels), parentDir



    def extractRewardPerFeat(self, dataPath, outputPath, featType, sourceType, rewardType):

        rewardperfeat = {}
        # tuple of shape {(file, content, id),'kmers'}
        resultLabel = Parsers.parseDatasetContents(dataPath, featType, sourceType)
        fileindex = list(set([i[0][0] for i in resultLabel]))

        for item in resultLabel:
            filename = item[0][0]
            label = Utils.getLabel(filename)
            content = item[0][1]
            idx = fileindex.index(filename)
            occ = 1 if label == 1 else -1

            if(content in rewardperfeat):
                if('label' in rewardType):
                    rewardperfeat[content][idx] = occ
                else:
                    rewardperfeat[content][idx] += occ
            else:
                rewardperfeat[content] = [0] * len(fileindex)
                if('label' in rewardType):
                    rewardperfeat[content][idx] = occ
                else:
                    rewardperfeat[content][idx] += occ

        outputstr = ''
        for k, v in rewardperfeat.items():
            outputstr += k + '\t' + (',').join(map(str, v)) + '\n'
        Utils.writeFile(outputPath, outputstr[:-1])

        return rewardperfeat




    def filterLen(self, thisTup, feats):
        id = thisTup[0]
        content = thisTup[1]
        resultTup = thisTup
        # means instance has only one feature type avail and
        # wasnt reduced by key - so clean tuple to return it
        if(len(content) > 2):
            resultTup = tuple([id, (content[0], content[2])])
        if(len(content[0]) >= len(feats)):
            return resultTup


    # handle instances with same ID to
    # either concatenate them, or
    # sum occurrences (keep length)
    def mergeFeatsSameId(self, instance1, instance2):
        occurences1 = instance1[0]
        occurences2 = instance2[0]
        try:
            type1 = instance1[2] # if merging for 2nd time, this will be the label instead
        except IndexError:
            type1 = "merged"
        type2 = instance2[2]
        label = instance2[1]

        mergedResult = []
        if('prot' in str(type1) or 'prot' in str(type2)):
            mergedResult = occurences1 + occurences2
        else:
            # sums value in each index for two arrays of same length
            mergedResult = (np.array(occurences1) + np.array(occurences2)).tolist()
        return ([mergedResult, label])


    # count occurrences (isolated) task
    def occurrence(self, entry, feats):
        occFeatures = []
        id = entry[0]
        type = entry[1]
        content = entry[2]
        label = entry[3]

        if('protein' in str(feats) and 'prot' in type):
            result, flattened, flattenedFlexDic = self.protAnalysis(content)
            for feat in feats:
                if('prot' in feat):
                    occFeatures.append(flattenedFlexDic.get(feat,0))
        else:
            for feat in feats:
                if ('protein' not in feat):
                    count = content.count(feat)
                    occFeatures.append(int(count))

        return ([id, [occFeatures, label, type]])


    # flatten a 'mixed' dictionary of {floats, float arrays, etc}
    def flatten(self, key, values):
        flat = {}
        for index, value in enumerate(values):
            if(isinstance(key, str)):
                flat[key+'-'+str(index+1)] = value
            else:
                feat = 'protein-' + key[index]
                if(feat in flat):
                    flat[feat] += value
                else:
                    flat[feat] = value
        return flat


    def setMinOcc(self, minOcc):
        self.minOcc = minOcc


    def setSize(self, size):
        self.size = size


    def initSpark(self):
        # create spark context instance for pipeline
        conf = SparkConf().setAppName('extractor')
        conf = (conf.setMaster('local[*]')
                .set('spark.executor.memory', '8G')
                .set('spark.driver.memory', '50G')
                .set('spark.driver.maxResultSize', '50G')
                .set('spark.network.timeout', '10000s')
                .set('spark.executor.heartbeatInterval', '120s')
                .set('spark.pyspark.python', os.path.dirname(__file__).join('.env/bin/python')))
        return conf


    def protAnalysis(self, content):
        result, resultFlexDic = dict(), dict()
        content = Parsers.normalizeSequence(content, self.sourceType)
        protein = ProteinAnalysis(content)

        result['proteinMWeight']   = protein.molecular_weight()
        result['proteinAroma']     = protein.aromaticity()
        result['proteinInstab']    = protein.instability_index()
        result['proteinIsoelec']   = protein.isoelectric_point()
        result['proteinGravy']     = protein.gravy()

        proteinStructure = protein.secondary_structure_fraction()
        protStruct = self.flatten('proteinSecstruc', proteinStructure)

        result = {**protStruct, **result}

        # merge result and protein Structure
        flexibility = protein.flexibility()
        flexibFlat = self.flatten('proteinFlex', flexibility)
        flexibAmino = self.flatten(list(content), flexibility)

        flattened = {**flexibFlat, **result}
        flattenedFlexDic = {**flexibAmino, **result}

        return result, flattened, flattenedFlexDic,


    def loadProtAnalysis(self):
        return ['proteinMWeight', 'proteinFlex', 'proteinAroma', 'proteinInstab', 'proteinIsoelec', 'proteinGravy', 'proteinSecstruc']