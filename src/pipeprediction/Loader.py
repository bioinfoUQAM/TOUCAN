import os
import sys, re
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from utils import Parsers
from utils import UtilMethods as Utils
from pipeprediction import Extractor
#from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from itertools import cycle
from operator import add
from gensim.models import KeyedVectors, Word2Vec
import numpy as np
from pyspark import broadcast
import pickle
sys.stderr = stderr


###############
# Manages dataset loading
###############


class Loader:

    def __init__(self, config, outputPath):
        self.sourcePath = config.get('prediction', 'source.path')
        self.sourcePath = Utils.normalizePath(self.sourcePath)
        self.trainPath = self.sourcePath + 'train/'
        self.outputPath = self.sourcePath + 'metricsDL/'
        self.sourceType = config.get('prediction', 'source.type')
        self.useEmbeddings = bool(config.get('prediction', 'use.embeddings'))
        self.embedPath = config.get('prediction', 'embed.path')
        self.embedPath = Utils.normalizePath(self.embedPath)

        if (self.useEmbeddings):
            self.featType = config.get('prediction', 'feat.type')
            self.featSize = config.get('prediction', 'feat.size')
            self.minOcc = config.get('prediction', 'feat.minOcc')
            self.embedSize = config.getint('prediction', 'embeddings.length')
            self.embeddingsName =  self.featType + self.featSize + 'minOcc' + str(self.minOcc) \
                                  + str(self.embedSize) + 'd'

        self.dictionary = dict()
        self.extractor = Extractor.Extractor(config, outputPath)
        self.featType = config.get('prediction', 'feat.type')
        self.maxLength = 0


    def getEmbeddings(self):
        matrix = np.zeros((self.dictLength(), self.embedSize))
        embfiles = Utils.listFilesExt(self.embedPath, 'w2v')
        for i in embfiles:
            if ('kmer' in i.lower() and 'kmer' in self.featType.lower()):
                matrix = self.mapEmbedWeights(i, 'kmer', matrix)
            elif ('domain' in i.lower() and 'domain' in self.featType.lower()):
                matrix = self.mapEmbedWeights(i, 'domain', matrix)
            elif ('go' in i.lower() and 'go' in self.featType.lower()):
                matrix = self.mapEmbedWeights(i, 'go', matrix)
        return matrix


    # load trained embeddings from a w2v file
    def loadW2V(self, embfile):
        ext = Utils.getFileExt(embfile)
        word2vec = []

        if('compact') in embfile:
            word2vec = Word2Vec.load(embfile, mmap='r')
        elif('.bin' in embfile):
            word2vec = KeyedVectors.load_word2vec_format(embfile, binary=True, limit=2000000)
        else:
            word2vec = KeyedVectors.load_word2vec_format(embfile, binary=False, limit=2000000)

        return word2vec


    def mapEmbedWeights(self, embfile, ftype, embeddingsMatrix):
        print('Loading embeddings:', embfile)
        word2vec = self.loadW2V(embfile)
        notfound = []
        # select features per type
        feature = 'PF\d' if 'domain' in ftype else 'GO:' if 'go' in ftype else ''
        if(feature):
            selectfeats = [item for item in self.dictionary.items() if re.match(feature, str(item[0]))]
        else:
            selectfeats = [item for item in self.dictionary.items() if str.isalpha(item[0])]

        matches = 0
        for word, i in selectfeats:
            try:
                embeddingVector = word2vec.word_vec(word)
                if embeddingVector is not None:
                    embeddingsMatrix[i] = embeddingVector
                    matches += 1
            except:
                notfound.append(word)
                pass

        if(len(embeddingsMatrix) > 0 ):
            print(str(matches), 'out of', str(len(selectfeats)),'features found.')
        else:
            print('No embeddings matching!')
            exit()

        return embeddingsMatrix


    # build dictionary of {unit, [representation]} (e.g, {word, [ID]})
    def build_dict(self, sparkContext):
        feats, featPerInstance, featCounts = self.extractor.extractFeatures(self.trainPath, sparkContext, featPerInst=False)
        feats = set(feats)
        for feat in feats:
            self.dictionary[feat] = len(self.dictionary)
        print('Done with dictionary.')


    def broadcast_dump(self, value, f):
        pickle.dump(value, f, 4)  # was 2, 4 is first protocol supporting >4GB
        f.close()
        return f.name


    def buildDataset(self, path, sparkContext):
        result, ids, labels = [], [], []
        #files = Utils.listFilesExt(path, 'fasta')
        # define pickle protocol to bypass 4GiB pickling limit
        broadcast.Broadcast.dump = self.broadcast_dump

        dataset = Parsers.parseDatasetContents(path, self.featType, self.sourceType)
        parentDir = os.path.split(os.path.dirname(dataset[0][0][0]))[1]

        listRDD = sparkContext.parallelize(dataset, numSlices=5000)
        # X tuple in format:
        # ((fileName, content, sequenceID), featureType)
        featuresRDD = listRDD.map(lambda x: (x[0][2], self.extractor.getFeatures(x)))

        concatRDD = ''
        if("pfam" in self.featType): # concatenate by pfam ID: label positive if at least one file contains domain
            concatRDD = featuresRDD.map(lambda x: (''.join(x[1]), [x[0]])).reduceByKey(add)
        else:
            # concatenate contents by file ID
            concatRDD = featuresRDD.map(lambda x: (x)).reduceByKey(add)

        # add instance
        # X tuple in format: (file ID, [feature, feature, ...]])
        datasetRDD = concatRDD.map(lambda x: self.addInstance(x))
        dataset = datasetRDD.collectAsMap() #if "pfam" not in self.featType else datasetRDD.collect()

        # get max length among all
        maxLen = 1 if "pfam" in self.featType else int(datasetRDD.sortBy(lambda x: x[0][1], False).first()[0][1])
        self.maxLength = maxLen if maxLen > self.maxLength else self.maxLength

        # 0 = fasta.id, 1 = instance length, 2 = file name, 3 = label
        for k, v in dataset.items():
            id, label = k[0], int(k[2])
            ids.append(id)
            labels.append(label)
            result.append(v)
        print('Done building dataset.')

        return ids, result, labels, parentDir


    def addInstance(self, info):
        # receives tuple in format:
        # (fileID, [list of extracted features])
        # for pfam only (inverted):
        # (pfamID, [list of fileIDs])
        instanceValues = list()

        label = Utils.getLabel(''.join(info[1])) if 'pfam' in self.featType else Utils.getLabel(info[0])
        fileID = info[0]
        features = [info[0]] if 'pfam' in self.featType else info[1]

        for feat in features:
            instanceValues.append(self.dictionary.get(feat)) if feat in self.dictionary else ""

        size = len(instanceValues)
        instanceID = (fileID, int(size), str(label))

        return instanceID, instanceValues



    def padSequences(self, data, type):
        print('\nPadding sequences (samples x time)...')
        #padData = sequence.pad_sequences(data, maxlen=self.maxLength, padding='post', dtype=np.uint8)
        padData = pad_sequences(data, maxlen=self.maxLength, padding='post', dtype=np.uint8)
        print('Seq shape: ')
        print('Type:', str(type))
        print('Shape:', padData.shape)
        print('Done padding!')

        return padData


    #####
    # returns the generator as well as the number of resulting batches in it (steps in keras)
    # e.g.
    # training_generator, training_steps = batch_gen(train_x, batch_size=BATCH_SIZE)
    # model.fit_generator(training_generator, steps_per_epoch=training_steps, ........
    #####
    def batch_genMultiple(self, sequences, targets, batch_size, nn, max_len=None):
        # expand 'sequences' when dimension is less than 3 for cnn or lstm
        # if ('lstm' in nn.lower() or 'cnn' in nn.lower()):
        #     if (int(sequences.ndim) < 3):
        #         sequences = np.expand_dims(sequences, 2)

        if len(sequences) != len(targets):
            print('number of sequences and targets is different')
            exit()

        def gen():
            for i in range(0, len(sequences), batch_size):
                batch_xK = np.array(sequences[i:i + batch_size])
                batch_xD = np.array(sequences[i:i + batch_size])
                batch_xG = np.array(sequences[i:i + batch_size])
                batch_y = np.array(targets[i:i + batch_size])

                if not max_len:
                    batch_xK = pad_sequences(batch_xK)
                    batch_xD = pad_sequences(batch_xD)
                    batch_xG = pad_sequences(batch_xG)
                    yield [batch_xK, batch_xD, batch_xG], batch_y
                else:
                    length = min(max_len, max(map(len, batch_xK)))  # decides whether the batch is shorter than max_len
                    batch_xK = pad_sequences(batch_xK, maxlen=length)
                    batch_xD = pad_sequences(batch_xD, maxlen=length)
                    batch_xG = pad_sequences(batch_xG, maxlen=length)
                    yield [batch_xK, batch_xD, batch_xG], batch_y

        return cycle(gen()), len(sequences) // batch_size+1  # cycle loops over the generator


    def batch_gen(self, sequences, targets, batch_size, nn, max_len=None):
        # expand 'sequences' when dimension is less than 3 for cnn or lstm
        if ('lstm' in nn.lower() or 'cnn' in nn.lower()):
            if (int(sequences.ndim) < 3):
                sequences = np.expand_dims(sequences, 2)

        if len(sequences) != len(targets):
            print('number of sequences and targets is different')
            exit()

        def gen():
            for i in range(0, len(sequences), batch_size):
                batch_x = np.array(sequences[i:i + batch_size])
                batch_y = np.array(targets[i:i + batch_size])
                if not max_len:
                    yield pad_sequences(batch_x), batch_y
                else:
                    length = min(max_len, max(map(len, batch_x)))  # decides whether the batch is shorter than max_len
                    yield pad_sequences(batch_x, maxlen=length), batch_y

        return cycle(gen()), len(sequences) // batch_size+1  # cycle loops over the generator


    def dictLength(self):
        return len(self.dictionary)


    def getMaxLength(self):
        return self.maxLength


    def setMaxLength(self, value):
        self.maxLength = value

    def getTupleFilename(self, tup):
        return Utils.getFileName(tup[0]).split('.')[0]