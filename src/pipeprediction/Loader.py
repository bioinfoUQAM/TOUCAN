import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from utils import Parsers
from utils import UtilMethods as Utils
from pipeprediction import Extractor
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros
from itertools import cycle
from gensim.models import KeyedVectors
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

        if (self.useEmbeddings):
            self.featType = config.get('prediction', 'feat.type')
            self.featSize = config.get('prediction', 'feat.size')
            self.minOcc = config.get('prediction', 'feat.minOcc')
            self.embedType = config.get('prediction', 'embeddings.type')
            self.embedSize = config.getint('prediction', 'embeddings.length')
            self.embeddingsName =  self.featType + self.featSize + 'minOcc' + str(self.minOcc) \
                                  + self.embedType.lower() + str(self.embedSize)
            self.embeddingsW2V = self.outputPath + self.embeddingsName + '.w2v'

        self.dictionary = dict()
        self.extractor = Extractor.Extractor(config, outputPath)
        self.featType = config.get('prediction', 'feat.type')
        self.dictionary = dict()
        self.featmapFile = outputPath + self.featType + str(self.extractor.size) + '_minOcc' + str(self.extractor.minOcc) + '.map'
        self.maxLength = 0


    # load trained embeddings from a w2v file
    def loadW2V(self):
        try:
            word2vec = KeyedVectors.load_word2vec_format(self.embeddingsW2V, binary=False)
        except (UnicodeDecodeError, ValueError) as e:
            word2vec = KeyedVectors.load_word2vec_format(self.embeddingsW2V, binary=True)
        return word2vec


    def loadEmbedWeights(self):
        print('Loading embeddings:', self.embeddingsW2V)
        word2vec = self.loadW2V()
        vocabSize = len(self.dictionary)
        vectorsLen = self.embedSize
        embeddingsMatrix = zeros((vocabSize, vectorsLen))

        for word, i in self.dictionary.items():
            try:
                embeddingVector = word2vec.word_vec(word)
                if embeddingVector is not None:
                    embeddingsMatrix[i] = embeddingVector
            except:
                pass

        if(len(embeddingsMatrix) > 0 ):
            print(str(len(embeddingsMatrix)), 'embeddings loaded.')
        else:
            print('No embeddings matching!')
            exit()

        return embeddingsMatrix


    # build dictionary of {unit, [representation]} (e.g, {word, [ID]})
    def build_dict(self, sparkContext):
        feats, featPerInstance, featCounts = self.extractor.extractFeatures(self.trainPath, sparkContext, featPerInst=False)
        for feat in feats:
            # if('prot' not in feat):
            self.dictionary[feat] = len(self.dictionary)
        print('Done with dictionary.')


    def broadcast_dump(self, value, f):
        pickle.dump(value, f, 4)  # was 2, 4 is first protocol supporting >4GB
        f.close()
        return f.name


    def buildDataset(self, path, sparkContext):
        result, ids, labels = [], [], []
        files = Utils.listFilesExt(path, 'fasta')

        # define pickle protocol to bypass 4GiB pickling limit
        broadcast.Broadcast.dump = self.broadcast_dump    

        datasetRDD = sparkContext.parallelize(files, numSlices=5000)
        datasetRDD = datasetRDD.map(lambda x: self.addInstance(x))
        dataset = datasetRDD.collectAsMap()

        # get max length among all
        maxLen = int(datasetRDD.sortBy(lambda x: x[0][1], False).first()[0][1])
        self.maxLength = maxLen if maxLen > self.maxLength else self.maxLength

        # 0 = fasta.id, 1 = instance length, 2 = file name, 3 = label
        for k, v in dataset.items():
            ids.append(k[0])
            labels.append(int(k[3]))
            result.append(v)
        print('Done building dataset.')
        return ids, result, labels


    # Transform data into vectors using dictionary index
    def addInstance(self, file):
        sequences = Parsers.parseFasta(file)
        label = Utils.getLabel(file)
        seqFile = os.path.basename(file)
        for fasta in sequences:
            instanceValues = list()
            sequence = str(fasta.seq.upper()) # deals with files having lowercase seqs
            if('kmer' in self.featType):
                sequence = self.extractor.kmers(sequence)
                for word in sequence:
                    instanceValues.append(self.dictionary.get(word)) if word in self.dictionary else ""
            if('prot' in self.featType):
                result, flattened, flattenedFlexDic = self.extractor.protAnalysis(sequence)
                for value in flattened.values():
                    instanceValues.append(value)

            size = len(instanceValues)
            instanceID = (fasta.id, int(size), seqFile, str(label))

        return instanceID, instanceValues


    def padSequences(self, data, type):
        print('\nPadding sequences (samples x time)...')
        padData = sequence.pad_sequences(data, maxlen=self.maxLength, padding='post', dtype=np.uint8)
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
                batch_x = sequences[i:i + batch_size]
                batch_y = targets[i:i + batch_size]
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