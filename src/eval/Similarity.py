from utils import UtilMethods as Utils
import os
import mmap, re
from pandas import read_csv
import timeit


class Similarity:

    def __init__(self, config):
        self.useSimilarity = config.getboolean('eval', 'similarity')
        self.similarityPath = config.get('eval', 'similarity.path')
        self.task = config.get('eval', 'task')

        # load similarity as dictionary[id1|id2] = pident
        #start_time = timeit.default_timer()
        self.dictionary = self.loadSimilarityDic()
        #print('Loading as original: ',timeit.default_timer() - start_time)


    def getParams(self, resultPath):
        if (self.useSimilarity):
            resultPath += '.similarity'
        return resultPath


    def computePredictionSimilarity(self, goldGenes, predictedGenes, genesFound):
        pairsDic = {}
        # filter already found genes from similarity comparison
        goldGenes = [gene for gene in goldGenes if gene not in genesFound]
        predictedGenes = [gene for gene in predictedGenes if gene not in genesFound]
        similarityScore = 0.0
        pairs = []

        # if there are still genes left
        if(len(predictedGenes) > 0 and len(goldGenes) > 0):
            for i in goldGenes:
                oldscore = 0
                for j in predictedGenes:
                    pair = i + "|" + j
                    # retrieve similarity for each [goldGene | predictedGene] pair
                    score = self.dictionary.get(pair)
                    # adjust similarity score to %
                    score = float(score) / 100 if score is not None else 0.0

                    # keep the highest score for gold gene
                    if(score > oldscore):
                        pairsDic[i] = [j, score]
                        oldscore = score

        if(pairsDic):
            # sort pairs in descending order
            pairsDic = sorted(pairsDic.values(), key = lambda x: x[1], reverse=True)
            similarPredictedGenes = set()
            # add similarity of highest score for predicted gene
            for pair in pairsDic:
                if(pair[0] not in similarPredictedGenes):
                    similarityScore += pair[1]
                    similarPredictedGenes.add(pair[0])

        return similarityScore



    def loadSimilarityDic(self):
        similaritydic = {}
        similarity = ""
        output = "id\tpident\tbitscore\tqcovs"

        if ("eval" in self.task and self.useSimilarity):
            similarity = Utils.readFileLines(self.similarityPath)[1:]
            if (len(similarity) > 0):
                for i in similarity:
                    ids = i.split('\t')[0]
                    score = float(i.split('\t')[1])
                    oldscore = similaritydic.get(ids)
                    oldscore = float(oldscore) if oldscore is not None else 0.0
                    score = max(score, oldscore)
                    similaritydic[ids] = score

                print("Loaded dictionary of", len(similaritydic), "similarity pairs.")

        return similaritydic



######################################################

    def getSimilarityScore(self, pair):
        pair = pair.replace('_', '\_').replace('|', '\|') + '.*'
        pattern = pair.encode("utf-8")
        result = re.findall(pattern, self.dictionaryNew)
        result = [i.decode() for i in result]
        score = 0
        for i in result:
            temp = float(i.split('\t')[1])
            score = max(score, temp)
        return score


    def loadSimilarities(self):
        columns = ['id','pident','bitscore','qcovs']
        df = read_csv(self.similarityPath, sep='\t', names=columns, index_col=False)
        return df

    def loadSimilarityDicNew(self):
        similaritydic = ""
        if("eval" in self.task and self.useSimilarity):
            with open(self.similarityPath, 'r', encoding="utf-8") as thisfile:
                filesize = os.path.getsize(self.similarityPath)
                similaritydic = mmap.mmap(thisfile.fileno(), filesize, access=mmap.ACCESS_READ)
        return similaritydic






