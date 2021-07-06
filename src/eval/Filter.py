from utils import UtilMethods as Utils
from utils import Parsers as Parsers
from pipeprediction import RL
import os, re



class Filter:

    def __init__(self, config, sparkContext):
        self.useFilter = config.getboolean('eval', 'rule.filter')
        self.useSplit = config.getboolean('eval', 'split')
        self.filterMap = config.get('eval', 'gene.map')
        self.filterList = config.get('eval', 'rule.list')

        self.QFilter = config.getboolean('eval', 'q.filter')
        self.splitthreshold = int(config.get('eval', 'split.threshold'))
        self.threshold = float(config.get('eval', 'threshold'))
        self.genesToFilter = self.loadFilterMap(sparkContext) if (self.useFilter or self.useSplit) else ''
        self.sparkContext = sparkContext
        self.QLearner = RL.QLearner()


    def doFilter(self, predictions, resultPath, similarityDictionary):

        if (self.useFilter or self.useSplit):
            predictions = self.Filter.computeFilterSplit(predictions, doFilter=self.useFilter, doSplit=self.useSplit)

        if (self.QFilter):
            predictions = self.QLearner.test(self.sparkContext, predictions, self.threshold, resultPath, similarityDictionary)

        return predictions



    def computeFilterSplit(self, predictions, doFilter, doSplit):
        result = []
        countFilter, countSplit = 0,0
        # only filters OUT selected genes
        for i in predictions:
            score = float(i.split('\t')[1])
            if(score > self.threshold):
                genes = i.split('\t')[0].split('|')
                filterMatch = [1 if gene in self.genesToFilter else 0 for gene in genes]

                splitIdx = self.findVoteSplits(filterMatch)
                tempcluster = ''
                j = 0
                while j < len(genes):
                    # if at split index or at end of loop
                    if((doSplit and j in splitIdx) or j == len(genes)-1):
                        # if split index is found, then update index
                        if(j in splitIdx):
                            j = splitIdx[j]
                            countSplit += 1
                        tempcluster = tempcluster[:-1]
                        # if cluster contains less than 3 genes, keep it for next cluster
                        if(tempcluster.count('|') > 3):
                            tempcluster = tempcluster[1:] if tempcluster.startswith('|') else tempcluster
                            result.append(tempcluster + '\t' + str(score))
                            tempcluster = ""
                        else:
                            tempcluster = tempcluster + '|'
                    else:
                        # retrieve flag to filter out gene or keep it
                        match = filterMatch[j] if doFilter else 0
                        countFilter = countFilter + 1 if match == 1 else countFilter
                        tempcluster += genes[j] + '|' if match == 0 else ""
                    j = j + 1
            else:
                result.append(i)

        if(doFilter): print('Done filtering out', str(countFilter), 'genes.')
        if(doSplit): print('Done making', str(countSplit), 'splits.')
        return result


    def findVoteSplits(self, filterMatch):
        # 0 = genes to keep, 1 = genes to filter
        splitIndices = {}
        # find indexes matching to sequence of positions to filter
        localVote, index, maximum = 0, 0, 0

        if(sum(filterMatch) > 0):
            for i, value in enumerate(filterMatch):
                if (value > 0):
                    localVote += 1
                else:
                    maximum = max(maximum, localVote)
                    localVote = 0
                    # get start and end position of split when reaching threshold
                    if(maximum >= self.splitthreshold):
                        start = i - maximum
                        end = i - 1
                        splitIndices[start] = end
                        maximum = 0
        return splitIndices


    def loadFilterMap(self, sparkContext):
        filterList = Utils.readFileLines(self.filterList)
        # returns tuple (((file, content), 'domains'))
        content = Parsers.parseDatasetContents(self.filterMap, 'domains', 'domains')

        domRDD = sparkContext.parallelize(content, numSlices=1000)
        domainsRDD = domRDD.map(lambda x: (Utils.getFileName(x[0][0]).replace('.domains', ''), x[0][1]))

        # lists genes that have any domains in filterList
        # discards ".\d+" end of Pfam ID
        filter = domainsRDD.filter(lambda x: any(domain in filterList for domain in re.split("[\n.]", x[1])))

        result = filter.collectAsMap().keys()
        genes = sorted([i for i in result])

        print('Loaded filter:', len(genes), ' genes will be filtered from', len(filterList), 'domains.')
        return genes


    def getParams(self, resultPath):
        if (self.useSplit or self.useFilter):
            resultPath += '.' + os.path.basename(self.filterList).split('.')[0]
            if (self.useSplit):
                resultPath += 'Split' + str(self.splitthreshold)
            if (self.useFilter):
                resultPath += 'Filter'

        if (self.QFilter):
            resultPath += '.qfilter_' + self.QLearner.params
            resultPath.replace('metrics', 'metricsQLearner')

        return resultPath



