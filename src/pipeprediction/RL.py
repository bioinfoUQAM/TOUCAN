from utils import UtilMethods as Utils, Parsers
from pipeprediction import Extractor
import numpy as np
import random
from operator import add
from pyspark import SparkContext
import os

class QLearner:

    def __init__(self):
        self.config = Utils.loadConfig()
        self.path = self.config.get('prediction', 'source.path')
        self.path = Utils.normalizePath(self.path)
        self.trainPath = self.path + 'train/'
        self.testPath = self.path + 'test/'
        self.outputPath = self.path + 'metricsQLearner/models/'
        self.geneMapPath = self.config.get('eval', 'filter.map')
        self.geneMap = {}
        self.extractor = Extractor.Extractor(self.config, self.outputPath)
        self.rewardType = 'occ'
        self.rewardPath = self.outputPath + self.rewardType + 'PerDomains.feat' # pfam domain list
        self.rewardList, self.rewardIDs, self.rewardLabels = '', '', ''
        self.actions = ['keep','skip']
        self.task = 'train'
        self.rewardTable, self.QTable = [], []
        self.episodes = int(self.config.get('prediction', 'episodes'))
        # hyperparams
        self.alpha = float(self.config.get('prediction', 'alpha')) # learning rate
        self.gamma = float(self.config.get('prediction', 'gamma')) # discount factor
        self.epsilon = float(self.config.get('prediction', 'epsilon')) # exploration
        self.penaltyThreshold = float(self.config.get('prediction', 'penalty.threshold')) # negative rewards mean penalty
        self.keepskipThreshold = float(self.config.get('prediction', 'keepskip.threshold')) # keep reward ratio wrt skip reward for domain to be kept
        self.useSimilarityWeight = False
        self.useCompWeights = False
        self.useNeighborWeight = self.config.getboolean('prediction', 'neighbor.weight')
        self.useDryIslands = self.config.getboolean('prediction', 'dry.islands')
        self.useAvAction = self.config.getboolean('prediction', 'average.action')
        self.weightsPath = self.config.get('eval', 'weights')
        self.weights = Utils.readFileLines(self.weightsPath) if self.useCompWeights or self.useNeighborWeight else ''
        self.params = self.rewardType  + '_keepgt'+ str(self.keepskipThreshold)  +'skip' + '_ep' + str(self.episodes) + '_alpha'  + str(self.alpha) + '_gamma' + str(self.gamma) + '_eps' + str(self.epsilon)
        self.params += '_neighbor' if self.useCompWeights else ''
        self.QTablePath = self.outputPath + 'Qtable_'+ self.params + '.npy'
        self.rewardTablePath = self.outputPath + 'Rewards_' + self.params + '.npy'
        self.IDmapPath = self.outputPath + 'RewardIDsmap_' + self.params + '.map'

    def main(self):
        sparkContext = SparkContext(conf=self.extractor.initSpark())
        if('train' in self.task):
            self.getRewards()
            self.train(sparkContext, outputStats=True)


    def prepareData(self, path, sparkContext):
        dataset = Parsers.parseDatasetContents(path, 'domains_pfam', 'domains')
        contentRDD = sparkContext.parallelize(dataset, numSlices=1000)
        perinstanceRDD = contentRDD.map(lambda x: (x[0][2], [x[0][1]])).reduceByKey(add)
        # format tuple {filename, ([domains], fastaID(genes))}
        return perinstanceRDD.collect()



    def train(self, sparkContext, outputStats):
        dataset = self.prepareData(self.trainPath, sparkContext)
        penalties = []
        logging = ['Episode\tPenalty']

        for ep in range(0, self.episodes):
            totalStates = 0
            penalty = 0
            for i, entry in enumerate(dataset):
                # check reward per cluster according to table
                states = entry[1] # domains only
                actionType = ''
                totalStates += len(states)
                #while not done:
                for j, state in enumerate(states):
                    state = state.split('.')[0]
                    stateIdx = self.rewardIDs.index(state)

                    if(random.uniform(0,1) < self.epsilon):
                        actionType = 'explore'
                        action = random.choice(self.actions)
                        action = self.actions.index(action)
                    else:
                        action = np.argmax(self.QTable[stateIdx])
                        actionType = 'exploit'

                    reward = self.rewardTable[stateIdx, action]

                    # check if last state in the cluster
                    if(j+1 < len(states)):
                        nextState = states[j+1]
                    else:
                        nextState = states[j]
                    nextStateIdx = self.rewardIDs.index(nextState)

                    oldQValue = self.QTable[stateIdx, action]
                    nextMax = np.max(self.QTable[nextStateIdx])

                    newQValue = oldQValue + self.alpha * (reward + self.gamma * nextMax - oldQValue)
                    self.QTable[stateIdx, action] = newQValue

                    if (reward < self.penaltyThreshold):  # better define penalties
                        penalty += 1

            penalties.append(penalty)


        np.save(self.QTablePath, self.QTable)
        np.save(self.rewardTablePath, self.rewardTable)
        Utils.writeFile(self.IDmapPath, '\n'.join(self.rewardIDs))

        self.outputStats() if outputStats else ''

        print('Done training!')



    def test(self, sparkContext, content, labelThreshold, resultpath, similarityDictionary):

        self.rewardTable = np.load(self.rewardTablePath)
        self.rewardIDs = Utils.readFileLines(self.IDmapPath)
        self.QTable = np.load(self.QTablePath)
        domainAnalysisPath = resultpath + '.eval.Qdomains'
        domainAnalysis = {} #'gene\tcandidateCluster\tstateAction\tdomains\n'

        if (not sparkContext):
            sparkContext = SparkContext(conf=self.extractor.initSpark())
        output = []
        outputidx = {}
        geneIdx = 0
        self.geneMap = self.loadGeneMap(sparkContext)

        dataset = []
        for i in content:
            temp = i.split('\t')
            dataset.append([temp[0].split('|'), temp[1]])

        noDomainGenes, chosenPositive, chosenNegative = [], [], []

        for prediction in dataset:
            epochs, penalty, reward = 0, 0, 0
            label = float(prediction[1])
            clustergenes = prediction[0]
            geneIdx += 1
            geneDecisions = [[],[],[]]
            geneWeights = []
            if(label < labelThreshold):
                output.append(('|').join(clustergenes) + '\t0')
            else:
                for gene in clustergenes:
                    states = self.geneMap.get(gene, [''])
                    outputidx[gene] = geneIdx
                    geneReward = [0,0]
                    domains = ''
                    geneWeight = 0
                    for state in states:
                        # if clause for domains not in reward IDs
                        stateIdx = self.rewardIDs.index(state) if state in self.rewardIDs else -1
                        #default action is keep
                        action = np.argmax(self.QTable[stateIdx]) if stateIdx > 0 else 0

                        # if clause for domains not in reward IDs
                        rewardIdx = self.rewardIDs.index(state) if state in self.rewardIDs else -1
                        reward = self.rewardTable[rewardIdx, action] if rewardIdx > 0 else 0
                        # accumulate rewards for keep and skip
                        geneReward[action] += reward
                        if(reward < self.penaltyThreshold):
                            penalty += 1

                        thisstate = state if state else '-'
                        domains += str(thisstate + ', ')
                        geneWeight += self.getWeightfactor(state)

                    ### check if max reward is from 'keep' or 'skip'
                    # among all domains related to one gene.
                    # flags gene if no domain was found for it
                    if(self.useAvAction or self.useDryIslands or self.useNeighborWeight or self.useSimilarityWeight):
                        geneAction = geneReward.index(max(geneReward)) if states[0] else 2 # use 2 to list genes without domains!
                    else:
                        geneAction = geneReward.index(max(geneReward)) if states[0] else 1

                    ############################### geneDecisions
                    geneDecisions[geneAction].append(gene)
                    geneWeights.append(geneWeight)
                    domainAnalysis[gene] = domains[:-2]


                ########################
                # average domain action
                ########################
                if(self.useAvAction):
                    ratioKeep = (len(geneDecisions[0]) * 100) / len(clustergenes)
                    if (geneDecisions[2]):
                        geneDecisions = self.noDomainAverageAction(geneDecisions, ratioKeep, threshold=50)

                ########################
                # dry islands (sequence of 0s)
                ########################
                if(self.useDryIslands):
                    geneDecisions = self.clearDryIslands(geneDecisions, geneWeights, clustergenes, threshold=3)

                ########################
                # annotation weights
                ########################
                if (self.useNeighborWeight):
                    # update geneWeights to contain "1" at positions were neighbor weights were considered
                    geneDecisions, geneWeights = self.adjustByNeighborWeights(geneDecisions, geneWeights, clustergenes, windowSize=1)

                for i, action in enumerate(geneDecisions[:2]):
                    for gene in action:
                        domainAnalysis[gene] = self.actions[i] + '\t'+ domainAnalysis.get(gene)

                if (len(geneDecisions) > 2):
                    # if using combined with other methods, this list also has to be updated:
                    noDomainGenes.append(geneDecisions)

                chosenPositive.append(geneDecisions[0])
                chosenNegative.append(geneDecisions[1])

            epochs +=1

        ########################
        # similar gene action
        ########################
        if (self.useSimilarityWeight):
            chosenPositive, chosenNegative = self.ActionBySimilarity(noDomainGenes, chosenPositive, chosenNegative, similarityDictionary, labelThreshold,
                                                        threshold=10)

        for i in chosenPositive:
            output.append('|'.join(sorted(i)) + '\t' + str(1))
        for i in chosenNegative:
            output.append('|'.join(sorted(i)) + '\t' + str(0))

        outDomainAnalysis = 'gene\tstateAction\tdomains\n'
        for k,v in domainAnalysis.items():
            outDomainAnalysis += k + '\t' + v + '\n'

        Utils.writeFile(domainAnalysisPath, outDomainAnalysis[:-1])
        print('Penalty sum:', str(penalty))

        return output



    def ActionBySimilarity(self, noDomainGenes, chosenPositive, chosenNegative, dictionary, labelThreshold, threshold):
        genesChosenPositive = sum(chosenPositive, [])
        # noDomainGenes has geneDecisions[positives,negatives,nodomains]
        # for each entry that had no domain genes identified
        print('gene\t', '\tmatches', '\tmaxPidents', '\tsimilar decisions sum')
        for entry in noDomainGenes:
            posIdx = chosenPositive.index(entry[0])
            negIdx = chosenNegative.index(entry[1])
            for gene in entry[2]:
                similarityDecisionScore = 0
                matches = {key: value for key, value in dictionary.items() if gene in key}
                matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:threshold]
                for i in matches:
                    similar = i[0].replace(gene,'').replace('|','')
                    if(similar in genesChosenPositive):
                        similarityDecisionScore += 0.1

                print(gene, '\t', len(matches), str(matches[0]), '\t', str(similarityDecisionScore))

                if(similarityDecisionScore >= labelThreshold):
                    chosenPositive[posIdx].append(gene)
                else:
                    chosenNegative[negIdx].append(gene)
        exit()
        return chosenPositive, chosenNegative




    # get Zero weight "dry" islands:
    # given a thresholg of (3?) consecutive 0 weights,
    # change decision from keep to skip (filter it out)
    # ------> does not apply to noDomain genes
    def clearDryIslands(self, geneDecisions, geneWeights, clusterGenes, threshold):
        sum, pre, pos, count = 0, 0, 0, 0
        islands = []
        for i, weight in enumerate(geneWeights):
            if (weight == 0):
                sum += weight
                if (sum == 0):
                    pos = i
                    count += 1
                if (i == len(geneWeights) - 1 and count >= threshold):
                    pre = pos - (count - 1)
                    islands.extend(clusterGenes[pre:pos])
            else:
                if (count >= threshold):
                    pre = pos - (count - 1)
                    islands.extend(clusterGenes[pre:pos])
                count = 0

        for id in islands:
            if(id in geneDecisions[0]):
                geneDecisions[0].remove(id)
            if(id not in geneDecisions[1]):
                if(len(geneDecisions) > 2):
                    if(id not in geneDecisions[2]):
                        geneDecisions[1].append(id)
                else:
                    geneDecisions[1].append(id)

        geneDecisions[0] = sorted(set(geneDecisions[0]))
        geneDecisions[1] = sorted(set(geneDecisions[1]))

        return geneDecisions


    # tags all genes that appear as immediate neighbors of annotated genes
    # independently of their decision (keep or skip)
    # (genes with weights: backbones, TEs, TFs, transporters)
    def adjustByNeighborWeights(self, geneDecisions, geneWeights, clusterGenes, windowSize):
        candidatesSkip = clusterGenes #geneDecisions[1]
        for gene in candidatesSkip:
            idx = clusterGenes.index(gene)
            # window of size 2
            precIdx = max(idx-windowSize, 0)
            postIdx = min(idx+windowSize+1, len(clusterGenes)-1) # shift one index position to be included in array slice
            precedentWeight = sum(geneWeights[precIdx:idx])
            posteriorWeight = sum(geneWeights[idx+1:postIdx])
            #ownWeight = geneWeights[idx]
            # value > 1 means it is an annotation
            if(precedentWeight > 1 or posteriorWeight > 1):
                geneWeights[idx] = 1 if geneWeights[idx] < 1 else geneWeights[idx] # value = 1 means it is a flagged neighbor
                if(gene in geneDecisions[1]):
                    geneDecisions[0].append(gene)
                    geneDecisions[1].remove(gene)
        return geneDecisions, geneWeights


    # recover the keep ratio action in the cluster
    # and assign this action to genes without any domains in cluster
    def noDomainAverageAction(self, geneDecisions, ratioKeep, threshold):
        nodomainAction = 0 if (ratioKeep > threshold) else 1
        geneDecisions[nodomainAction].extend(geneDecisions[2])
        geneDecisions = geneDecisions[:2]

        geneDecisions[0] = sorted(set(geneDecisions[0]))
        geneDecisions[1] = sorted(set(geneDecisions[1]))

        return geneDecisions


    def outputStats(self):
        stat = ['id\trewardKeep\trewardSkip\tqvalueKeep\tqvalueSkip']
        for i, id in enumerate(self.rewardIDs):
            stat.append(id + '\t' + str(self.rewardTable[i][0]) + '\t' + str(self.rewardTable[i][1]) + '\t' + str(self.QTable[i][0])+ '\t' + str(self.QTable[i][1]))        #
        Utils.writeFile(self.outputPath + self.params + '.stat','\n'.join(stat))
        print('Stats saved.')


    def getRewards(self):
        if (not os.path.isfile(self.rewardPath)):
            print('Counting rewards...')
            featType = 'domains_pfam'
            sourceType = 'domains'
            self.extractor.extractRewardPerFeat(self.trainPath, self.rewardPath, featType, sourceType, self.rewardType)
        self.loadReward()


    def loadReward(self):
        print('Loading rewards...')
        self.rewardList = Utils.readFileLines(self.rewardPath)
        self.rewardIDs, self.rewardLabels = self.normalizeRewardIDs()
        if (self.rewardIDs):
            self.rewardTable = np.zeros((len(self.rewardIDs), len(self.actions)))
            self.QTable = np.zeros((len(self.rewardIDs), len(self.actions)))
        else:
            print('Rewards list empty.')
            exit(0)

        #statlabels = ['id\tsum\tlen\tlabels']
        for i, line in enumerate(self.rewardIDs):
            scores = self.rewardLabels[i]
            posSum = sum(i for i in scores if i > 0)
            negSum = sum(i for i in scores if i < 0)

            ############  reward function to be defined here
            #keepreward = (sumscores / nbscores)
            keepreward = posSum / len(self.rewardIDs)
            #skipreward = ((nbscores - sumscores) / nbscores)
            skipreward = abs(negSum / len(self.rewardIDs))
            ############

            if(self.useCompWeights):
                weightFactors = self.getWeightfactor(line)
                if(weightFactors):
                    for factor in weightFactors:
                        keepreward = keepreward * factor # in theory only keeps will be annotated
                        #skipreward = skipreward * factor

            # favors 'keep'.
            # a different way would be keepreward > skipreward, would skip ambiguous domains
            if(keepreward > (skipreward * self.keepskipThreshold)): # it's a keep
                self.rewardTable[i,0] = keepreward # keep action
                self.rewardTable[i, 1] = -keepreward # skip action, much worse error
            else: # it's a skip:
                self.rewardTable[i, 0] = -skipreward # keep action
                self.rewardTable[i, 1] = skipreward # skip action

        print('Done filling reward table.')



    def getWeightfactor(self, domain):
        output = 0
        weightMatches = [item.split('\t')[1] for item in self.weights if domain in item] if domain else ''
        if (weightMatches):
            for type in weightMatches:
                if('backbone' in type):
                    output += 2
                elif('tailor' in type or 'transport' in type or 'transcript'):
                    output += 1.5
        return output


    def loadGeneMap(self, sparkContext):
        content = Parsers.parseDatasetContents(self.geneMapPath, 'domains', 'domains')
        contentRDD = sparkContext.parallelize(content, numSlices=1000)
        genemapRDD = contentRDD.map(lambda x: (x[0][2], x[0][1].split('\n'))).reduceByKey(add)
        genemap = genemapRDD.collectAsMap()

        return genemap


    def normalizeRewardIDs(self):
        outputIDs, outputLabels = [], []
        # handles simple or long domain names
        for i in self.rewardList:
            temp = i.split('|')
            id = temp[0].split('.')[0] if '.' in i else i.split('\t')[0]
            item = temp[1] if len(temp) > 1 else temp[0]
            labels = item.split('\t')[1]
            labels = [int(i) for i in labels.split(',')]
            outputIDs.append(id)
            outputLabels.append(labels)
        return outputIDs, outputLabels

if __name__ == '__main__':
    QLearner().main()
