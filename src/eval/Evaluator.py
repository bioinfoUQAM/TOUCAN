from utils import UtilMethods as Utils
from collections import Counter
import os
from pyspark import SparkContext
from eval import Filter, Merger, Similarity


###############
# Manages evaluation step:
# checks outputted predictions
# against file of gold clusters
###############
from utils.UtilMethods import unfoldResultData


class GeneGoldProcess:

    def __init__(self):
        self.config = Utils.loadConfig()
        self.task = self.config.get('eval', 'task')
        self.gold = self.config.get('eval', 'goldID.path')
        self.result = self.config.get('eval', 'result.path')
        self.threshold = float(self.config.get('eval', 'threshold'))
        self.sparkContext = SparkContext(conf=Utils.getSparkConf('filter'))

        self.Similarity = Similarity.Similarity(self.config)
        self.Filter = Filter.Filter(self.config, sparkContext=self.sparkContext)
        self.Merger = Merger.Merger(self.config)

        self.goldIDs = Utils.readFileLines(self.gold)[1:]
        self.resultFiles = Utils.listFilesExt(self.result, 'IDs.test')

        # total nb of gold genes
        self.nbGoldGenes = len(self.goldIDs)
        # total nb of gold clusters
        self.foldedGold = Utils.foldClusterData(self.goldIDs, 'gold', 0)
        self.goldGenes = [gene for genes in self.foldedGold.values() for gene in genes]
        self.nbGoldClusters = len(self.foldedGold)
        self.outputheader = 'goldClusterID\tgoldGeneID\tpredictedClusterLabel\tpredictedClusterID\n'
        self.scoreheader = 'goldClusterID\tpredictedClusterID\tclusterScore\n'


    def main(self):
        if('eval') in self.task:
            self.evaluate()
        elif('summ') in self.task:
            self.summarize()


    def getParams(self, resultPath):
        resultPath = self.Filter.getParams(resultPath)
        resultPath = self.Similarity.getParams(resultPath)
        resultPath = self.Merger.getParams(resultPath)
        return  resultPath


    def evaluate(self):
        output = self.outputheader

        for file in self.resultFiles:
            clusterScores = {}
            resultPath = os.path.abspath(file)

            resultPath = self.getParams(resultPath)

            outputPath = resultPath + '.eval.metrics'
            scorePath = resultPath + '.eval.score'
            overestPath = resultPath + '.eval.over'

            if (not os.path.isfile(outputPath)):
                predictions = Utils.readFileLines(file)
                predictionsSorted = sorted(predictions)

                posOrigClusters, posOrigGenes = unfoldResultData(self.threshold, predictionsSorted)

                # performs successive, merge, majority vote
                predictionsSorted = self.Merger.doMerges(predictionsSorted)

                # performs annotation filter or Q-learning filter
                predictionsSorted = self.Filter.doFilter(predictionsSorted, resultPath, self.Similarity.dictionary)

                # get total nb of positive predictions
                posPredClusters, posPredGenes = unfoldResultData(self.threshold, predictionsSorted)
                posPredGenes = sorted(posPredGenes)

                overestimatedMetrics = self.computeOverestimateError(posOrigClusters, posOrigGenes, posPredClusters, posPredGenes) #if self.merge else 0
                Utils.writeFile(overestPath, overestimatedMetrics)

                # nb matches positive predicted and gold genes
                clustersFound = set()
                genesFound = []
                geneSimilarities = 0

                # for entry in self.goldIDs:
                for goldID, goldGenes in self.foldedGold.items():
                    outerClusterScore, clusterScore, geneScore = 0, 0, 0
                    label, predLabel = 0, 0
                    predIDs, outerPredLine = "",""
                    posClusterMatch = []
                    posGeneMatch = []
                    for goldGene in goldGenes:
                        # get all predicted clusters matching this gene from results
                        found = [i for i, s in enumerate(predictionsSorted) if goldGene in s]
                        # retrieve predicted line
                        if (not found):
                            found = [-1]
                            predLine, predClusterID = "not found", "not found"
                        else:
                            predLine = predictionsSorted[found[0]]
                            predLabel = float(predLine.split("\t")[1])
                            predIDs = predLine.split("\t")[0].split("|")
                            predClusterID = 'Cluster' + predIDs[0] + '_' + predIDs[-1]
                            if(not outerPredLine):
                                outerPredLine = predLine

                            if (goldGene in posPredGenes):
                                posClusterMatch.append(predClusterID)
                                posGeneMatch.append(predIDs)
                                geneScore += 1
                                if(outerPredLine is predLine):
                                    clusterScore += 1
                                else:
                                    outerClusterScore = clusterScore
                                    clusterScore = 1
                                    outerPredLine = predLine
                                genesFound.append(goldGene)
                                label = 1
                                # to change minimum nb of gold genes to consider a found cluster
                                #if(clusterScore > 1):
                                clustersFound.add(goldID)

                        output += goldID + '\t' + goldGene + '\t' + str(label) + '\t' + predClusterID + '\n'
                        label = 0

                    clusterScore = max(outerClusterScore, clusterScore)

                    if (self.Similarity.useSimilarity and predLabel > self.threshold):
                        similarity = self.Similarity.computePredictionSimilarity(goldGenes, predIDs, genesFound)
                        #print("cluster score: ", clusterScore, "\tgene score: ", geneScore, "\tcluster len: ", len(goldGenes), "\tsimilarity: ", similarity)
                        clusterScore += similarity
                        geneSimilarities += similarity

                    clusterScore = clusterScore / len(goldGenes)
                    # get match with most positive gene matches to rank clusters for score comparison of top K
                    matchedCluster = max(set(posClusterMatch), key=posClusterMatch.count) if posClusterMatch else predClusterID
                    matchedGenes = posGeneMatch[posClusterMatch.index(matchedCluster)] if posGeneMatch else predIDs
                    clusterScores[goldID] = [matchedCluster, clusterScore, "|".join(matchedGenes)]

                clusterMatches = len(clustersFound)
                geneMatches = len(genesFound) + geneSimilarities

                clusterScores = sorted(clusterScores.items(), key=lambda x: x[1], reverse=True)
                clusterScores = [i[0] + '\t' + i[1][0] + '\t' + str(i[1][1])  + '\t' + i[1][2] for i in clusterScores]
                clusterScores = "\n".join(clusterScores)

                metrics = 'Metrics - genes\t\t\tMetrics - clusters\n\tP\tR\tF\t\tP\tR\tF\t\npos\t' + \
                          self.outputMetrics(geneMatches, len(posPredGenes), self.nbGoldGenes) + '\t\t' + \
                          self.outputMetrics(clusterMatches, len(posPredClusters), self.nbGoldClusters) + '\n\n\n'

                Utils.writeFile(scorePath, self.scoreheader + '\n' + clusterScores)
                Utils.writeFile(outputPath, metrics + output)
                metrics = ""
                output = self.outputheader

        print('Done evaluation!')




    def computeOverestimateError(self, posOrigClusters, posOrigGenes, posPredClusters, posPredGenes):
        output = 'cluster IDs\tcluster genes\tgold gene matches (top 2)\tlen preprocessed\tlen postprocessed\t% overestimated orig genes\t% overestimated postprocessed genes\t% total overestimated genes'
        # consider all genes between FIRST gold and LAST gold gene
        test = [cluster for x in self.foldedGold.values() for cluster in x]
        genesInGoldRegions = {}
        for key, value in self.foldedGold.items():
            org_prefix = value[0].split('_')[0]
            first = int(value[0].split('_')[1])
            last = int(value[len(value)-1].split('_')[1])+1
            thisrange = range(first, last)
            # if cluster is too spread, tag only existing genes and not entire region
            # so as to not overwrite smaller regions within
            if(len(thisrange) > 100):
                thisrange = value
            for i in thisrange:
                temp = i if org_prefix in str(i) else org_prefix + '_' + "{:05d}".format(i)
                if(temp in genesInGoldRegions):
                    genesInGoldRegions[temp].append(key)
                else:
                    genesInGoldRegions[temp] = [key]

        percAllOriginal, percAllMerge, percAllTotal = [], [], []

        for cluster in posPredClusters:
            IDs = cluster.split('|')
            clusterID = 'Cluster' + IDs[0] + '_' + IDs[-1].split('\t')[0]
            preprocessed = ''
            # match current post-processed cluster with an original prediction according to max nb of gene matches.
            # use only first half of IDs to match start of candidate clusters that where merged.
            # orig average could VARY depending on post-processing because of IDs in current post-processed cluster
            matchRatio = self.Merger.successive if self.Merger.successive > 0 and len(IDs) > self.Merger.successive else 1
            preprocessed = self.findBestMatch(IDs[:len(IDs)//matchRatio], posOrigClusters)

            prelen = len(preprocessed.split('|'))
            poslen = len(IDs)

            # retrieve gold genes from best gold matches
            goldMatchList = [goldcluster for gene, goldcluster in genesInGoldRegions.items() if gene in cluster]
            # cluster match if at least one gene in gold regions
            # otherwise considered false positive
            if(goldMatchList):
                goldMatchesFlat = [item for clusters in goldMatchList for item in clusters]
                goldMatches = Counter(goldMatchesFlat).most_common(2)
                goldBestMatches = [ ('|').join(self.foldedGold.get(goldMatches[i][0])) for i in range(len(goldMatches))]

                fromGold = [gene for gene in genesInGoldRegions.keys() if gene in cluster]
                # check if also in current cluster to account for filtering
                fromOrig = [gene for gene in posOrigGenes if gene in preprocessed and gene in cluster]
                fromMerge = [gene for gene in posPredGenes if (gene in cluster and gene not in preprocessed)]

                origGenesinGold = list(set(fromGold) & set(fromOrig))
                origGenesnotGold = len(fromOrig) - len(origGenesinGold)
                mergeGenesinGold = list(set(fromGold) & set(fromMerge))
                mergeGenesnotGold = len(fromMerge) - len(mergeGenesinGold)

                percOverOriginal = (origGenesnotGold * 100) / len(fromOrig)if fromOrig else 0.0
                percAllOriginal.append(percOverOriginal)
                percOverMerge = (mergeGenesnotGold * 100) / len(fromMerge) if fromMerge else 0.0
                percAllMerge.append(percOverMerge)
                percTotal =  ((origGenesnotGold + mergeGenesnotGold) * 100) / len(IDs)
                percAllTotal.append(percTotal)

                output += '\n' + clusterID + '\t' + cluster.split('\t')[0] + '\t' + (', ').join(goldBestMatches) + '\t' + str(prelen) + '\t' + str(poslen) + '\t' + '{:0.2f}'.format(percOverOriginal) + '\t' + '{:0.2f}'.format(percOverMerge) + '\t' + '{:0.2f}'.format(percTotal)

        meanOrig = sum(percAllOriginal) / len(percAllOriginal)
        meanMerge = sum(percAllMerge) / len(percAllMerge)
        meanTotal = sum(percAllTotal) / len(percAllTotal)

        header = 'average % overestimated orig genes\taverage % overestimated postprocessed genes\taverage % overestimated total genes\n' + '{:0.2f}'.format(meanOrig) + '\t' + '{:0.2f}'.format(meanMerge) + '\t' + '{:0.2f}'.format(meanTotal) + '\n'
        output = header + output

        print('Done computing overestimating error.')

        return output



    def findBestMatch(self, target, clusters):
        idx, max = 0, 0
        for i, cluster in enumerate(clusters):
            # if first cluster is exactly the same,
            # use that as best match
            start = cluster.split('|')[0]
            if(start in target[0]):
                idx = i
                break
            # otherwise count max gene matches
            # to find best match
            else:
                occ = sum(1 for ID in target if ID in cluster)
                if(occ > max):
                    max = occ
                    idx = i

        return clusters[idx] if idx > -1 else ""




    def outputMetrics(self, matches, nbPosPredicted, nbGold):
        precision, recall, fmeasure = self.computeMetrics(matches, nbPosPredicted, nbGold)
        metrics =        str(precision) + '\t' + str(recall)+ '\t' + str(fmeasure)
        return metrics


    # computes precision, recall, fmeasure
    def computeMetrics (self, matches, nbPosPredicted, nbGold):
        try:
            precision = matches / nbPosPredicted
            recall = matches / nbGold

            # when multiple gold clusters are found in single test cluster
            precision = 1.0 if precision > 1 else precision
            recall = 1.0 if recall > 1 else recall

            fmeasure = 2 * ((precision * recall) / (precision + recall))
            fmeasure = 1.0 if fmeasure > 1 else fmeasure

            # round metrics to 3 decimals
            precision = round(precision, 3)
            recall = round(recall, 3)
            fmeasure = round(fmeasure, 3)

        except ZeroDivisionError:
             precision, recall, fmeasure = 'N/A', 'N/A', 'N/A'

        return precision, recall, fmeasure




    def summarize(self):
        metricFiles = Utils.listFilesExt(self.result, 'metrics')
        metricFiles = sorted(metricFiles)
        output, pos = "", ""
        outputFile = Utils.normalizePath(self.result) + "results.summary"
        if("pos" in self.result):
            pos = self.result.split("pos")[1][0:2]

        for file in metricFiles:
            metrics = Utils.readFileLines(file)[2].replace("pos\t","")
            filename = os.path.basename(file)
            classifier = filename.split("_")[0]
            feats = filename.split("_")[1] + "+" + filename.split("_")[2]
            len = filename.split("len")[1].split("_")[0]
            overlap = filename.split("overlap")[1].split("_")[0][0:2]

            evaltype = filename.split("IDs.test.")[1].replace("eval.metrics", "").replace(".","")
            if(not evaltype):
                evaltype = "succ0"
            if("similar" in evaltype):
                evaltype = evaltype.replace("similar", "sim")
            if("merge" in evaltype):
                evaltype = evaltype.replace("succ","")


            line = feats + "\t" + classifier + "\t" + pos + "\t" + len + "\t" + overlap + "\t" + evaltype + "\t" + metrics + "\n"
            output += line
        Utils.writeFile(outputFile, output)

if __name__ == '__main__':
    GeneGoldProcess().main()


