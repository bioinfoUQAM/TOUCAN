from utils import UtilMethods as Utils
import os


###############
# Manages evaluation step:
# checks outputted predictions
# against file of gold clusters
###############


class GeneGoldProcess:

    def __init__(self):
        self.config = Utils.loadConfig()
        self.task = self.config.get('eval', 'task')
        self.gold = self.config.get('eval', 'goldID.path')
        self.result = self.config.get('eval', 'result.path')
        self.threshold = float(self.config.get('eval', 'threshold'))
        self.successive = int(self.config.get('eval', 'successive'))
        self.merge = self.config.getboolean('eval', 'merge')
        self.similarityPath = self.config.get('eval', 'similarity.path')
        self.useSimilarity = self.config.getboolean('eval', 'similarity')
        self.goldIDs = Utils.readFileLines(self.gold)[1:]
        self.resultFiles = Utils.listFilesExt(self.result, 'IDs.test')
        # load similarity as dictionary[id1|id2] = pident
        self.similarityDic = self.loadSimilarityDic()
        # total nb of gold genes
        self.nbGoldGenes = len(self.goldIDs)
        # total nb of gold clusters
        self.foldedGold = self.foldGoldData(self.goldIDs)
        self.nbGoldClusters = len(self.foldedGold)
        self.outputheader = 'goldClusterID\tgoldGeneID\tpredictedClusterLabel\tpredictedClusterID\n'
        self.scoreheader = 'goldClusterID\tpredictedClusterID\tclusterScore\n'


    def main(self):
        if('eval') in self.task:
            self.evaluate()
        elif('summ') in self.task:
            self.summarize()


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


    def evaluate(self):
        output = self.outputheader
        for file in self.resultFiles:
            clusterScores = {}
            resultPath = os.path.abspath(file)

            if (self.useSimilarity):
                resultPath += '.similar'
            if (self.merge):
                resultPath += '.merge'
            if (self.successive > 0):
                resultPath += '.succ' + str(self.successive)
            outputPath = resultPath + '.eval.metrics'
            scorePath = resultPath + '.eval.score'

            if (not os.path.isfile(outputPath)):
                predictions = Utils.readFileLines(file)
                predictionsSorted = sorted(predictions)

                if (self.successive > 0):
                    predictionsSorted = self.computeSuccessiveMerge(predictionsSorted)

                # get total nb of positive predictions
                posPredClusters, posPredGenes = self.unfoldResultData(predictionsSorted)
                posPredGenes = sorted(posPredGenes)

                # nb matches positive predicted and gold genes
                geneMatches = 0
                clustersFound = set()
                genesFound = []

                # for entry in self.goldIDs:
                for goldID, genes in self.foldedGold.items():
                    outerClusterScore, clusterScore = 0,0
                    indexInResults = set()
                    label, predLabel = 0, 0
                    predIDs, outerPredLine = "",""
                    for gene in genes:
                        # get all predicted clusters matching this gene from results
                        found = [i for i, s in enumerate(predictionsSorted) if gene in s]
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

                            if (gene in posPredGenes):
                                geneMatches += 1
                                if(outerPredLine is predLine):
                                    clusterScore += 1
                                else:
                                    outerClusterScore = clusterScore
                                    clusterScore = 1
                                    outerPredLine = predLine
                                genesFound.append(gene)
                                label = 1
                                clustersFound.add(goldID)

                        output += goldID + '\t' + gene + '\t' + str(label) + '\t' + predClusterID + '\n'
                        label = 0

                    clusterScore = max(outerClusterScore, clusterScore)

                    if (self.useSimilarity and predLabel > self.threshold):
                        similarity = self.computePredictionSimilarity(genes, predIDs, genesFound)
                        clusterScore += similarity
                        geneMatches += similarity

                    clusterScore = clusterScore / len(genes)
                    # just to rank clusters for score comparison of top K
                    clusterScores[goldID] = [predClusterID, clusterScore, "|".join(predIDs)]

                clusterMatches = len(clustersFound)
                # geneMatches = len(genesFound)

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


    def computePredictionSimilarity(self, goldGenes, predictedGenes, genesFound):
        pairs = {}
        # filter already found genes from similarity comparison
        goldGenes = [gene for gene in goldGenes if gene not in genesFound]
        predictedGenes = [gene for gene in predictedGenes if gene not in genesFound]
        similarityScore = 0.0

        # if there are still genes left
        if(len(predictedGenes) > 0 and len(goldGenes) > 0):
            for i in goldGenes:
                score = 0
                for j in predictedGenes:
                    pair = i + "|" + j
                    # retrieve similarity for each [goldGene | predictedGene] pair
                    dictscore = self.similarityDic.get(pair)
                    # adjust similarity score to %
                    dictscore = float(dictscore)/100 if dictscore is not None else 0.0
                    # keep the highest score for gold gene
                    if(dictscore > score):
                        pairs[i] = [j, dictscore]
                        score = dictscore
            # sort pairs in descending order
            pairs = sorted(pairs.values(), key = lambda x: x[1], reverse=True)
            similarPredictedGenes = set()
            # add similarity of highest score for predicted gene
            for pair in pairs:
                if(pair[0] not in similarPredictedGenes):
                    similarityScore += pair[1]
                    similarPredictedGenes.add(pair[0])

        return similarityScore


    def loadSimilarityDic(self):
        similaritydic = {}
        similarity = ""
        if("eval" in self.task and self.useSimilarity):
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


    def computeSuccessiveMerge(self, predictions):
        resultIterator = enumerate(predictions, start=0)
        results = []

        for i, item in resultIterator:
            id = item.split('\t')[0]
            confidence = float(item.split('\t')[1])
            count = 1
            if(confidence >= self.threshold and i < len(predictions)-self.successive):
                mergeID = id + '|'

                # reach X many items ahead for the merge
                while count <= self.successive:
                    successorItem = predictions[i + count]
                    successorConf = float(successorItem.split('\t')[1])

                    # perform evaluation merging potential clusters or not
                    if(self.merge):
                        mergeID += successorItem.rsplit('\t')[0] + '|'
                    else:
                        if(successorConf < self.threshold):
                            upItem = successorItem.rsplit('\t')[0] + '\t' + str(confidence)
                            predictions[i + count] = upItem

                    count += 1
                    resultIterator.__next__()

                if(self.merge):
                    upItem = mergeID[:-1] + '\t' + str(confidence)
                    results.append(upItem)
            else:
                results.append(item)
        if(self.merge):
            predictions = results

        return predictions


    def outputMetrics(self, matches, nbPosPredicted, nbGold):
        precision, recall, fmeasure = self.computeMetrics(matches, nbPosPredicted, nbGold)
        metrics =        str(precision) + '\t' + str(recall)+ '\t' + str(fmeasure)
        return metrics


    # computes precision, recall, fmeasure
    def computeMetrics (self, matches, nbPosPredicted, nbGold):
        try:
            precision = matches / nbPosPredicted
            recall = matches / nbGold

            # TODO: double check this condition
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


    def foldGoldData(self, list):
        unique = {}
        for entry in list:
            id = entry.split('\t')[0]
            gene = entry.split('\t')[1]
            if(unique.get(id) is None):
                unique[id] = [gene]
            else:
                unique[id].append(gene)
        return unique


    def unfoldResultData(self, list):
        unique = set()
        pos = [x for x in list if float(x.split('\t')[1]) >= self.threshold]
        for entry in pos:
            genes = entry.split('|')
            for gene in genes:
                gene = gene.split('\t')[0]
                unique.add(gene)

        return pos, unique


if __name__ == '__main__':
    GeneGoldProcess().main()






#     # get all matching genes from gold data
#     indexGeneGold = [i for i, elem in enumerate(self.goldIDs) if gene in elem]
#     # get all matching genes from results
#     indexGeneResults = [i for i, elem in enumerate(resultsSort) if gene in elem]
#
#
#     if (len(indexGeneResults) > 0):
#         # for all matching genes
#         for i in range(len(indexGeneResults)):
#             # do not add them more times than they appear in gold data
#             if(i < len(indexGeneGold)):
#                 # retrieve index of gene in results
#                 resultPos = indexGeneResults[i]
#                 # retrieve entire result line for given gene
#                 line = resultsSort[resultPos]
#                 # split line according to {predictedClusterGeneIDs, confidence value}
#                 lineSplit = line.split('\t')
#                 try:
#                     predLabel = float(lineSplit[1])
#                     predLabel = 1 if predLabel >= self.threshold else 0
#                     #geneMatches += int(predLabel)
#
#                     # if at least one gene is found, consider cluster found
#                     if(predLabel > 0):
#                         genesFound.append(gene)
#                         geneMatches += 1
#                         clustersFound.add(goldID)
#                 except ValueError:
#                     predLabel = (lineSplit[1])
#
#                 predIDs = lineSplit[0].split('|')
#                 predDlusterID = 'Cluster' + predIDs[0] + '_' + predIDs[-1]
#
#                 output += goldID + '\t' + gene + '\t' + str(predLabel) + '\t' + predDlusterID  +  '\n'
#                 i += 1
#