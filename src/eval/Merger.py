from utils import UtilMethods as Utils


class Merger:

    def __init__(self, config):

        self.successive = int(config.get('eval', 'successive'))
        self.merge = config.getboolean('eval', 'merge')
        self.useMajority = config.getboolean('eval', 'majority')
        self.geneLengthPath = config.get('eval', 'gene.length')
        self.threshold = float(config.get('eval', 'threshold'))

    def getParams(self, resultPath):

        if (self.useMajority):
            resultPath += '.majority'

        if (self.merge):
            resultPath += '.merge'
        if (self.successive > 0):
            resultPath += '.succ' + str(self.successive)

        return resultPath

    def doMerges(self, predictions):
        if (self.successive > 0):
            predictions = self.computeSuccessiveMerge(predictions)

        if (self.useMajority):
            predictions = self.computeMajorityVote(predictions, 50, 10000)


        return predictions


    def computeMajorityVote(self, predictedClusters, thresholdPerc, thresholdLen):
        predictPerGene, majority, geneLengths = {}, {}, {}
        majorityClusters = []
        threshold = thresholdPerc / 100
        lengths = Utils.readFileLines(self.geneLengthPath)

        for i in lengths:
            temp = i.split('\t')
            geneLengths[temp[0].split('.')[0]] = int(temp[1])

        # get number of times gene appears, and with which label
        for cluster in predictedClusters:
            clustergenes = cluster.replace('||','|').replace('|\t','\t').split('\t')
            label = 1 if float(clustergenes[1]) >= self.threshold else 0
            genes = clustergenes[0].split('|')
            for gene in genes:
                gene = gene.split('.')[0]
                if(gene in predictPerGene.keys()):
                    predictPerGene[gene].append(label)
                else:
                    if(gene):
                        predictPerGene[gene] = [label]

        # compute score for each gene regarding the number of times it appears
        for gene,label in predictPerGene.items():
            majority[gene] = sum(label) / len(label)

        # concatenate genes scoring above threshold
        tempcluster, tempscore, templength = '', 0, 0
        for gene, score in majority.items():
            if(score >= threshold):
                if(tempscore >= threshold ):
                    tempcluster += gene + '|'
                    templength += geneLengths.get(gene)
                else:
                    majorityClusters.append(tempcluster[:-1] + '\t' + '0')
                    tempcluster = gene + '|'
                    templength = geneLengths.get(gene)
            else:
                if(tempscore < threshold):
                    # concatenate until around threshold length for negatives
                    if(templength < thresholdLen):
                        tempcluster += gene + '|'
                        templength += geneLengths.get(gene)
                    else:
                        majorityClusters.append(tempcluster[:-1] + '\t' + '0')
                        tempcluster = gene + '|'
                        templength = geneLengths.get(gene)
                else:
                    majorityClusters.append(tempcluster[:-1] + '\t' + '1')
                    tempcluster = gene + '|'
                    templength = geneLengths.get(gene)

            tempscore = score

        print('Done computing majority vote.')
        return majorityClusters


    def succMerge(self, predictions):
        results = []
        succ = self.successive
        for item in predictions:
            id = item.split('\t')[0]
            confidence = float(item.split('\t')[1])
            last = results[-1] if results else ""
            lastConfidence = float(last.split('\t')[1]) if last else 0

            if(lastConfidence >= self.threshold):
                # finding two positives consecutively
                if(confidence >= self.threshold):
                    results.append(item)
                    # restart successive count
                    succ = self.successive
                else:
                    if(succ > 0):
                        if(self.merge):
                            results[-1] = last.split('\t')[0] + '|' + id + '\t' + str(lastConfidence)
                        else:
                            results.append(id + '\t' + str(lastConfidence))
                        succ -= 1
                    else:
                        results.append(item)
            else:
                results.append(item)

        return results



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

                    # remove duplicated genes
                    successorGenes = ''
                    for gene in successorItem.split('|'):
                        if(gene not in mergeID):
                            successorGenes += gene + '|'

                    successorItem = successorGenes[:-1]

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



