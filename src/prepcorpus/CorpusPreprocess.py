import os                   # for IO operations
from utils import UtilMethods as Utils
from utils import Parsers
import random
from shutil import copy
from pipedata import DataPipeline
from pyspark import SparkContext
import pandas
from io import StringIO

###############
# Processes corpus files.
# Corpus negative instances in a given directory
# must be named "*negative*.fasta"; other *.fasta
# in directory are considered positive.
###############

class CorpusPreprocess:

    def __init__(self):
        self.config = Utils.loadConfig()
        self.corpus_path = self.config.get('corpusPrep', 'corpus.home')
        self.source_path = self.config.get('corpusPrep', 'source.path')
        self.result_path = self.config.get('corpusPrep', 'result.path')
        self.task = self.config.get('corpusPrep', 'task')
        self.ext = self.config.get('corpusPrep', 'source.ext')
        self.seqType = self.config.get('corpusPrep', 'source.type')
        self.posPath = self.config.get('corpusPrep', 'pos.path')
        self.negPath = self.config.get('corpusPrep', 'neg.path')
        self.validPerc = self.config.get('corpusPrep', 'valid.perc')
        self.posPerc = self.config.getint('corpusPrep', 'pos.perc')
        self.targetIds = self.config.get('corpusPrep', 'pos.ids')
        self.filterIds = self.config.get('corpusPrep', 'pos.filters')
        self.length = self.config.getint('corpusPrep', 'cluster.length')
        self.windowOverlap = self.config.getint('corpusPrep', 'window.overlap')
        self.clusterLen = self.config.getint('corpusPrep', 'cluster.length')


    def main(self):
        if('shuffle' not in self.task and 'selectvalid' not in self.task):
            self.result_path = Utils.normalizePath(self.result_path)
            os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

        if('split' in self.task):
            self.splitAsClusters()

        if('shuffle' in self.task):
            posPerc = self.config.get('corpusPrep', 'pos.perc')
            posPerc = int(posPerc) if float(posPerc).is_integer() else float(posPerc)
    
            if(self.result_path.endswith('/')):
                self.result_path = self.result_path[:-1]
            self.result_path = self.result_path + '_pos' + str(posPerc) + '/'
    
            if(not os.path.isdir(self.result_path)):
                os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
                self.createNegShuffle(posPerc)
            else:
                print('Result path already exists.')
    
        if('createdataset' in self.task):
            self.createDataset()
    
        if('domain' in self.task):
            self.createDomainDataset()

        if('goterms' in self.task):
            self.createGoDataset()

        if ('similarity' in self.task):
            self.createSimilarityMatrix()

        if('pfamtsv' in self.task):
            self.createPfamTsv()


    #################################
    # Generates a matrix of BLAST similarities of all vs. all
    # (genes) "*.fasta" files in a given directory
    #################################
    def createSimilarityMatrix(self):
        source_type = self.config.get('dataPipeline', 'source.type')
        list = Utils.listFilesExt(self.source_path, "fasta")
        outputFile = self.result_path + '/similarity.blast'
        outputRedFile = self.result_path + '/similarity.blast.similarity'
        similarity = ""
        columns = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart',
                   'send', 'evalue', 'bitscore', 'qcovs']

        if(not os.path.isfile(outputFile)):

            # generate all gene pairs within a genome
            allpairs = {(i,j) for i in list for j in list}
            # filter out duplicate pairs, e.g. (2,8) and (8,2)
            file_content = set(tuple(sorted(p)) for p in allpairs)

            datapipe = DataPipeline.DataPipeline(source_type=source_type, source_path=self.source_path, result_path=self.result_path)

            sparkContext = SparkContext(conf=datapipe.initSpark("blastSimilarity"))
            similarity = datapipe.getBLAST(file_content, sparkContext, blastTask="similarity")

            result = ""
            for entry in similarity:
                if(entry[1]):
                    result += entry[1] + "\n"

            Utils.writeFile(outputFile, result)
            df = pandas.read_csv(StringIO(result), sep='\t', names=columns, index_col=False)

        else:
            df = pandas.read_csv(outputFile, sep='\t', names=columns, index_col=False)

        # generate leaner matrix with only selected columns,
        # output to new file
        if(not os.path.isfile(outputRedFile)):
            df = df[['qseqid', 'sseqid', 'pident', 'bitscore', 'qcovs']]
            df['id'] = df[['qseqid', 'sseqid']].agg('|'.join, axis=1)
            df.drop('qseqid', 1)
            df.drop('sseqid', 1)
            df = df[['id', 'pident', 'bitscore', 'qcovs']]
            df = df.sort_values('id')
            df.to_csv(sep='\t', header=True, path_or_buf=outputRedFile, index=False)

        print('done!')


    #################################
    # Generates "*.go" files from # "*.fasta"
    # files to use GO term features
    #################################
    def createGoDataset(self):
        source_type = self.config.get('dataPipeline', 'source.type')
        blastPath = self.config.get('blaster','blastdb.path')
        blastPath = Utils.normalizePath(blastPath)
        blastName = self.config.get('blaster','blastdb.name')
        blastMapping = blastPath + blastName + '.tab'

        datapipe = DataPipeline.DataPipeline(source_type=source_type, source_path=self.source_path, result_path=self.result_path)
        list, file_content = Parsers.parseFastaToList(self.source_path, "")
        file_content = [content for content in file_content if not os.path.isfile(self.result_path + os.path.basename(content[0]).replace('.fasta', '.go'))]

        sparkContext = SparkContext(conf=datapipe.initSpark("goDataset"))
        goterms = datapipe.getBLAST(file_content, sparkContext, blastTask="goTerms")

        count = 0
        notFound = 0
        for file, content in goterms.items():

            length = content.split('\n')
            if (len(length) == 2 and not str(length[1])):
                notFound += 1
            else:
                filename = os.path.basename(file)
                resultFile = self.result_path + filename
                resultFile = resultFile.replace('.fasta', '.go')
                Utils.writeFile(resultFile, content)
                count += 1

        print('Done generating', str(count), 'GO term files. \nNo GO terms found for', str(notFound), 'files.')


    #################################
    # Generates "*.domains" files from # "*.fasta"
    # files to use Pfam domain features
    #################################
    def createDomainDataset(self):
        useID  = True
        files = Utils.listFilesExt(self.source_path, self.ext)
        files = [fileName for fileName in files if not os.path.isfile(self.result_path + os.path.basename(fileName).replace('.fasta', '.domains'))]

        source_type = self.config.get('dataPipeline', 'source.type')
        count = 0
        countNone = 0
        datapipe = DataPipeline.DataPipeline(source_type=source_type, source_path=self.source_path, result_path=self.result_path)

        sparkContext = SparkContext(conf=datapipe.initSpark("domainDataset"))
        pfamDomains = datapipe.getDomains(sparkContext)
    
        for file in files:
            fileName = os.path.basename(file)
            IDs = open(file, 'r').readline()
    
            resultFile = self.result_path + fileName.replace('.fasta','.domains')
            result = pfamDomains.get(file)
    
            #if(len(result) > 1):
            if (result is not None):
                result = result.split('\n')
                outF = open(resultFile, 'w')
                outF.write(IDs)
                output = ""
    
                for line in result:
                    if(len(line.strip()) > 1):
                        items = line.split('\t')
                        domainID = items[5]
                        domain = items[6]
                        bitscore = items[11]
                        if(useID):
                            domain = domainID +'|'+ domain
    
                        outF.write(domain+'\n')
                        output += domain + '\n'
    
                outF.close()
                count += 1
            else:
                print('None for file: ', file)
                countNone += 1
    
        print('Done generating', str(count), 'domain files. \nNo domain found for', str(countNone), 'files.' )
    
    #################################
    # shuffles content of negative sequences
    # (order of nucleotides or aminoacids)
    #################################
    def createNegShuffle(self, posPerc):
        files = Utils.listFilesExt(self.source_path, self.ext)
        negPerc = 100 - posPerc
        positives = len(files)
        negativeSize = int((negPerc * positives) / posPerc)
        print(  'Negative percentage: ' + str(negPerc) + '% \n'
              + 'Negative instances: '  + str(negativeSize) + '\n'
              + 'Positive percentage: ' + str(posPerc) + '% \n'
              + 'Positive instances: '  + str(positives) + '\n'
              + 'Total corpus size: '   + str(negativeSize + positives))
    
        thisDecRatio = 0.0
        count = 0
        ratio = (negativeSize / positives)
        decRatio = ratio - int(ratio)
    
        print('Generating...')
        for file in files:
            # add up the decimal ratio part
            thisDecRatio += round(decRatio,2)
            # reset range
            ratioRange = int(negativeSize / positives)

            # check if decimal ratio added up to a duplicate
            if(thisDecRatio >= 1):
                ratioRange = int(ratio + thisDecRatio)
                thisDecRatio = 0

            for i in range(0, ratioRange):
                name = os.path.basename(file)
                result_file = name.split('.')[0] + '_' + str(i) + '.shuffled.negative.fasta'

                if('nuc' in self.seqType):
                    content = Parsers.genBankToNucleotide(file)
                if('amino' in self.seqType):
                    list, content = Parsers.genBankToAminoacid(file)
                content = Utils.charGramShuffle(content, 2)
                content = '>'+ name + '\n' + content

                count +=1

                Utils.writeFile(self.result_path + result_file, content)
    
        print('Total generated: ' + str(count) + '. Done!')
    
    
    #################################
    # Splits randomly "*.fasta" files from given
    # directory into train and validation
    #################################
    def createDataset(self):
        neg_path = Utils.normalizePath(self.negPath)
        pos_path = Utils.normalizePath(self.posPath)
        negatives = Utils.listFilesExt(neg_path, self.ext)
        positives = Utils.listFilesExt(pos_path, self.ext)
        subject = Utils.normalizePath(self.result_path)

        negLen = len(negatives)
        posLen = len(positives)
        negPerc = 100 - self.posPerc
        negTotal = (posLen * negPerc) / self.posPerc
    
        if (negLen < negTotal):
            print("Not enough negative instances. Try another %")
            exit()
        else:
            if(not negatives or not positives):
                print('List of files was empty. '
                      'Please check \'neg.path\' and \'pos.path\' in the self.config file.')

            subject += 'pos' + str(self.posPerc)

            if(len(subject) > 1):
                os.makedirs(subject, exist_ok=True)
                destTrain = subject + '/train/'
                destValid = subject + '/validation/'
    
                if(os.path.exists(destTrain) or os.path.exists(destValid)):
                    print('Dataset already splitted for train and validation. '
                      '\nRename '+ destTrain + ' or ' + destValid + ' and try again.')
                else:
                    os.makedirs(destTrain, exist_ok=False)
                    os.makedirs(destValid, exist_ok=False)
    
                    perc = int(self.validPerc) / 100
    
                    readme = 'Source negative: ' + neg_path + \
                    '\nSource positive: ' + pos_path + \
                    '\n# negative files: ' + str(negLen) + \
                    '\n(final) # negative files: ' + str(negTotal) + \
                    '\n# positive files: ' + str(posLen) + \
                    '\nValidation data percentage (from total): ' + str(self.validPerc) + '%'
    
                    Utils.writeFile(subject + '/README.md',readme)
    
                    # select validation files
                    validNegatives = random.sample(negatives, int(perc*negTotal))
                    validPositives = random.sample(positives, int(perc*posLen))
    
                    # remove validation files from list
                    negatives = [f for f in negatives if f not in validNegatives]
                    positives = [f for f in positives if f not in validPositives]
    
                    # select randomly corresponding nb of negatives
                    negatives = random.sample(negatives, int(negTotal-len(validNegatives)))

                    train = negatives + positives
                    validation = validPositives + validNegatives
    
                    for f in validation:
                        name = os.path.basename(f)
                        copy(f, destValid + name)
    
                    for f in train:
                        name = os.path.basename(f)
                        copy(f, destTrain + name)
    
                    print('Done splitting randomly ' + str(len(train)) + ' train and ' + str(len(validation)) + ' files.')
    

    #################################
    # Reads a genome and splits into synthetic
    # clusters according to a given length
    #################################
    def splitAsClusters(self):
        self.source_path = Utils.normalizePath(self.source_path)
        slimIDs = self.config.getboolean('corpusPrep', 'slim.id')
        files = Utils.listFilesExt(self.source_path, self.ext)
        result = []
    
        overlap = int((self.windowOverlap / 100) * self.length)
    
        for file in files:
            fileName = os.path.basename(file).split('.')[0]
    
            self.result_path = self.result_path + fileName + '_len' + str(self.length) + '_overlap' + str(self.windowOverlap)
            if(slimIDs):
                self.result_path += '_slimIDs'
            #self.result_path += '/'
            self.result_path += '.fasta'
    
            if(os.path.isfile(self.result_path)):
                print('File already exists: ' + self.result_path + '.\nDone.')
    
            else:
                file = Parsers.sortFasta(file)
                sequences = Parsers.parseFasta(file)
                content, ids, entry, overlapIds = '', '', '', ''
                for fasta in sequences:
                    content += str(fasta.seq.upper())
                    ids +=  str(fasta.id) if not ids else '|' + str(fasta.id)

                    if(slimIDs):
                        allIds = ids.split('|')
                        ids = allIds[0] + '|to|' + allIds[len(allIds)-1]

                    while(len(content) > 0):
                        varSize = self.length-(len(entry))
                        if (varSize <= overlap):
                            overlapIds += str(fasta.id) if not overlapIds else '|' + str(fasta.id)
                        entry += content[0:varSize]
    
                        if(len(entry) == self.length):
                            # move cursor on real sequence according to variable length added
                            content = content[varSize:]
                            # add chunk to list
                            if (slimIDs):
                                allIds = ids.split("|")
                                ids = allIds[0] + '|to|' + allIds[len(allIds) - 1]
                            result.append('>' + ids + '\n' + entry)
                            # make sure that entry contains overlap
                            entry = entry[len(entry)-overlap:]
    
                            if (len(content) > 0):
                                ids = overlapIds
                                overlapIds = ''
                            else:
                                ids = ''
    
                        elif(len(content) > 0 and len(entry) < self.length):
                            content = content[len(entry):]
    
                prev = 0
                pos = self.length

            result = '\n'.join(result)
            Utils.writeFile(self.result_path, result)

        print('Done.')
    

    #################################
    # Generate ".pfamTSV" files from "*.domain" files
    #################################
    def createPfamTsv(self):
        listFiles = Utils.listFilesExt(self.source_path, 'domains')

        head = 'sequence_id\tprotein_id\tgene_start\tgene_end\tgene_strand\tpfam_id\tin_cluster\n'
        contentPos = ''
        contentNeg = ''

        for file in listFiles:
            fileContent = Utils.readFileLines(file)
            id = fileContent[0].replace('>','')
            fileContent = fileContent[1:]
            inCluster = 1 if 'bgc' in os.path.basename(file).lower() else 0

            for line in fileContent:
                pfamId = line.split('|')[0]
                product = line.split('|')[1]
                currentLine = id + '\t' + product + '\t0\t0\t0\t' + pfamId + '\t' + str(inCluster) + '\n'
                if(inCluster == 1):
                    contentPos += currentLine
                else:
                    contentNeg += currentLine

        contentPos = head + contentPos[:-1]
        contentNeg = head + contentNeg[:-1]

        folder = os.path.basename(os.path.dirname(file))
        resultPos = self.result_path + folder + '.positives.pfam.tsv'
        resultNeg = self.result_path + folder + '.negatives.pfam.tsv'

        Utils.writeFile(resultPos, contentPos)
        Utils.writeFile(resultNeg, contentNeg)


if __name__ == '__main__':
    CorpusPreprocess().main()