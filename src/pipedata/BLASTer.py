import os
from utils import UtilMethods as Utils
from pipedata import WebService as webService
from io import StringIO
import pandas

###############
# Runs BLAST service and provides
# access to BLAST DBs
###############

class BLASTer:

    def __init__(self, blastTask):
        self.config = Utils.loadConfig()
        self.sourceType = self.config.get('dataPipeline', 'source.type')
        self.blastTask = blastTask
        self.blastdb = self.config.get('blaster', 'blastdb.path')
        self.blastdb = Utils.normalizePath(self.blastdb)
        self.blastdbName = self.config.get('blaster', 'blastdb.name')
        if(not self.blastdbName.endswith('fasta')):
            self.blastdbName += '.fasta'

        self.goTerms = True if 'goterm' in blastTask.lower() else False
        self.mappingFile = self.blastdb + self.blastdbName.replace('.fasta','.tab')
        self.mapping = ''
        if(self.goTerms):
            self.mapping = self.loadMapping()


    def main(self, source_file, sequence, db):
        if(not db):
            # add blastdb path to run environment
            os.environ["BLASTDB"] = self.blastdb
        else:
            self.blastdbName = db

        useTempFile = False
        tempFile = ""
        if(not os.path.isfile(sequence)):
            useTempFile = True

        if(useTempFile):
            # if query is a sequence, create a temp file
            tempLine = str(sequence).split('\n')[0]
            tempLine = tempLine.split(' ') if ' ' in tempLine else tempLine.split('|')
            tempLine = tempLine[0] + tempLine[len(tempLine)-1]
            tempFile = tempLine + '.temp'
        else:
            # if query is a file, do not create a file
            tempFile = sequence

        # create/open temp file in project directory
        with open(tempFile, 'r+') as tempfile:
            # write sequence in a temp file, and return cursor to file position 0
            if(useTempFile):
                tempfile.write(str(sequence))
                tempfile.seek(0)

            result = webService.blast(tempFile, self.blastdbName, self.sourceType, self.blastTask)
            id, result = self.parseBlast(result)

            if(self.goTerms):
                id, result = self.parseMapping(id, result)
            # clear temp file content
            if(useTempFile):
                tempfile.flush()

        # remove temp files
        if (useTempFile and os.path.isfile(tempFile)):
            os.remove(tempFile)

        result = id + result

        return ([source_file, result])


    def parseBlast(self, result):

        result = result.stdout.decode('utf-8')
        id = ""

        if (self.goTerms):
            # add blast result columns
            columns = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'qcovs']
            df = pandas.read_csv(StringIO(result), sep='\t', names=columns, index_col=False)
            # recover cluster ID
            id = df.iloc[0]['qseqid']
            id = '>' + id + '\n'

            # select results considering evalue and qcovs thresholds
            df = df.loc[(df['evalue'] < 0.00001) & (df['qcovs'] > 50)]
            df = df.sort_values(by='qcovs', ascending=False)
            # selecting only unique IDs
            df = df['sseqid'].unique()
            result = pandas.Series(df)

        return id, result


    def loadMapping(self):
        return pandas.read_csv(self.mappingFile, sep='\t', header=0, index_col=False)


    def parseMapping(self, id, result):
        goIds = set()
        for item in result:
            idsFromBlast = self.mapping.loc[(self.mapping['entry'].str.contains(item))]
            theseIds = idsFromBlast['goids'].drop_duplicates().values.tolist()
            theseIds = ''.join(theseIds).split(';')
            goIds.update(theseIds)

        goIds = '\n'.join(goIds).replace(' ','')

        return id, goIds