import os, re
from utils import Parsers
from pipedata import DomainFinder, SixFrameTranslator, BLASTer
from utils import UtilMethods as Utils
from pyspark import SparkConf, SparkContext

###############
# Processes sequences with:
# - BLAST (to build similarity matrix or find GO terms)
# - Pfam (to identify protein domains)
# Performs 6-frame translation if input is nucleotide.
###############

class DataPipeline:


    def __init__(self, source_type=None, source_path=None, result_path=None):
        self.config = Utils.loadConfig()
        self.task = self.config.get('dataPipeline', 'task')
        self.source_path = self.config.get('dataPipeline', 'source.path') if source_path is None else source_path
        self.source_type = self.config.get('dataPipeline', 'source.type') if source_type is None else source_type
        self.result_path = self.config.get('dataPipeline', 'result.path') if result_path is None else result_path
        self.result_path = Utils.normalizePath(self.result_path)
        # create if it doesnt exist
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        # recover the species name for using in temp files
        self.species = Utils.getSpecies(self.source_path)
        # temp dir + file used by sub-pipelines
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.path += '/temp/'
        os.makedirs(os.path.dirname(self.path), exist_ok=True)


    def main(self):
        print('Starting '+ self.task +' process of fasta files for ' + self.species +'...')
        sparkContext = SparkContext(conf=self.initSpark("datapipeline"))
        thisfile, result = '', ''

        if('domains' in self.task):
            result = self.getDomains(sparkContext)

        if('blast' in self.task):
            result = self.getBLAST(sparkContext, "", "")

        for filename, content in result:
            if (len(content) > 0):
                with open(filename, 'w+') as f:
                    # write domains to file, and return cursor to file position 0
                    f.write(str(content))

        if(os.path.exists(self.path)):
            try:
                os.rmdir(self.path)
            except:
                print("Warning: Temp folder not removed since has temp files being used.")

        print('Finished all processes.')


    def getBLAST(self, file_content, sparkContext, blastTask):
        blaster = BLASTer.BLASTer(blastTask)
        # create RDD with source sequences
        sourceRDD = sparkContext.parallelize(file_content, numSlices=2000)

        # find similarity
        if("similar" in blastTask):
            blastRDD = sourceRDD.map(lambda x: blaster.main("", x[0], db=x[1]))
            result = blastRDD.collect()
        else:
            blastRDD = sourceRDD.map(lambda x: blaster.main(x[0], x[1], db=""))
            result = blastRDD.collectAsMap()

        return result



    def getDomains(self, sparkContext):

        # recover the species name for using in temp files
        self.species = Utils.getSpecies(self.source_path)
        domainFinder = DomainFinder.DomainFinder()

        # load source sequences into a single list
        if("fasta" in self.source_type):
            list, file_content = Parsers.parseFastaToList(self.source_path, "")
        elif("genbank" in self.source_type):
            list = Parsers.genBankToAminoacid(self.source_path)

        print('Processing domains...')

        # create RDD with source sequences
        sourceRDD = sparkContext.parallelize(file_content, numSlices=2000)

        if("nucleotide" in self.source_type):
            # execute sixFrame translation for each sequence in RDD
            sourceRDD = sourceRDD.map(lambda x: SixFrameTranslator.main(x))

        # execute Pfam domain prediction for each sixFrame translation in RDD
        domainsRDD = sourceRDD.map(lambda x: domainFinder.main(x[0], x[1]))
        processedRDD = domainsRDD.map(lambda x: self.processDomainOutput(x[0], x[1]))

        # recover Pfam domain prediction results from RDD
        result = processedRDD.collectAsMap()

        print('Done!')

        return result


    def processDomainOutput(self, filename, content):
        if(len(content) > 1):
            content = str(content)
            content = content.replace('\\n ', '\\n')
            content = re.sub(' ', '\t', content)
            content = re.sub('\t+', '\t', content)

            if(len(content) > 1 and content[0] == '\t'):
                content.replace('\t', '', 1)
            content = content.replace('\\n', '\n')

            return filename, content
        else:
            return filename, ''



    def initSpark(self, taskName):
        conf = SparkConf().setAppName(taskName)
        # create spark context instance for pipeline
        conf = (conf.setMaster('local[*]')
                .set('spark.executor.memory', '8G')
                .set('spark.driver.memory', '50G')
                .set('spark.driver.maxResultSize', '20G')
                .set('spark.network.timeout', '10000s')
                .set('spark.executor.heartbeatInterval', '120s')
                .set('spark.pyspark.python', os.path.dirname(__file__).join('.env/bin/python')))

        return conf


if __name__ == '__main__':
    DataPipeline().main()