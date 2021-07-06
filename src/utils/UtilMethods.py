import configparser
import glob
import os
import pandas
import numpy as np
import gzip
from pyspark import SparkConf

###############
# Util methods for IO.
###############


# config
def loadConfig():
    path = os.path.dirname(__file__)
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(path + '/../config.init')

    return config


# shuffles char-grams
def charGramShuffle(content, size):
    test = list(content)
    pairs = [test[i:i + size] for i in range(0, (len(test) - 1), size)]
    np.random.shuffle(pairs)
    result = ''.join(str(r) for v in pairs for r in v)

    return result


def BlastOutToTable(file):
    tab = pandas.read_table(file, index_col=False)
    # removing less likely significant values
    tempEval = tab[tab.evalue < 0.01]
    tempSort = tempEval.sort_values(by=['sseqid','bitscore','pident','qseqid'], ascending=[True,False,False,True])

    return tempSort


# load dictionary of target cluster IDs (MIBiG fungi)
def loadTargetClusterList(targetClusters):
    dic = {}
    with open(targetClusters) as file:
        for line in file:
            if(len(line) > 1):
                (key, val) = line.replace('\n','').replace(' ','_').split('\t')
                dic[key] = val
    return dic


def foldClusterData(list, filetype, threshold):
    unique = {}
    list = [line for line in list[2:] if float(line.split('\t')[2]) >= threshold] if not filetype else list
    list = sorted(list)
    for entry in list:
        id, gene = '', []
        if('gold' in filetype):
            id = entry.split('\t')[0]
            gene.append(entry.split('\t')[1])
        else:
            line = entry.split('\t')
            id = line[1]
            gene = line[3].split('|')

        if (unique.get(id) is None):
            unique[id] = gene
        else:
            if(not set(gene).issubset(unique[id])):
                unique[id].extend(gene)

    return unique


def unfoldResultData(threshold, list):
    unique = set()
    pos = [x for x in list if float(x.split('\t')[1]) >= threshold]
    for entry in pos:
        genes = entry.split('|')
        for gene in genes:
            gene = gene.split('\t')[0]
            if(gene):
                unique.add(gene)

    return pos, unique


def getFileName(path):
    name = os.path.basename(path)

    return name


def getFileExt(path):
    result = os.path.splitext(path)

    return result


# IO
def writeFile(path, content):
    with open(path, 'w+') as file:
        # write content in a file, and return cursor to file position 0
        file.write(str(content))
        file.seek(0)
        file.close()


def normalizePath(path):
    if(os.path.isdir(path)):
        if not str(path).endswith('/'):
            path += '/'
        return path
    else:
        print('Path does not exist: ' + path)
        exit()


#IO
def readFileLines(file):
    if(os.path.exists(file)):
        with open(file) as thisfile:
            return [line.rstrip('\n') for line in thisfile]
    else:
        print('File does not exist: ' + file)
        return ""


#IO
def extractGz(file):
    f = gzip.open(file, 'rb')
    file_content = f.read()
    file_content = str(file_content, 'utf-8')
    f.close()

    return file_content


# get name of species from path
def getSpecies(path):
    # recover the species name from a path
    species = ''
    if (len(path) > 0):
        species = path.split('/')
        if (len(species) >= 2):
            if(os.path.isfile(path)):
                species = species[len(species) - 2]
            else:
                species = species[len(species) - 1]

    return species


# provide label according to filename
def getLabel(filename):
    if 'bgc' in str(filename).lower():
        return 1
    else:
        return 0


# list files in a directory
def listFilesExt(path, extension):
    if (os.path.isdir(path)):
        files = glob.glob(path + '/**/*.'+ extension, recursive=True)
        return files
    else:
        print('Path does not exist: ' + path)
        exit()

def getSparkConf(name):
    conf = SparkConf().setAppName(name)
    conf = (conf.setMaster('local[*]')
            .set('spark.executor.memory', '8G')
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G')
            .set('spark.network.timeout', '10000s')
            .set('spark.executor.heartbeatInterval', '120s')
            .set('spark.pyspark.python', os.path.dirname(__file__).join('.env/bin/python')))

    return conf
