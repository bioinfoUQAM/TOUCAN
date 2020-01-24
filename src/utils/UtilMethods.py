import configparser
import glob
import os
import pandas
import numpy as np
import gzip

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
       return [line.rstrip('\n') for line in open(file)]
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

    if 'negative' in filename:
        return 0
    else:
        return 1


# list files in a directory
def listFilesExt(path, extension):
    if (os.path.isdir(path)):
        files = glob.glob(path + '/**/*.'+ extension, recursive=True)
        return files
    else:
        print('Path does not exist: ' + path)