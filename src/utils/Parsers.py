from Bio import SeqIO
from utils import UtilMethods as Utils
import os

###############
# Parser methods.
################


def parseDatasetContents(dataPath, featType, sourceType):
    files, result = [], []
    if ('domain' in featType or 'dictionary' in featType):
        domainFiles = Utils.listFilesExt(dataPath, 'domains')
        files += domainFiles
        if (len(domainFiles) < 1):
            print('No domains / dictionary files found in', dataPath)
            exit()

    if ('kmers' in featType or 'prot' in featType):
        fastaFiles = Utils.listFilesExt(dataPath, 'fasta')
        files += fastaFiles
        if (len(fastaFiles) < 1):
            print('No fasta files found in', dataPath)
            exit()

    if ('go' in featType):
        goTermFiles = Utils.listFilesExt(dataPath, 'go')
        files += goTermFiles
        if (len(goTermFiles) < 1):
            print('No GO term files found in', dataPath)
            exit()

    for file in files:
        ext = os.path.splitext(file)[1]
        if ('fasta' in ext):
            content = Utils.readFileLines(file)[1].upper()
            content = normalizeSequence(content, sourceType)
            if('kmers' in featType):
                result.append(((file, content),'kmers'))
            if('prot' in featType):
                result.append(((file, content), 'protanalys'))

        elif ('domain' in ext):
            content = Utils.readFileLines(file)[1:]
            content = [line.split('|')[0] for line in content]
            content = "\n".join(content)
            result.append(((file, content), 'domains'))

        elif ('go' in ext):
            content = Utils.readFileLines(file)[1:]
            content = "\n".join(content)
            result.append(((file, content), 'go'))

    return result


def normalizeSequence(content, sourceType):
    translationTab = ''
    if('nucleotide' in sourceType):
        translationTab = dict.fromkeys(map(ord, 'BDEFHIJKLMNOPQRSUVXYWZ'), None)
    elif('aminoacid' in sourceType):
        translationTab = dict.fromkeys(map(ord, 'BJXZ-'), None)
    result = content.translate(translationTab)

    return result


# parse fasta using biopython
def parseFasta(entry):
    if(os.path.isfile(entry)):
        entry = open(entry)

    return SeqIO.parse(entry, 'fasta')


# parse fasta using biopython
# returns a list of [> ID \n sequences]
def parseFastaToList(path, filter):
    thislist, files, filterIDs, filename_content = [],[],[], []

    if(os.path.isfile(path)):
        files.append(path)
    else:
        files = Utils.listFilesExt(path, 'fasta')

    if (os.path.isfile(filter)):
        filterIDs = Utils.readFileLines(filter)
    else:
        filterIDs = filter.split('\n')

    for file in files:
        sequences = parseFasta(file)
        for fasta_record in sequences:
            output = '>' + str(fasta_record.id) + '\n' + str(fasta_record.seq)
            if(len(filterIDs) > 0):
                if(str(fasta_record.id)) not in str(filterIDs):
                    thislist.append(output)
                    filename_content.append(tuple([file, output]))
            else:
                thislist.append(output)
                filename_content.append(tuple([file, output]))

    return thislist, filename_content


# parse aminoacid sequences from
# GenBank file to a list
def genBankToAminoacid(path):
    list = []
    # only aminoacid sequence
    translations = ''
    files = []
    if(os.path.isfile(path)):
        files.append(path)
    else:
        files = Utils.listFilesExt(path, 'gbk')

    for file in files:
        species = Utils.getSpecies(file)
        records = parseGenBank(file)

        for record in records:
            locus = record.id
            for feature in record.features:
                #if feature.key == "CDS":
                if feature.type == "CDS":
                    id, locus_tag, gene, protein_id, translation, \
                    product, function, description  = '','','','','','','',''

                    for key, value in feature.qualifiers.items():
                        # get rid of the quotes around the qualifier
                        # find entry ID
                        if key == "translation":
                            translation = value[0]
                        elif key == "gene":
                            gene = value[0]
                        elif key == "locus_tag":
                            locus_tag = value[0]
                        elif key == "protein_id":
                            protein_id = value[0]
                            protein_id = protein_id.replace('/','')
                        elif key == "product":
                            product = value[0]
                        elif key == "function":
                            function = value[0]

                    #priority for gene ID
                    id = locus_tag if not id and len(locus_tag) > 1 else id
                    id = gene if not id and len(gene) > 1 else id
                    id = protein_id if not id and len(protein_id) > 1 else id

                    description = product if product.strip() else description
                    description += '|' + function if function.strip() else description

                    entry = '>' + locus + '|' + species + '|' + id + '|' + description + '\n' + translation
                    if(entry not in list):
                        list.append(entry)
                        translations += translation

    return list, translations


# parse GenBank file to a string (nucleotide sequence)
def genBankToNucleotide(path):
    if (path.endswith(".gbk")):
        records = parseGenBank(path)
    seq = ''
    for record in records:
        seq += str(record.seq)

    return seq


def parseGenBank(entry):
    if(os.path.isfile(entry)):
        entry = open(entry)
    records = SeqIO.parse(entry, 'genbank')

    return records


def genBankToFasta():
    config = Utils.loadConfig()
    source = config.get('parsers', 'source.path')

    if not str(source).endswith('/'):
        output = source + '_fasta/'
        source += '/'
    else:
        source[len(source) - 1] = ''
        output = source + '_fasta/'
        source += '/'

    os.makedirs(os.path.dirname(output), exist_ok=True)

    list = genBankToAminoacid(source)
    content = ''

    for item in list:
        content += item + '\n'

    Utils.writeFile(output + 'fungi_complete' + '.fasta', content)


#handles single fasta file with list of cluster contents
def mergeByClusterIds(source, filterIds):
    result = []
    fastaList = []
    if(isinstance(source, str) and os.path.isdir(source)):
        fastaList, fastaListFiles = parseFastaToList(source,filterIds)
    else:
        fastaList = source
    fastaIter = enumerate(fastaList, start=0)
    clusterdic = {}

    for i, item in fastaIter:
        id = item.split('\n')[0]
        clusterId = id.split('|')[0].split('.')[0]
        sequence = item.split('\n')[1]

        if(clusterId not in clusterdic.keys()):
            clusterdic[clusterId] = sequence
        else:
            clusterdic[clusterId] += sequence

    for key, value in clusterdic.items():
        content = key + '\n' + value
        result.append(content)

    return result, clusterdic