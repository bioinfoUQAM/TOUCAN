from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from gff3 import Gff3
from utils import UtilMethods as Utils
from collections import OrderedDict
import os, re

###############
# Parser methods.
################


def orthogroupSeqs(orthofile, seqpath, limit):
    orthodir = os.path.dirname(seqpath)
    ortholines = Utils.readFileLines(orthofile)[1:]
    seqpath = Utils.listFilesExt(seqpath, "fasta")
    threshold =  limit if(limit) else len(seqpath)
    orthogroups = [re.split('\t|;|,', item)[1:] for item in ortholines][1:threshold+1]
    sequences, output = dict(), dict()

    print('Loading files and seqIDs...')
    for seqfile in seqpath:
        sequences.update(SeqIO.index(seqfile, "fasta"))

    orthodir = orthodir + '/orthologs_threshold' + str(threshold) + '/'

    if os.path.isdir(orthodir):
        print('Orthogroup path', orthodir, 'already exists.')
        exit()
    else:
        os.makedirs(orthodir)

        print('Loading sequences per IDs...')
        for group in orthogroups:
            for id in group:
                id = id.strip(' ')
                tempseq = sequences.get(id)

                if(tempseq is not None and len(tempseq) > 1):
                    thisseqfile = orthodir + tempseq.id + '.fasta'
                    content = '>' + tempseq.id + '\n' + tempseq.seq
                    Utils.writeFile(thisseqfile, content)
                # else:
                #     print('ID not found', str(id))

    print('Done writing seqs for orthogroups.')
    return output



def clustersToGFF(clusterspath, gffpath, goldpath, annotpath, source_type):
    gffcontent = Gff3(gffpath)
    clustercontent, goldContent, annotationContent = "","",""

    clustercontent = Utils.readFileLines(clusterspath)
    clusters = Utils.foldClusterData(clustercontent, "", 0.5) if 'score' in clusterspath else Utils.foldClusterData(clustercontent, "gold", "")

    goldContent = '\t'.join(Utils.readFileLines(goldpath)) if goldpath else ""
    annotationList = Utils.readFileLines(annotpath) if annotpath else ""
    annotationContent = ('\n').join(annotationList) if annotpath else ""

    # sort dict by key
    clusters = OrderedDict(sorted(clusters.items(), key=lambda x: x[0]))
    gffclusterfile = clusterspath.rsplit('.',1)[0] + '.percluster.gff3'
    gffgenefile = clusterspath.rsplit('.',1)[0] + '.pergene.gff3'

    outputcluster, outputgene = "##gff-version 3\n", "##gff-version 3\n"
    # filter only "mRNA" features, return dict {gene name, gff line}
    mRNAdict = {line['attributes']['Name'].replace('.1',''):line  for line in gffcontent.lines if line['type'] == 'mRNA'}

    for key, value in clusters.items():
        for gene in value:
            gene = gene.replace('.1', '')
            thisgene = mRNAdict.get(gene)

            if(thisgene is not None):
                chr = thisgene['seqid']
                position = str(thisgene['start']) + '\t' + str(thisgene['end'])
                score = '?'
                strand = thisgene['strand']
                phase = thisgene['phase']
                info = 'Name=' + gene + ';Note=' + key + '\n'

                if(goldContent):
                    if(gene in annotationContent):
                        annot = [item for item in annotationList if gene in item]
                        annot = annot[0].split('\t')[1] if annot else ''
                        if('backbone' in annot):
                            info = info.replace("\n", ";color=#EE0000\n") # red
                        elif('tailor' in annot):
                            info = info.replace("\n", ";color=#EE9300\n")  # orange
                        elif('transcript') in annot:
                            info = info.replace("\n", ";color=#048014\n")  # forest green
                        elif('transport' in annot):
                            info = info.replace("\n", ";color=#1888f0\n")  # light blue
                    elif(gene in goldContent):
                        info = info.replace("\n", ";color=#9931f2\n") # bright purple
                outputgene += chr + '\t' + source_type + '\t' + position + '\t' + score + '\t' + strand + '\t' + phase + '\t' + info

            else:
                print('gene not found:', gene)

        startID = value[0].replace('.1', '')
        endID = value[-1].replace('.1', '')
        startGene = mRNAdict.get(startID)
        endGene = mRNAdict.get(endID)
        chr = startGene['seqid']
        position = str(startGene['start'])  + '\t' + str(endGene['end'])

        strand = startGene['strand']
        phase = startGene['phase']
        score = '?'
        info = 'Name=' + key + ';Note=' +('|').join(value) + '\n'
        outputcluster += chr + '\t' + source_type + '\t' + position + '\t' + score + '\t' + strand + '\t' + phase + '\t' + info

    Utils.writeFile(gffclusterfile, outputcluster)
    Utils.writeFile(gffgenefile, outputgene)

    return gffcontent




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
        lines = Utils.readFileLines(file)
        #handle genes with an added version number as NRRL3_00129.1
        id = lines[0].replace('>','').replace('a','').split('.')[0]
        if ('fasta' in ext):
            #content = Utils.readFileLines(file)[1].upper()
            content = lines[1].upper()
            content = normalizeSequence(content, sourceType)
            if('kmers' in featType):
                result.append(((file, content, id),'kmers'))
            if('prot' in featType):
                result.append(((file, content, id), 'protanalys'))

        elif ('domain' in ext):
            #content = Utils.readFileLines(file)[1:]
            content = lines[1:]
            # Comment out next line to keep domain name:
            content = [line.split('.')[0] for line in content]
            content = "\n".join(content)

            if('pfam' in featType):
                temp = content.split('\n')
                for entry in temp:
                    result.append(((file, entry, id), 'domains'))
            else:
                if(content):
                    result.append(((file, content, id), 'domains'))

        elif ('go' in ext):
            #content = Utils.readFileLines(file)[1:]
            content = lines[1:]
            content = "\n".join(content)
            result.append(((file, content, id), 'go'))

    return result


def normalizeSequence(content, sourceType):
    translationTab = ''
    if('nucleotide' in sourceType):
        translationTab = dict.fromkeys(map(ord, 'BDEFHIJKLMNOPQRSUVXYWZ'), None)
    elif('aminoacid' in sourceType):
        translationTab = dict.fromkeys(map(ord, 'BJXZ-'), None)
    result = content.translate(translationTab)

    return result


def translateGenBank2file(input):

    results = parseGenBank(input)
    region = input.split('region')[1].replace('.gbk', '')
    outputseq = input.replace('gbk', 'fasta')
    outputtranslate = outputseq.replace('/fasta', '/fastatranslation')

    for record in results:
        uniqueid = record.id + region
        proteinseq = record.seq.translate()
        descript = record.annotations.get('organism')
        translation = ''
        for item in record.features:
            try:
                if(item.qualifiers['translation']):
                    translation += item.qualifiers['translation'][0]
            except:
                pass

        tempseq = SeqRecord(id=uniqueid, seq=proteinseq, description=descript)
        temptranslate = SeqRecord(id=uniqueid, seq=Seq(translation, IUPAC), description=descript)
        SeqIO.write(tempseq, outputseq, 'fasta')
        SeqIO.write(temptranslate, outputtranslate, 'fasta')


    # for record in parseFasta(input):
    #     recid = str(record.id).split('|')
    #     uniqueid = recid[0] + '_' + recid[1]
    #     # get description without id
    #     descript = str(record.description).replace(('|').join(recid[0:2]),'')
    #     filename = outputpath + uniqueid + '.fasta'
    #     proteinseq = record.seq.translate()
    #     temp = SeqRecord(id=uniqueid, seq=proteinseq, description=descript)
    #     SeqIO.write(temp, filename, 'fasta')


# parse fasta using biopython
def parseFasta(entry):
    if (os.path.isfile(entry)):
        entry = open(entry)
    return SeqIO.parse(entry, 'fasta')


def sortFasta(entry):
    filename = str(entry)
    if('sorted' not in entry):
        filename = filename.replace('.fasta', '_sorted.fasta')
    if(not os.path.isfile(filename)):
        records = list(SeqIO.parse(entry, 'fasta'))
        records.sort(key=lambda seq: seq.id)
        SeqIO.write(records, filename, 'fasta')
    return filename


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

# merges a list of cluster genes into a
# list of clusters (by cluster gene ID)
def mergeGenesToClusters(genes):
    output = []
    previousId = ''
    length = len(genes)
    for i, gene in enumerate(genes):
        id = gene.split('|')[0]
        seq = gene.split('\n')[1]

        if (id in previousId):
            entry += seq
        else:
            if(previousId):
                output.append(entry)
            previousId = id
            entry = id + '\n' + seq

    output.append(entry)

    return output