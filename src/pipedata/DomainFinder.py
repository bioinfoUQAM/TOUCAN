import os
import re
from pipedata import WebService
from utils import UtilMethods as Utils

###############
# Annotates Pfam domains in a given sequence
# and provides access to Pfam resources
# (script, HMM models)
###############

class DomainFinder:

    def __init__(self):
        self.config = Utils.loadConfig()


    def main(self, source_file, content):
        self.pfamScript = self.config.get('domainFinder', 'pfamScript')
        self.pfamDBs = self.config.get('domainFinder', 'pfamDBs.path')
        self.pfamScan = self.config.get('domainFinder', 'pfamScan.path')
        self.hmmer = self.config.get('domainFinder', 'hmmerBin.path')
        os.environ["PATH"] += ':' + self.hmmer
        os.environ['PERL5LIB'] = self.pfamScan

        tempLine = str(content).split('\n')[0]
        tempLine = tempLine.split(' ')[0] if ' ' in tempLine else tempLine.split('|')[0]
        tempFile = tempLine.replace('>','') + '.temp'

        with open(tempFile, 'w+') as tempfile:
            # write sequence in a temp file, and return cursor to file position 0
            tempfile.write(content)
            tempfile.seek(0)

            pfam = WebService.domains(self.pfamScript, self.pfamDBs, tempFile)
            pfam = self.parseDomain(pfam)
            tempfile.flush()

        #remove temp files
        if os.path.isfile(tempFile):
            os.remove(tempFile)

        return source_file, pfam


    def parseDomain(self, result):
        temp = result.stdout.decode('utf-8')
        result = ''
        # remove header
        temp = re.split('>\s\s', temp)

        if(len(temp) > 1):
            result = temp[1]

        # if no domain is found, then return empty string
        if(len(result.replace('\s','')) == 0):
            result = ''

        return result


    def parseDicValues(self, dic):
        str = ''
        for value in dic.values():
            str += value + '\n'
        return str


if __name__ == '__main__':
    DomainFinder.main()