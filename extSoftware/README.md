### External software

#### Emboss

Option 1: ```user@foo:~$ sudo apt-get install emboss```

Option 2: 

```bash
user@foo:~$ wget ftp://emboss.open-bio.org/pub/EMBOSS/EMBOSS-6.5.7.tar.gz
user@foo:~$ gunzip EMBOSS-6.5.7.tar.gz
user@foo:~$ tar xvf EMBOSS-6.5.7.tar.gz
user@foo:~$ ./configure
user@foo:~$ make
```

#### PfamScan

```bash
user@foo:fungalbgcs~$ wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/Tools/PfamScan.tar.gz
user@foo:~fungalbgcs$ tar zxvf PfamScan.tar.gz
user@foo:~fungalbgcs$ cd PfamScan
```

[More details](https://gist.github.com/olgabot/f65365842e27d2487ad3)

#### HMMER

```bash
user@foo:fungalbgcs/PfamScan~$ wget http://eddylab.org/software/hmmer3/3.1b2/hmmer-3.1b2.tar.gz
user@foo:fungalbgcs/PfamScan~$ tar zxf hmmer-3.1b2.tar.gz
user@foo:fungalbgcs/PfamScan~$ cd hmmer-3.1b2
user@foo:fungalbgcs/PfamScan~$ ./configure
user@foo:fungalbgcs/PfamScan~$ make 
user@foo:fungalbgcs/PfamScan~$ make check
user@foo:fungalbgcs/PfamScan~$ make install
user@foo:fungalbgcs/PfamScan~$ export PATH=/path/to/hmmerExecs:$PATH
```

#### CPAN Moose + BioPerl

```bash
user@foo:~$ cpan Moose
cpan > d /bioperl/
cpan > install C/CJ/CJFIELDS/BioPerl-1.007001.tar.gz
cpan > exit
user@foo:~$ export PERL5LIB=/path/to/PfamScan:$PERL5LIB
```

#### Pfam DBs

```bash
user@foo:fungalbgcs/PfamScan~$ wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
user@foo:fungalbgcs/PfamScan~$ wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz
user@foo:fungalbgcs/PfamScan~$ tar zxf Pfam-A.hmm.dat.gz
user@foo:fungalbgcs/PfamScan~$ mkdir /PfamDB
user@foo:fungalbgcs/PfamScan~$ mv Pfam-A.hmm.dat /PfamDB
user@foo:fungalbgcs/PfamScan~$ tar zxf Pfam-A.hmm.gz
user@foo:fungalbgcs/PfamScan~$ cd hmmer-3.1b2
user@foo:fungalbgcs/PfamScan~$ hmmpress ../PfamDB/Pfam-A.hmm
```





