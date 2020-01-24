from flask import Flask
import subprocess           # for running processes

app = Flask(__name__)

@app.route("/sixframe")
def sixFrame(sixpackPath, tempfile):
    result = subprocess.run(["sixpack",
                             "-sequence", tempfile,
                             "-outfile", sixpackPath,
                             "-outseq", "stdout",
                             "-auto", "Y"], stdout=subprocess.PIPE)
    return result


@app.route("/domains")
def domains(pfamScript, pfamDBs, tempfile):
    # run pfam and output domains
    result = subprocess.run([pfamScript,
                             "-fasta", tempfile,
                             "-dir", pfamDBs],
                            stdout=subprocess.PIPE)
    return result


@app.route("/blast")
def blast(tempfile, blastdbName, fileType, blastTask):
    service = "blastx" if("nucleotide") in fileType else "blastp" if ("protein") in fileType else ""
    source = "-subject" if "similar" in blastTask.lower() else "-db"

    # run blastx
    result = subprocess.run([service,
                             source, blastdbName,
                             "-query", tempfile,
                             "-outfmt", "6 std qcovs"], # output 'std' columns and 'qcovs' query coverage
                            stdout=subprocess.PIPE)
    return result