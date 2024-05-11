from Bio import PDB
import os
from Bio.PDB import PDBParser, PPBuilder, Polypeptide, is_aa
import pandas as pd

df = pd.DataFrame(columns=["class", "protein", "AAC"])
aa_codes = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def pdb_to_fasta(pdb_file, fasta_file):
    # 创建PDB解析器
    parser = PDBParser(PERMISSIVE=1)

    # 读取PDB文件
    structure = parser.get_structure("structure", pdb_file)
    # ppb = PPBuilder()
    # seq=''
    # 获取所有的氨基酸序列
    # for pp in ppb.build_peptides(structure):
    #     seq+=pp.get_sequence()

    # 获取所有的氨基酸序列
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # if residue.get_id()[0] == " ":  # 排除非标准残基
                #     residues.append(residue)
                if is_aa(residue):
                    aa = residue.get_resname()
                    if aa in aa_codes:
                        residues.append(residue)
    ls = [Polypeptide.three_to_one(residue.get_resname()) for residue in residues]
    seq = "".join(ls)
    # 将氨基酸序列写入Fasta文件
    with open(fasta_file, "w") as f:
        f.write(str(seq))


def predictSS(file_path):
    os.system("%s %s " % ("cd ../psipred;./runpsipred ", file_path))
    dirname = "../psipred"
    if os.path.exists("../psipred/" + file_path[-10:-6] + ".horiz"):
        os.system("mv  " + dirname + "/*.horiz  /4T/lssFile/CPUpocket/horiz/")
        os.system("rm -rf " + dirname + "/*.ss*")
        return "predicted"
    return "noSS"


def horizRead(file_path):
    a = {}
    HandleData = []
    data = pd.read_csv(file_path)
    SSseq = ""
    AAC = ""
    # id=file[0:-6]
    for i in range(len(data)):
        ls = data["# PSIPRED HFORMAT (PSIPRED V4.0)"][i]
        if ls[0:4] == "Pred":
            SSseq = SSseq + ls[6:]
        elif ls[0:4] == "  AA":
            AAC = AAC + ls[6:]

    a["AAC"] = AAC
    a["SSseq"] = SSseq

    HandleData.append(a)
    variables = list(HandleData[0].keys())
    dataframe = pd.DataFrame(
        [[i[j] for j in variables] for i in HandleData], columns=variables
    )
    return dataframe
