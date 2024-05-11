import argparse

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

    # 获取所有的氨基酸序列
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    aa = residue.get_resname()
                    if aa in aa_codes:
                        residues.append(residue)
    ls = [Polypeptide.three_to_one(residue.get_resname()) for residue in residues]
    seq = "".join(ls)
    # 将氨基酸序列写入Fasta文件
    with open(fasta_file, "w") as f:
        f.write(str(seq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", type=str)
    parser.add_argument("--fasta_file", type=str)
    args = parser.parse_args()
    pdb_to_fasta(args.pdb_file, args.fasta_file)
