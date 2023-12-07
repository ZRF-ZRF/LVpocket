#If you want to know which structure class your input protein is, you can use the protein structure class prediction algorithm we built to do so.

## process
#1、	Install PSIPRED
#2、	Use the pdb_to_fasta method to convert the PDB format file to FASTA format
#3、	The PSIPPRED tool was used to predict the secondary structure sequence of the protein
#4、	Extract features based on the secondary structure sequence of the protein and make predictions
##Python Environment
conda install biopython

