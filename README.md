## LVpocket
We proposed LVPocket, a novel method that synergistically captures both local and global information of protein data through the integration of Transformer encoders, which help the model achieve better performance in binding pockets prediction. And then we tailored prediction models for data of four distinct structural classes of proteins using the transfer learning. The four fine-tuned models were trained on the baseline LVPocket model which was trained on the sc-PDB dataset. LVPocket exhibits superior performance on three independent datasets compared to current state-of-the-art methods. Additionally, the fine-tuned model outperforms the baseline model in terms of performance. <br>

In addition, we have developed a protein structure classifier(SCOP classifier) to help users confirm the structure classification of the proteins used. And the results could provide reference for users when choosing protein binding pocket prediction models.


#  Usage

## 1.  Download the model files and KV3K dataset

   The Zenodo doi of KV3K dataset：[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10633986.svg)](https://doi.org/10.5281/zenodo.10633986)
  
   The Zenodo doi of model files: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10633690.svg)](https://doi.org/10.5281/zenodo.10633690)

## 2.	 Installation
####  1.Clone this repository
    git clone https://github.com/ZRF-ZRF/LVpocket.git
    cd LVpocket
#### 2.Create the environment with all dependencies
    conda env create -f environment.yml -n lvpocket
    conda activate lvpocket
###  3.Optionally run tests to make sure everything works as expected
    conda install pytest
    pytest -v
    
## 3.  Data preparation
We used scPDB dataset to train our model, you can get the dataset at this link:http://bioinfo-pharma.u-strasbg.fr/scPDB/. Before training, you need to build a training dataset using following code:

    python scripts/prepare_dataset.py --dataset /path/scpdb_path --output scpdb_dataset.hdf --exclude data/scPDB_blacklist.txt data/scPDB_leakage.txt
## 4.  Model training
You can training the model use the following code:

    python scripts/trian.py --input scpdb_data.hdf --output /output_path --test_ids data/test_ids

## 5.  The prediction of protein pockets
You can choose baseline model or SCOP fine-tuned models to predict protein binding pockets based on the protein sctructure.

### (1). The classification of protein structure

#### Additional: If you want to know which structure class your input protein is, you can use the protein structure classifier we built to do so.The usage process of this method is as follows:
   ##### ①Install [PSIPRED](https://github.com/psipred/psipred). The download address: http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/
   ##### ②Use the pdb_to_fasta method to convert the PDB format file to FASTA format.
         cd scop_classifier
         python pdb_to_fasta.py --pdb_file=path/[protein.pdb] --fasta_file=path/[results.fasta]

         --pdb_file The filepath of protein for conversion, it can deal with the format of '.pdb'.
         --fasta_file The output path of the conversion results.
   ##### ③Use the PISPRED to predict the secondary structure sequence of the protein.
         cd psipred/
         ./runpsipred path/[resulfs.fasta]
   ##### ④Predict the structural classification of the protein.
         python classifier.py --horiz_file_path=path/[your.horiz]

         --horiz_file_path The filepath to the horiz file predicted by PSIPRED.
### (2).  Protein binding pockets prediction    

    python scripts/predict.py --input data/test_protein.pdb --output prediction_output --model baseline_model.hdf --format pdb
    
    --input The filepath of protein for prediction, we can deal with the format of '.pdb' or '.mol2';
    --output The output path of the prediction results;
    --model The model file to predict. We have five models, you can select one which is good for the input protein classification;
    --format The protein format of the input protein file('.pdb/.mol2)'.
    
    
