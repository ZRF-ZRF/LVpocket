## LVpocket
  
   We propose a deep learning model- LVpocket, which combines the strength of the V-NET model and the ResNet model to predict protein-ligand binding pockets, which could solve the problems of imbalanced sample data, gradient disappearance, and explosion during training. Specially, we preprocess the data from scPDB and divide the proteins into four classes: all α, all β, α + β, α / β, which we use to fine-tune four models which have different hyperparameters based on LVPocket to customize a predictive model for each classification protein structure. The results of our model on three independent data sets highlight its better performance over current state-of-the-art methods and its ability to generalize novel protein structures well.



#  Usage

## 1.  Download the model files and KV3K dataset

   Links：https://pan.baidu.com/s/1kOUircaFHDdapegPvhzH3w 
  
   Code：lvlv

## 2.	 Clone this repository

    git clone https://github.com/ZRF-ZRF/LVpocket.git
    cd LVpocket
   
## 3.  Python environment

    conda create -n lvpocket python=3.6.13
    conda activate lvpocket
    conda install numpy=1.19
    conda install tensorflow-gpu=1.3
    conda install keras=2.2
    conda install openbabel=3.1.1
    conda install -c cheminfIBB tfbio=0.3
    conda install scikit-image=0.17
    conda install scipy=1.5
    conda install tqdm
    conda install biopython

## 4.  Data preparation
We used scPDB dataset to train our model, you can get the dataset at this link:http://bioinfo-pharma.u-strasbg.fr/scPDB/. Before training, you need to build a training dataset using following code:

    python model/prepare_dataset.py --dataset /path/scpdb_path/ --output scpdb_dataset.hdf 
## 5.  Model training
You can training the model use the following code:

    python trian.py --input scpdb_data.hdf --output /output_path/

## 6.  The prediction of protein pockets
You can choose baseline model or SCOP fine-tuned models to predict protein binding pockets based on the protein sctructure.

### (1). The classification of protein structure

If you want to know which structure class your input protein is, you can use the protein structure classifier we built to do so.The usage process of this method is as follows:

1、	Install PSIPRED

2、	Use the pdb_to_fasta method to convert the PDB format file to FASTA format

3、	The PSIPRED tool was used to predict the secondary structure sequence of the protein

4、	Extract features based on the secondary structure sequence of the protein and make predictions

### (2).  Protein binding pockets prediction    

    python predict.py --input 1a26_out.pdb(mol2) --output 1a26_pre --model baseline_model.hdf --format pdb(mol2)
    
    --input The filepath of protein for prediction, we can deal with the format of '.pdb' or '.mol2';
    --output The output path of the prediction results;
    --model The model file to predict. We have five models, you can select one which is good for the input protein classification;
    --format The protein format of the input protein file('.pdb/.mol2)'.
    
    
