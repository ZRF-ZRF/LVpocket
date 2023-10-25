## LVpocket
  
   We propose a deep learning model- LVpocket, which combines the strength of the V-NET model and the ResNet model to predict protein-ligand binding pockets, which could solve the problems of imbalanced sample data, gradient disappearance, and explosion during training. Specially, we preprocess the data from scPDB and divide the proteins into four classes: all α, all β, α + β, α / β, which we use to fine-tune four models which have different hyperparameters based on LVPocket to customize a predictive model for each classification protein structure. The results of our model on three independent data sets highlight its better performance over current state-of-the-art methods and its ability to generalize novel protein structures well.



#  Usage

## 1. Download the model files

   Links：https://pan.baidu.com/s/1kOUircaFHDdapegPvhzH3w 
  
   Code：lvlv

## 2.	 Clone this repository

    git clone https://github.com/ZRF-ZRF/LVpocket.git
    cd LVpocket
   
## 3.   Python environment

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

## 4.   Prediction

    python predict.py --input 1a26_out.pdb(mol2) --output 1a26_pre --model all_scPDB_model.hdf --format pdb(mol2)
    
    --input The filepath of protein for prediction, we can deal with the format of '.pdb' or '.mol2';
    --output The output path of the prediction results;
    --model The model file to predict. We have five models, you can select one which is good for the input protein classification;
    --format The protein format of the input protein file('.pdb/.mol2)'.
    
    
