# Dynafomer

A deep learning framework for encoding molecular dynamic features to enhance protein-ligand binding affinity prediction.


## Install and environment settings

### Install Anaconda
This code should be run on Linux PC/server with Anaconda environment. If you have not Anaconda installed, using the following script to install Miniconda, which is a mininal version of Anaconda. Refer to [Miniconda's website](https://docs.conda.io/en/latest/miniconda.html) for more information.

```bash
wget -O Miniconda_install.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda_install.sh
./Miniconda_install.sh
# follow the instructions to install
conda --version
# you should get the output like this
# conda 23.1.0
```

### Install dependencies
```bash
git clone --recursive https://github.com/Minys233/Dynafomer.git
cd Dynafomer/Dynaformer
conda create -n dynaformer python=3.9 -y
conda activate dynaformer
./install.sh
```
This will automatically install all the dependencies for evaluating or training the model.

## Using Dynaformer
The scripts in this section should be run under `dynaformer` conda environment created in the previous section.

### Evaluation checkpoints
Refer to `checkpoint/README.md` to download and unzip the checkpoints to the `checkpoint` folder. Then, run `run_evaluate.sh` script to evaluate three checkpoints on CASF-2013 and CASF-2016 benchmark dataset.

### Training & finetuning
Refer to `Dynaformer/examples/md_pretrain/md_train.sh` and `Dynaformer/examples/finetune/finetune.sh` to train from scratch or fintuning a pre-trained model.

### Custom input
Currently, for custom input, users will need to run two scripts for graph data preparation and evaluation. We provide examples that were used in case studies of the paper. Additional dependencies are required for processing data, and use this command to install them: `conda install -c conda-forge pymol-open-source openbabel -y` under the `dynaformer` environment.

1. Run `preprocess/custom_input.py` to convert protein PDB files and ligand SDF files into (PyG)[https://www.pyg.org/] graphs. 

Users need provide a CSV file with column `receptor`, `ligand`, `name` and `pk` for protein PDB file path, ligand SDF file path, custom name of the pair and the pk value as label. Also an output path is specified for store the graphs in pickle format. Note that the `pk` column is only used for calculating deviations/correlations between prediction and ground-truth, set it to -1 if you don't have it. Here is an example:

```bash
# Run this command in Dynaformer/preprocess directory
python custom_input.py ../example_data/example.csv ../example_data/example.pkl
```

The content of the `example.csv` file should be similar to the following template:
```csv
receptor,ligand,name,pk
../example_data/2v7a/2v7a_protein.pdb,../example_data/2v7a/2v7a_ligand.sdf,2v7a,8.30
../example_data/3udh/3udh_protein.pdb,../example_data/3udh/3udh_ligand.sdf,3udh,2.85
...(other lines)...
```

Note that the paths in `example.csv` use relative paths for demonstration purposes, but absolute paths are more recommended here to avoid file not exist errors.

2. Run `run_custom_input.sh` to predict the binding affinity using the model checkpoints.

```bash
# Run this command in Dynaformer (project root) directory
./run_custom_input.sh
```

The results should be stored in the `checkpoint` directory. Unlike PDBBind, for custom input files, we pick pockets on the fly rather than as input. We use `pymol` to select the pocket from protein, and use `Openbabel` to convert the pocket from PDB format to SDF format. Note: When using files from CASF as custom inputs, the predicted results will differ slightly from the `Evaluation checkpoints` section above. This difference is caused by the pocket files, i.e. CASF provides pocket files for each complex, but for custom input pocket files are generated at runtime.


## Known issues

1. `preprocess/custom_input.py`: When processing data in parallel, there may be competing data writes resulting in incorrect contents of the intermediate file (SDF), resulting in failure to output .pkl results.

2. Automatic evaluation with custom protein (PDB) & ligand (SDF) is under development. For now, need to use multiple scripts.

3. Training data release.