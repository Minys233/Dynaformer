import utils
from pathlib import Path
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("pdbbind_dir", type=Path)
args = parser.parse_args()


dirlst = [d for d in args.pdbbind_dir.iterdir() if len(d.name)==4]
for d in tqdm(dirlst, ncols=80):
    utils.obabel_pdb2mol(d / f"{d.name}_pocket.pdb", d / f"{d.name}_pocket.sdf")
    # utils.obabel_sdf2mol(d / f"{d.name}_ligand.sdf", d / f"{d.name}_ligand.mol")
    # utils.obabel_mol22mol(d / f"{d.name}_ligand.mol2", d / f"{d.name}_ligand.mol")
    # utils.pymol_convert(d / f"{d.name}_pocket.pdb", d / f"{d.name}_pocket.sdf")
    # utils.pymol_convert(d / f"{d.name}_ligand.mol2", d / f"{d.name}_ligand.mol")


