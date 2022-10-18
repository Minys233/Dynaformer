import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
from preprocess import gen_feature, gen_graph, to_pyg_graph, get_info, RF_score, GB_score, GetECIF
from joblib import Parallel, delayed
from utils import read_mol, obabel_pdb2mol, pymol_pocket, correct_sanitize_v2
import numpy as np
from rdkit import Chem, RDLogger
import tempfile
import pandas as pd
import os



def parallel_helper(proteinpdb, ligandsdf, name_prefix, mol, pdict, protein_cutoff, pocket_cutoff, spatial_cutoff):
    RDLogger.DisableLog('rdApp.*')
    _, templigand = tempfile.mkstemp(suffix='.sdf')
    os.close(_)
    _, temppocketpdb = tempfile.mkstemp(suffix='.pdb')
    os.close(_)
    _, temppocketsdf = tempfile.mkstemp(suffix='.sdf')
    os.close(_)
    pymol_pocket(proteinpdb, ligandsdf, temppocketpdb)
    obabel_pdb2mol(temppocketpdb, temppocketsdf)
    assert "_Name" in pdict, f'Property dict should have _Name key, but currently: {pdict}'
    name = name_prefix + f'_{pdict["_Name"]}'
    try:
        ligand = correct_sanitize_v2(mol)
        Chem.MolToMolFile(ligand, templigand)
        pocket = read_mol(temppocketsdf)
        proinfo, liginfo = get_info(proteinpdb, templigand)
        res = gen_feature(ligand, pocket, name)
        res['rfscore'] = RF_score(liginfo, proinfo)
        res['gbscore'] = GB_score(liginfo, proinfo)
        res['ecif'] = np.array(GetECIF(str(proteinpdb), str(templigand)))
    except RuntimeError as e:
        print(proteinpdb, temppocketsdf, templigand, "Fail on reading molecule")
        return None

    ligand = (res['lc'], res['lf'], res['lei'], res['lea'])
    pocket = (res['pc'], res['pf'], res['pei'], res['pea'])
    try:
        raw = gen_graph(ligand, pocket, name, protein_cutoff=protein_cutoff, pocket_cutoff=pocket_cutoff, spatial_cutoff=spatial_cutoff)
    except ValueError as e:
        print(f"{name}: Error gen_graph from raw feature {str(e)}")
        return None
    graph = to_pyg_graph(list(raw) + [res['rfscore'], res['gbscore'], res['ecif'], -1, name], frame=-1, rmsd_lig=0.0, rmsd_pro=0.0)
    os.remove(templigand)
    os.remove(temppocketpdb)
    os.remove(temppocketsdf)
    return graph


def process_complex(proteinpdb: Path, ligandsdf: Path, name_prefix: str, njobs: int, protein_cutoff, pocket_cutoff, spatial_cutoff):
    suppl = Chem.SDMolSupplier(str(ligandsdf), sanitize=False, strictParsing=False)
    mols = list(suppl)
    graphs = []

    res = Parallel(n_jobs=njobs)(delayed(parallel_helper)(proteinpdb, ligandsdf, f"{idx}_{name_prefix}", mol, mol.GetPropsAsDict(True), protein_cutoff, pocket_cutoff, spatial_cutoff) for idx, mol in enumerate(mols))
    for i in res:
        if i: graphs.append(i)
    return graphs




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('protein', type=Path)
    parser.add_argument('ligand', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--protein_cutoff', type=float, default=5.)
    parser.add_argument('--pocket_cutoff', type=float, default=5.)
    parser.add_argument('--spatial_cutoff', type=float, default=5.)

    args = parser.parse_args()

    if args.protein.name.split('.')[-1] != 'pdb': raise ValueError('Make sure your protein file is in pdb format.')
    if args.ligand.name.split('.')[-1] not in ['sdf', 'mol']: raise ValueError("Make sure your ligand file is in sdf/mol format.")
    
    protein, ligand, output = args.protein, args.ligand, args.output

    name_prefix = protein.name.rsplit('.', 1)[0]
    graphs = process_complex(protein, ligand, name_prefix, args.njobs, args.protein_cutoff, args.pocket_cutoff, args.spatial_cutoff)
    with open(args.output, 'wb') as f:
        pickle.dump(graphs, f)



