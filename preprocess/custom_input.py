import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
from preprocess import gen_feature, gen_graph, to_pyg_graph, get_info, RF_score, GB_score, GetECIF
from joblib import Parallel, delayed
from utils import read_mol, obabel_pdb2mol, pymol_pocket
import numpy as np
from rdkit import Chem, RDLogger
import tempfile
import pandas as pd
import os


def process_one(proteinpdb: Path, ligandsdf: Path, name: str, pk: float, protein_cutoff, pocket_cutoff, spatial_cutoff):
    RDLogger.DisableLog('rdApp.*')

    if not (proteinpdb.is_file() and ligandsdf.is_file()):
        print(f"{proteinpdb} or {ligandsdf} does not exist.")
        return None
    pocketpdb = proteinpdb.parent / (proteinpdb.name.rsplit('.', 1)[0] + '_pocket.pdb')
    pocketsdf = proteinpdb.parent / (proteinpdb.name.rsplit('.', 1)[0] + '_pocket.sdf')
    if not pocketpdb.is_file():
        pymol_pocket(proteinpdb, ligandsdf, pocketpdb)
    if not pocketsdf.is_file():
        obabel_pdb2mol(pocketpdb, pocketsdf)

    try:
        ligand = read_mol(ligandsdf)
        pocket = read_mol(pocketsdf)
        proinfo, liginfo = get_info(proteinpdb, ligandsdf)
        res = gen_feature(ligand, pocket, name)
        res['rfscore'] = RF_score(liginfo, proinfo)
        res['gbscore'] = GB_score(liginfo, proinfo)
        res['ecif'] = np.array(GetECIF(str(proteinpdb), str(ligandsdf)))
    except RuntimeError as e:
        print(proteinpdb, pocketsdf, ligandsdf, "Fail on reading molecule")
        return None

    ligand = (res['lc'], res['lf'], res['lei'], res['lea'])
    pocket = (res['pc'], res['pf'], res['pei'], res['pea'])
    try:
        raw = gen_graph(ligand, pocket, name, protein_cutoff=protein_cutoff, pocket_cutoff=pocket_cutoff, spatial_cutoff=spatial_cutoff)
    except ValueError as e:
        print(f"{name}: Error gen_graph from raw feature {str(e)}")
        return None
    graph = to_pyg_graph(list(raw) + [res['rfscore'], res['gbscore'], res['ecif'], pk, name], frame=-1, rmsd_lig=0.0, rmsd_pro=0.0)
    return graph




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_csv', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--protein_cutoff', type=float, default=5.)
    parser.add_argument('--pocket_cutoff', type=float, default=5.)
    parser.add_argument('--spatial_cutoff', type=float, default=5.)

    args = parser.parse_args()
    filedf = pd.read_csv(args.file_csv)
    receptors = filedf['receptor']
    ligands = filedf['ligand']
    names = filedf['name']
    pks = filedf['pk']
    graphs = Parallel(n_jobs=args.njobs)(delayed(process_one)(Path(rec), Path(lig), name, pk, args.protein_cutoff, args.pocket_cutoff, args.spatial_cutoff) for rec, lig, name, pk in zip(tqdm(receptors), ligands, names, pks))
    graphs = list(filter(None, graphs))
    pickle.dump(graphs, open(args.output, 'wb'))
    # for rec, lig in zip(receptors, ligands):
    #     rec, lig = Path(rec), Path(lig)
    #     g = process_one(rec, lig, args.protein_cutoff, args.pocket_cutoff, args.spatial_cutoff)
    #     print(g)
    #     break
