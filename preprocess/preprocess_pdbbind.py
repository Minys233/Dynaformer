import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from preprocess import gen_feature, load_pk_data, gen_graph, to_pyg_graph, get_info, RF_score, GB_score, GetECIF
from joblib import Parallel, delayed
from utils import read_mol
import numpy as np


def parallel_helper(refined_path: Path, name: str):
    try:
        ligand = read_mol(refined_path / name / f"{name}_ligand.sdf")
        pocket = read_mol(refined_path / name / f"{name}_pocket.sdf")
        proinfo, liginfo = get_info(refined_path / name / f"{name}_protein.pdb",
                                    refined_path / name / f"{name}_ligand.sdf")
        res = gen_feature(ligand, pocket, name)
        res['rfscore'] = RF_score(liginfo, proinfo)
        res['gbscore'] = GB_score(liginfo, proinfo)
        res['ecif'] = np.array(GetECIF(str(refined_path / name / f"{name}_protein.pdb"), str(refined_path / name / f"{name}_ligand.sdf")))

    except RuntimeError as e:
        return None
    return res


def process_pdbbind(core_path: Path, refined_path: Path, dataset_name: str, output_path: Path,
                    protein_cutoff: float, pocket_cutoff: float, spatial_cutoff: float, seed: int):
    core_set_list = [d.name for d in core_path.iterdir() if len(d.name) == 4]
    refined_set_list = [d.name for d in refined_path.iterdir() if len(d.name) == 4]

    # load pka (binding affinity) data
    pk_file_gen = list((refined_path / 'index').glob('INDEX_general_PL_data.*'))[0]  # general set
    pk_file_ref = list((refined_path / 'index').glob('INDEX_refined_data.*'))[0]  # refined set
    print(f"Loading pk data from {pk_file_gen} and {pk_file_ref}")
    pk_dict = {**load_pk_data(pk_file_gen), **load_pk_data(pk_file_ref)}
    assert set(pk_dict.keys()).issuperset(refined_set_list)
    assert set(pk_dict.keys()).issuperset(core_set_list)
    # atomic feature generation
    res = Parallel(n_jobs=24)(delayed(parallel_helper)(core_path, name) for name in tqdm(core_set_list, desc="Load core", ncols=80))
    res += Parallel(n_jobs=24)(delayed(parallel_helper)(refined_path, name) for name in tqdm(refined_set_list, desc="Load refined", ncols=80))
    res = [r for r in res if r is not None]
    processed_dict = {feat['pdbid']: feat for feat in res}

    for k in pk_dict:
        # some complex cannot be processed
        if k in processed_dict:
            processed_dict[k]['pk'] = pk_dict[k]

    refined_data, core_data, logstr = [], [], []
    for name, v in tqdm(processed_dict.items(), desc="Construct graph", ncols=80):
        ligand = (v['lc'], v['lf'], v['lei'], v['lea'])
        pocket = (v['pc'], v['pf'], v['pei'], v['pea'])
        try:
            raw = gen_graph(ligand, pocket, name,
                            protein_cutoff=protein_cutoff, pocket_cutoff=pocket_cutoff, spatial_cutoff=spatial_cutoff)
        except ValueError as e:
            logstr.append(f"{name}: Error gen_graph from raw feature {str(e)}")
            continue
        graph = to_pyg_graph(list(raw) + [v['rfscore'], v['gbscore'], v['ecif'], v['pk'], name+'_pdbbind'], frame=-1, rmsd_lig=0.0, rmsd_pro=0.0)
        if name[:4] in core_set_list:
            core_data.append(graph)
        else:
            refined_data.append(graph)

    print(f"Got {len(refined_data)} / {len(refined_set_list)} graphs for train and val, {len(core_data)} / {len(core_set_list)} for test")
    # split train and valid
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{dataset_name}_train_val.pkl", 'wb') as f:
        pickle.dump(refined_data, f)

    with open(output_path / f"{dataset_name}_test.pkl", 'wb') as f:
        pickle.dump(core_data, f)
    print(*logstr, sep='\n')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path_core', type=Path)
    parser.add_argument('data_path_refined', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--dataset_name', type=str, default='pdbbind')
    parser.add_argument('--protein_cutoff', type=float, default=5.)
    parser.add_argument('--pocket_cutoff', type=float, default=5.)
    parser.add_argument('--spatial_cutoff', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    process_pdbbind(args.data_path_core, args.data_path_refined, args.dataset_name, args.output_path,
                    args.protein_cutoff, args.pocket_cutoff, args.spatial_cutoff, args.seed)

