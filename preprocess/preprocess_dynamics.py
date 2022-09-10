import argparse
from multiprocessing.sharedctypes import Value
from pathlib import Path
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import pickle
from preprocess import gen_feature, load_pk_data, gen_graph, to_pyg_graph, get_info, RF_score, GB_score, GetECIF
from joblib import Parallel, delayed
from utils import read_mol
import pandas as pd
import numpy as np


def parallel_helper(refined_path: Path, name: str):
    frames = []
    rmsd_df = pd.read_csv(refined_path / name / "rmsd.csv")
    for i in range(100):  #trange(100, ncols=80, desc="Reading sdf", leave=False):
        try:
            ligand = read_mol(refined_path / name / f"{name}_ligand_{i}.sdf")
            pocket = read_mol(refined_path / name / f"{name}_pocket_{i}.sdf")
            res = gen_feature(ligand, pocket, name)
            res['frame'] = i
            res['rmsd_lig'] = rmsd_df[rmsd_df['frame'] == i]['rmsd_lig'].item()
            res['rmsd_pro'] = rmsd_df[rmsd_df['frame'] == i]['rmsd_pro'].item()

            proinfo, liginfo = get_info(refined_path / name / f"{name}_protein_{i}.pdb",
                                        refined_path / name / f"{name}_ligand_{i}.sdf")
            res['rfscore'] = RF_score(liginfo, proinfo)
            res['gbscore'] = GB_score(liginfo, proinfo)
            res['ecif'] = np.array(GetECIF(str(refined_path / name / f"{name}_protein_{i}.pdb"),
                                           str(refined_path / name / f"{name}_ligand_{i}.sdf")))

        except RuntimeError as e:
            continue
        except OSError as e:
            continue
        frames.append(res)
    return frames


def process_dynamics(core_path: Path, refined_path: Path, md_path: Path, dataset_name: str, output_path: Path,
                    protein_cutoff: float, pocket_cutoff: float, spatial_cutoff: float, seed: int):
    core_set_list = [d.name for d in core_path.iterdir() if len(d.name) == 4]
    refined_set_list = [d.name for d in refined_path.iterdir() if len(d.name) == 4]
    md_set_list = [d.name for d in md_path.iterdir()]
    # load pka (binding affinity) data
    pk_file_gen = list((refined_path / 'index').glob('INDEX_general_PL_data.*'))[0]  # general set
    pk_file_ref = list((refined_path / 'index').glob('INDEX_refined_data.*'))[0]  # refined set
    print(f"Loading pk data from {pk_file_gen} and {pk_file_ref}")
    pk_dict = {**load_pk_data(pk_file_gen), **load_pk_data(pk_file_ref)}
    assert set(pk_dict.keys()).issuperset(refined_set_list)
    assert set(pk_dict.keys()).issuperset(core_set_list)
    # assert set(pk_dict.keys()).issuperset(md_set_list)
    # md_set_list = [i for i in md_set_list if i[:4] not in set(core_set_list)]
    print(f"Total {len(md_set_list)} md data")
    # atomic feature generation
    if not Path('./rawgraph.pkl').is_file():
        res = Parallel(n_jobs=22)(delayed(parallel_helper)(md_path, name) for name in tqdm(md_set_list, desc="Load refined", ncols=80))
        pickle.dump(res, open('./rawgraph.pkl', 'wb'))
    else:
        res = pickle.load(open('./rawgraph.pkl', 'rb'))

    processed_dict = {feat[0]['pdbid']: feat for feat in res}
    
    for pdbid in processed_dict:
        if pdbid[:4] not in pk_dict:
            print("pdbid:", pdbid, "not in pk_dict")
        pk = pk_dict[pdbid[:4]]
        for i in range(len(processed_dict[pdbid])):
                processed_dict[pdbid][i]['pk'] = pk

    refined_data, core_data, logstr = [], [], []
    for name, vv in tqdm(list(processed_dict.items()), desc="Construct graph", ncols=80):
        graphs = []
        for v in tqdm(vv, ncols=80, leave=False, desc="Frame"):
            ligand = (v['lc'], v['lf'], v['lei'], v['lea'])
            pocket = (v['pc'], v['pf'], v['pei'], v['pea'])
            try:
                raw = gen_graph(ligand, pocket, name,
                                protein_cutoff=protein_cutoff, pocket_cutoff=pocket_cutoff, spatial_cutoff=spatial_cutoff)
                edge_nums = raw[5]
                if edge_nums[1] <= 3:
                    raise ValueError("<4 protein edges (fewer nodes)")
                if edge_nums[2] <= 3:
                    raise ValueError("<4 protein-ligand edges (fewer nodes)")
            except ValueError as e:
                logstr.append(f"{name} - {v['frame']}: Error gen_graph from raw feature: {str(e)}")
                continue
            except IndexError as e:
                logstr.append(f"{name} - {v['frame']}: Error gen_graph from raw feature: {str(e)}")
                continue
            g = to_pyg_graph(list(raw) + [v['rfscore'], v['gbscore'], v['ecif'], v['pk'], name], frame=v['frame'], rmsd_lig=v['rmsd_lig'], rmsd_pro=v['rmsd_pro'])
            m = (g['pos'].unsqueeze(0) - g['pos'].unsqueeze(1)).norm(dim=-1).max()
            if m > 45:
                logstr.append(f"{name} - {v['frame']}: max dist {m:.3f}, maybe out of boundary")
                continue
            graphs.append(g)
        
        if name[:4] in core_set_list:
            core_data.append(graphs)
        else:
            refined_data.append(graphs)

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
    parser.add_argument('md_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--dataset_name', type=str, default='pdbbind')
    parser.add_argument('--protein_cutoff', type=float, default=5.)
    parser.add_argument('--pocket_cutoff', type=float, default=5.)
    parser.add_argument('--spatial_cutoff', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    process_dynamics(args.data_path_core, args.data_path_refined, args.md_path, args.dataset_name, args.output_path,
                    args.protein_cutoff, args.pocket_cutoff, args.spatial_cutoff, args.seed)
