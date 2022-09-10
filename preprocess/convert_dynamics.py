import MDAnalysis as mda
import MDAnalysis.analysis.rms
from MDAnalysis.analysis import align
from pathlib import Path
import pandas as pd
import numpy as np
import re
from tqdm import tqdm, trange
from joblib import Parallel, delayed
import argparse
import utils


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("refined_path", type=Path)
parser.add_argument("core_path", type=Path)
parser.add_argument("save", type=Path)
parser.add_argument("--cutoff", type=float, default=10)


def load_filter_label(refined_path: Path, core_path: Path) -> dict:
    full = pd.read_csv(refined_path / 'index' / 'INDEX_refined_data.2019', header=None,
                       names=['pdbid', 'resolution', 'release_year', '-logKd/Ki',
                              'Kd/Ki', 'token', 'reference', 'ligand_name'],
                       delim_whitespace=True, comment='#')
    full['ligand_name'] = [s.strip("()") for s in full['ligand_name']]
    full.drop(['Kd/Ki', 'token', 'reference', ], axis=1, inplace=True)
    coreset = {d.name for d in core_path.iterdir() if len(d.name) == 4}
    ligand_name_mask = np.ones(len(full), dtype=bool)
    coreset_mask = np.ones(len(full), dtype=bool)

    for idx, line in full.iterrows():
        # currently only accept ligand name with 3-letter
        if len(line['ligand_name']) != 3:
            ligand_name_mask[idx] = False
        if line['ligand_name'] not in coreset:  # intention, coreset_mask = False for all, preprocess all
            coreset_mask[idx] = False
    valid = full[ligand_name_mask & (~coreset_mask)]
    valid = valid.sort_values(by='pdbid')
    res_dict = {pdbid: res for pdbid, res in zip(valid['pdbid'], valid['ligand_name'])}
    return res_dict


def psf_correct(md_root: Path, pdbid: str):
    # this function will fix the letter problem in in step3_input.psf
    # the new psf will be: step3_input_corrected.psf
    subdir = md_root / pdbid / "namd"
    with open(subdir / "step3_input.psf", 'r') as file:
        lines = file.readlines()
    new_files = []
    for i in lines[7:]:
        if not i[18:25].isnumeric() and len(i.split()) == 9:
            new = re.sub(r'\D', ' ', i[18:25])
            i = i.replace(i[18:25], new, 1)
        new_files.append(i)
    data = lines[:7]+new_files
    with open(subdir / "step3_input_corrected.psf", 'w') as f2:
        for line in data:
            f2.write(line)


def rms(u, ligname):
    trj = u.copy()
    ref = u.copy()
    _ = ref.trajectory[0]

    align.AlignTraj(trj, ref, select='protein and name CA', in_memory=True).run()
    frames, rms_D_lig, rms_D_pro = [], [], []
    ligand = ref.select_atoms(f"resname {ligname} and not name LP*", updating=True)
    ligand_segid = ligand.segids[0]
    if len(np.unique(ligand.segids)) > 1:
        ligand_segid = ligand.segids[0] if ligand.segids[0] != 'PROA' else ligand.segids[-1]
    for frame, ts in enumerate(trj.trajectory):
        _ = trj.trajectory[frame]
        trj_ligand = trj.select_atoms(f'segid {ligand_segid} and not (name H* ) and not name LP*')
        ref_ligand = ref.select_atoms(f'segid {ligand_segid} and not (name H* ) and not name LP*')
        rms_D_lig.append(mda.analysis.rms.rmsd(trj_ligand.atoms.positions, ref_ligand.atoms.positions, superposition=False))
        trj_pro = trj.select_atoms('protein and not (name H* )')
        ref_pro = ref.select_atoms('protein and not (name H* )')
        rms_D_pro.append(mda.analysis.rms.rmsd(trj_pro.atoms.positions, ref_pro.atoms.positions, superposition=False))
        frames.append(frame)
    ligand = trj.select_atoms(f'segid {ligand_segid} and not (name H* ) and not name LP*')
    rms_f = mda.analysis.rms.RMSF(ligand).run()
    rms_F_lig = rms_f.results.rmsf

    return frames, rms_D_lig, rms_F_lig, rms_D_pro


def psfdcd2pdb(md_root: Path, save_root: Path, pdbid: str, resname: str, dist_cutoff: float):
    import warnings
    warnings.simplefilter("ignore")
    psf_path = md_root / pdbid / 'namd' / "step3_input.psf"
    dcd_path = md_root / pdbid / 'namd' / "step5_production.dcd"
    logstr = []
    if (save_root / pdbid / "rmsd.csv").is_file():
        return logstr

    if not psf_path.is_file():
        logstr.append(f"{pdbid} has no psf file")
        return logstr
    if not dcd_path.is_file():
        logstr.append(f"{pdbid} has no dcd file")
        return logstr
    try:
        u = mda.Universe(str(psf_path), str(dcd_path))
    except ValueError as e:
        # print(f"{pdbid} - {resname}: need to correct this")
        psf_path = md_root / pdbid / 'namd' / "step3_input_corrected.psf"
        psf_correct(md_root, pdbid)
        u = mda.Universe(str(psf_path), str(dcd_path))

    guessed_elements = mda.topology.guessers.guess_types(u.atoms.names)
    u.add_TopologyAttr('elements', guessed_elements)
    # select ligand
    ligand = u.select_atoms(f"resname {resname}")
    try:
        ligand_segid = ligand.segids[0]
    except IndexError as e:
        logstr.append(f"{pdbid} cannot select ligand")
        return logstr
    if len(np.unique(ligand.segids)) > 1:
        ligand_unique_segid = ligand.segids[0] if ligand.segids[0] != 'PROA' else ligand.segids[-1]
        ligand_segid = ligand_unique_segid

    ligand = u.select_atoms(f'segid {ligand_segid} and not name LP*', updating=True)
    if len(ligand.atoms) <= 1:
        logstr.append(f"{pdbid} ligand empty")
        return logstr

    frames, rms_D_lig, rms_F_lig, rms_D_pro = rms(u, resname)
    assert len(frames) == len(rms_D_lig) == len(rms_D_pro) == len(u.trajectory), f"Check {pdbid}"
    df = pd.DataFrame({"frame": frames,
                       "rmsd_lig": rms_D_lig,
                       "rmsd_pro": rms_D_pro})

    if '/' in pdbid:
        pdbid = pdbid.replace("/", "__")

    (save_root / pdbid).mkdir(exist_ok=True, parents=True)
    df.to_csv(save_root / pdbid / "rmsd.csv", index=False)
    for frame, ts in enumerate(u.trajectory):
        ligand = u.select_atoms(f'segid {ligand_segid} and not name LP*', updating=True)
        pocket = u.select_atoms(
            f"(not resname WAT HOH SOL TIP3 and not segid IONS and not name LP*) and (around {dist_cutoff} segid {ligand_segid})",
            updating=True)
        protein = u.select_atoms(
            f"(not resname WAT HOH SOL TIP3 and not segid IONS and not name LP*) and (not segid {ligand_segid})",
            updating=True)
        try:
            ligand.write(save_root / f"{pdbid}" / f"{pdbid}_ligand_{frame}.pdb")
        except IndexError as e:
            logstr.append(f'{pdbid} {frame} ligand has no atom')
        try:
            pocket.write(save_root / f"{pdbid}" / f"{pdbid}_pocket_{frame}.pdb")
        except IndexError as e:
            logstr.append(f'{pdbid} {frame} pocket has no atom')

        try:
            protein.write(save_root / f"{pdbid}" / f"{pdbid}_protein_{frame}.pdb")
        except IndexError as e:
            logstr.append(f'{pdbid} {frame} protein has no atom')
    return logstr


def convert_helper(d):
    for frame in range(100):
        pro_pdb, lig_pdb = d / f"{d.name}_pocket_{frame}.pdb", d / f"{d.name}_ligand_{frame}.pdb"
        pro_sdf, lig_sdf = d / f"{d.name}_pocket_{frame}.sdf", d / f"{d.name}_ligand_{frame}.sdf"
        if pro_pdb.is_file() and not pro_sdf.is_file():
            utils.obabel_pdb2mol(pro_pdb, pro_sdf)
        if lig_pdb.is_file() and not lig_sdf.is_file():
            utils.obabel_pdb2mol(lig_pdb, lig_sdf)


if __name__ == "__main__":
    args = parser.parse_args()
    root, save, cutoff = args.root, args.save, args.cutoff
    refined_path, core_path = args.refined_path, args.core_path
    res_dict = load_filter_label(refined_path, core_path)

    dirlst = list(root.iterdir())

    res = Parallel(n_jobs=22)(delayed(psfdcd2pdb)(root, save, d.name, res_dict[d.name[:4]], cutoff)
                              for d in tqdm(dirlst, ncols=80, desc="Process psf/dcd") if d.name[:4] in res_dict)
    res = [i for i in res if i]
    for i in res:
        if not i: continue
        print(i)
    Parallel(n_jobs=20)(delayed(convert_helper)(d) for d in tqdm(list(save.iterdir()), ncols=80, desc="Convert to mol2"))




