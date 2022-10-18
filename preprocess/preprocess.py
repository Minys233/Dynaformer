import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
from torch_geometric.data import Data
import torch
from rdkit import Chem
from mol2graph import mol2graph
from pymol import cmd
from ecif import GetECIF

SPATIAL_EDGE = [4, 0, 0]


def dim2(arr: np.ndarray):
    return len(arr.shape) == 2


def reindex(atom_idx: list, edge_index: np.ndarray):
    if len(edge_index.shape) != 2:
        return edge_index
    indexmap = {old: new for new, old in enumerate(atom_idx)}
    mapfunc = np.vectorize(indexmap.get)
    edge_index_new = np.array([mapfunc(edge_index[0]), mapfunc(edge_index[1])])
    return edge_index_new


def remove_duplicated_edges(ei: np.ndarray, ea: np.ndarray, ref_ei: np.ndarray):
    if not len(ea):
        return ei, ea
    existing_set = {(i, j) for i, j in zip(*ref_ei)}
    mask = []
    for i, j in zip(*ei):
        if (i, j) in existing_set:
            mask.append(False)  # delete
        else:
            mask.append(True)  # keep
    mask = np.array(mask, dtype=bool)
    ei_n = np.array([ei[0][mask], ei[1][mask]])
    ea_n = ea[mask]
    return ei_n, ea_n


def gen_feature(ligand: Chem.Mol, pocket: Chem.Mol, pdbid: str):
    ligand_dict = mol2graph(ligand)
    pocket_dict = mol2graph(pocket)
    ligand_coords, ligand_features, ligand_edge_index, ligand_edge_attr = ligand_dict['coords'], ligand_dict[
        'node_feat'], ligand_dict['edge_index'], ligand_dict['edge_feat']
    pocket_coords, pocket_features, pocket_edge_index, pocket_edge_attr = pocket_dict['coords'], pocket_dict[
        'node_feat'], pocket_dict['edge_index'], pocket_dict['edge_feat']

    # shape check
    if not (dim2(ligand_coords) and dim2(ligand_features) and dim2(ligand_edge_index) and dim2(ligand_edge_attr)):
        raise RuntimeError(f"Ligand feature shape error")
    if not (dim2(pocket_coords) and dim2(pocket_features) and dim2(pocket_edge_index) and dim2(pocket_edge_attr)):
        raise RuntimeError(f"Protein feature shape error")

    return {'lc': ligand_coords, 'lf': ligand_features, 'lei': ligand_edge_index, 'lea': ligand_edge_attr,
            'pc': pocket_coords, 'pf': pocket_features, 'pei': pocket_edge_index, 'pea': pocket_edge_attr,
            'pdbid': pdbid}


def gen_spatial_edge(dm: np.ndarray, spatial_cutoff: float = 5):
    if spatial_cutoff <= 0.1:
        return np.array([]), np.array([])
    src, dst = np.where((dm <= spatial_cutoff) & (dm > 0.1))
    # already symmetric!
    edge_index = [(x, y) for x, y in zip(src, dst)]
    edge_attr = np.array([SPATIAL_EDGE for _ in edge_index])

    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_attr, dtype=np.int64)
    return edge_index, edge_attr


def gen_ligpro_edge(dm: np.ndarray, pocket_cutoff: float):
    lig_num_atom, pro_num_atom = dm.shape
    lig_idx, pro_idx = np.where(dm <= pocket_cutoff)
    edge_index = [(x, y + lig_num_atom) for x, y in zip(lig_idx, pro_idx)]
    edge_index += [(y + lig_num_atom, x) for x, y in zip(lig_idx, pro_idx)]
    edge_attr = np.array([SPATIAL_EDGE for _ in edge_index])
    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_attr, dtype=np.int64)
    return edge_index, edge_attr


def gen_graph(ligand: tuple, pocket: tuple, name: str, protein_cutoff: float, pocket_cutoff: float,
              spatial_cutoff: float):
    lig_coord, lig_feat, lig_ei, lig_ea = ligand
    pro_coord, pro_feat, pro_ei, pro_ea = pocket

    assert len(lig_coord) == len(lig_feat)
    assert len(pro_coord) == len(pro_coord)

    # new pocket graph based on protein_cutoff (smaller than 10 A)
    assert protein_cutoff >= pocket_cutoff, \
        f"Protein cutoff {protein_cutoff} should be larger than pocket cutoff {pocket_cutoff}"
    assert pocket_cutoff >= spatial_cutoff, \
        f"Protein cutoff {protein_cutoff} should be larger than spatial cutoff {spatial_cutoff}"

    # select protein atoms within protein cutoff
    pro_atom_mask = np.zeros(len(pro_coord), dtype=bool)
    pro_atom_mask[np.where(cdist(lig_coord, pro_coord) <= protein_cutoff)[1]] = 1
    pro_edge_mask = np.array([True if pro_atom_mask[i] and pro_atom_mask[j] else False for i, j in zip(*pro_ei)])

    pro_coord = pro_coord[pro_atom_mask]
    pro_feat = pro_feat[pro_atom_mask]
    pro_ei = np.array([pro_ei[0, pro_edge_mask], pro_ei[1, pro_edge_mask]])
    pro_ea = pro_ea[pro_edge_mask]
    pro_ei = reindex(np.where(pro_atom_mask)[0], pro_ei)

    # add spatial edges based on spatial cutoff
    lig_dm = cdist(lig_coord, lig_coord)
    lig_sei, lig_sea = gen_spatial_edge(lig_dm, spatial_cutoff=spatial_cutoff)
    lig_sei, lig_sea = remove_duplicated_edges(lig_sei, lig_sea, lig_ei)

    pro_dm = cdist(pro_coord, pro_coord)
    pro_sei, pro_sea = gen_spatial_edge(pro_dm, spatial_cutoff=spatial_cutoff)
    pro_sei, pro_sea = remove_duplicated_edges(pro_sei, pro_sea, pro_ei)
    # add interaction edges based on pocket cutoff
    dm_lig_pro = cdist(lig_coord, pro_coord)
    lig_pock_ei, lig_pock_ea = gen_ligpro_edge(dm_lig_pro, pocket_cutoff=pocket_cutoff)
    # construct ligand-pocket graph
    comp_coord = np.vstack([lig_coord, pro_coord])
    comp_feat = np.vstack([lig_feat, pro_feat])
    comp_ei, comp_ea = lig_ei, lig_ea
    if len(pro_ei.shape) == 2 and len(pro_ei.T) >= 3:
        comp_ei = np.hstack([comp_ei, pro_ei + len(lig_feat)])
        comp_ea = np.vstack([comp_ea, pro_ea])
    if len(lig_sei.shape) == 2 and len(lig_sei.T) >= 3:
        comp_ei = np.hstack([comp_ei, lig_sei])
        comp_ea = np.vstack([comp_ea, lig_sea])
    if len(pro_sei.shape) == 2 and len(pro_sei.T) >= 3:
        comp_ei = np.hstack([comp_ei, pro_sei + len(lig_feat)])
        comp_ea = np.vstack([comp_ea, pro_sea])
    if len(lig_pock_ei.shape) == 2 and len(lig_pock_ei.T) >= 3:
        comp_ei = np.hstack([comp_ei, lig_pock_ei])
        comp_ea = np.vstack([comp_ea, lig_pock_ea])
    # comp_dist = np.array([euclidean(comp_coord[i], comp_coord[j]) for i, j in zip(*comp_ei)], dtype=np.float32)
    # TODO: sort edges
    comp_num_node = np.array([len(lig_feat), len(pro_feat)], dtype=np.int64)
    comp_num_edge = np.array(
        [lig_ei.T.shape[0], pro_ei.T.shape[0], lig_pock_ei.T.shape[0], lig_sei.T.shape[0], pro_sei.T.shape[0]],
        dtype=np.int64)
    return comp_coord, comp_feat, comp_ei, comp_ea, comp_num_node, comp_num_edge


def load_pk_data(data_path: Path):
    pdbid, pk = [], []
    for line in open(data_path):
        if line[0] == '#': continue
        elem = line.split()
        v1, _, _, v2 = elem[:4]
        pdbid.append(v1)
        pk.append(float(v2))
    res = {i: p for i, p in zip(pdbid, pk)}
    return res


def to_pyg_graph(raw: list, **kwargs):
    comp_coord, comp_feat, comp_ei, comp_ea, comp_num_node, comp_num_edge, rfscore, gbscore, ecif, pk, name = raw

    d = Data(x=torch.from_numpy(comp_feat).to(torch.long), edge_index=torch.from_numpy(comp_ei).to(torch.long), edge_attr=torch.from_numpy(comp_ea).to(torch.long),
             pos=torch.from_numpy(comp_coord).to(torch.float32), y=torch.tensor([pk], dtype=torch.float32), pdbid=name,
             num_node=torch.from_numpy(comp_num_node).to(torch.long), num_edge=torch.from_numpy(comp_num_edge).to(torch.long),
             rfscore=torch.from_numpy(rfscore).to(torch.float32), gbscore=torch.from_numpy(gbscore).to(torch.float32),
             ecif=torch.from_numpy(ecif).to(dtype=torch.float32),
             **kwargs)
    return d


def get_info(protein_file, ligand_file):
    cmd.reinitialize()
    cmd.load(protein_file, 'receptor')
    cmd.load(ligand_file, 'ligand')
    cmd.remove('sol.')
    cmd.h_add()
    proinfo = {"elem": [], "resn": [], "coord":[], }
    liginfo = {"elem": [], "resn": [], "coord":[], }
    cmd.iterate_state(1, 'receptor', 'info["elem"].append(elem); info["resn"].append(resn); info["coord"].append(np.array([x, y, z]))',space={"info": proinfo, "np": np})
    cmd.iterate_state(1, 'ligand', 'info["elem"].append(elem); info["resn"].append(resn); info["coord"].append(np.array([x, y, z]))',space={"info": liginfo, "np": np})
    for k in proinfo.keys():
        proinfo[k] = np.array(proinfo[k])
        liginfo[k] = np.array(liginfo[k])
    return proinfo, liginfo


def GB_score(lig_info: dict, pro_info: dict) -> np.ndarray:
    # 400 dim
    amino_acid_groups = [
    {"ARG", "LYS", "ASP", "GLU"},
    {"GLN", "ASN", "HIS", "SER", "THR", "CYS"},
    {"TRP", "TYR", "MET"},
    {"ILE", "LEU", "PHE", "VAL", "PRO", "GLY", "ALA"},
    ]
    elements = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    distmap = cdist(lig_info['coord'], pro_info['coord'])
    restype = np.zeros(len(pro_info['resn'])) - 1
    elem_mask = {k: pro_info["elem"] == k for k in elements}

    fp = np.zeros([len(elements), len(elements), len(amino_acid_groups)])

    for idx, r in enumerate(pro_info['resn']):
        if r in amino_acid_groups[0]: restype[idx] = 0
        if r in amino_acid_groups[1]: restype[idx] = 1
        if r in amino_acid_groups[2]: restype[idx] = 2
        if r in amino_acid_groups[3]: restype[idx] = 3

    for i, el in enumerate(elements):
        lmask = lig_info["elem"] == el
        if lmask.sum() < 1: continue
        for j, ep in enumerate(elements):
            pmask = elem_mask[ep]
            if pmask.sum() < 1: continue
            for k, rt in enumerate(range(4)):
                rt_mask = restype == rt
                m = pmask & rt_mask
                if m.sum() < 1: continue
                d = distmap[lmask][:, m]
                v = (1 / d[d<=12]).sum()
                fp[i, j, k] = v

    return fp.flatten()


def RF_score(lig_info: dict, pro_info: dict):
    LIG_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    PRO_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    distmap = cdist(lig_info['coord'], pro_info['coord'])
    fp = np.zeros([10, 10])
    for i, el in enumerate(LIG_TYPES):
        lmask = lig_info['elem'] == el
        if lmask.sum() < 1: continue
        for j, ep in enumerate(PRO_TYPES):
            pmask = pro_info['elem'] == ep
            d = distmap[lmask][:, pmask]
            v = d[d < 12].shape[0]
            fp[i, j] = v
    return fp.flatten()





