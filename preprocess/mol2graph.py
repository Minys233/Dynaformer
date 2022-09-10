from features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from rdkit import Chem
import numpy as np
from utils import read_mol

def mol2graph(mol: Chem.Mol):
    conformer = mol.GetConformer(0)
    # atoms
    atom_features_list, coords = [], []
    atom_map = dict()
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == "H": continue
        atom_features_list.append(atom_to_feature_vector(atom))
        coords.append(conformer.GetAtomPosition(atom.GetIdx()))
        atom_map[idx] = len(coords) - 1
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if mol.GetAtomWithIdx(i).GetSymbol() == "H": continue
            if mol.GetAtomWithIdx(j).GetSymbol() == "H": continue

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((atom_map[i], atom_map[j]))
            edge_features_list.append(edge_feature)
            edges_list.append((atom_map[j], atom_map[i]))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['coords'] = np.array(coords)

    return graph 


if __name__ == '__main__':
    graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    print(graph)