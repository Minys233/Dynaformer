
from pathlib import Path
from subprocess import run, DEVNULL
from rdkit import Chem
from rdkit.Chem import rdmolops


def read_mol(mol_file: Path):
    """
    For ligand, use sdf, for protein, use sdf converted from pdb by pymol
    """
    mol = Chem.MolFromMolFile(str(mol_file), sanitize=False, strictParsing=False)
    if mol is None:
        raise RuntimeError(f"{mol_file} cannot be processed")
    mol = correct_sanitize_v2(mol)
    # try:
    #     mol = neutralize_atoms(mol)
    # except Exception:
    #     pass
    return Chem.RemoveHs(mol, sanitize=False,)


def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def find_atom_bond_around(at: Chem.Atom, sym: str, bt: Chem.BondType, inring: bool, aromatic: bool):
    mol = at.GetOwningMol()
    res = []
    for n in at.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(at.GetIdx(), n.GetIdx())
        c1 = (n.GetSymbol() == sym) if sym is not None else True
        c2 = (bond.GetBondType() == bt) if bt is not None else True
        c3 = (n.IsInRing() == inring) if inring is not None else True
        c4 = (n.GetIsAromatic() == aromatic) if aromatic is not None else True
        if c1 and c2 and c3 and c4:
            res.append(n)
    return res

def fix_phosphoryl_group(at: Chem.Atom, mol: Chem.Mol):
    if at.GetSymbol() == 'P':
        double_bonded_o = find_atom_bond_around(at, "O", Chem.BondType.DOUBLE, False, False)
        if len(double_bonded_o) >= 2:
            mol.GetBondBetweenAtoms(at.GetIdx(), double_bonded_o[0].GetIdx()).SetBondType(Chem.BondType.SINGLE)
            double_bonded_o[0].SetFormalCharge(-1)
            at.SetFormalCharge(0)


def fix_carboxyl_group(at: Chem.Atom, mol: Chem.Mol):
    if at.GetSymbol() == 'C':
        double_bonded_o = find_atom_bond_around(at, "O", Chem.BondType.DOUBLE, False, False)
        single_bonded_o = find_atom_bond_around(at, "O", Chem.BondType.SINGLE, False, False)
        if len(double_bonded_o) == 2:
            mol.GetBondBetweenAtoms(at.GetIdx(), double_bonded_o[0].GetIdx()).SetBondType(Chem.BondType.SINGLE)
            double_bonded_o[0].SetFormalCharge(-1)
            at.SetFormalCharge(0)
        if len(double_bonded_o) == 1 and len(single_bonded_o) == 1 and at.GetFormalCharge() == -1:
            at.SetFormalCharge(0)
            single_bonded_o[0].SetFormalCharge(-1)
            

def fix_guanidine_amidine_group(at: Chem.Atom, mol: Chem.Mol):
    if at.GetSymbol() == 'C' and not at.IsInRing():
        double_bonded_n = find_atom_bond_around(at, "N", Chem.BondType.DOUBLE, None, False)
        single_bonded_n = find_atom_bond_around(at, "N", Chem.BondType.SINGLE, None, False)
        # free guanidine or amidine group, not in ring
        if len(double_bonded_n) in {2, 3}:
            # find first terminal N
            is_set = False
            for n in double_bonded_n:
                num_non_h = 0
                for a in n.GetNeighbors():
                    if a.GetSymbol() != "H": num_non_h += 1
                if num_non_h == 1 and not is_set:
                    is_set = True
                    mol.GetBondBetweenAtoms(at.GetIdx(), n.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                    n.SetFormalCharge(1)
                else:
                    mol.GetBondBetweenAtoms(at.GetIdx(), n.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                    n.SetFormalCharge(0)
            at.SetFormalCharge(0)
        if len(double_bonded_n) == 1 and len(single_bonded_n) in {1, 2}:
            at.SetFormalCharge(0)
            double_bonded_n[0].SetFormalCharge(1)


def fix_sulfonyl_group(at: Chem.Atom, mol: Chem.Mol):
    if at.GetSymbol() == 'S':
        double_bonded_o = find_atom_bond_around(at, "O", Chem.BondType.DOUBLE, False, False)
        if len(double_bonded_o) > 2:
            for o in double_bonded_o[2:]:
                mol.GetBondBetweenAtoms(at.GetIdx(), o.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                o.SetFormalCharge(-1)


def fix_guanine_group(at: Chem.Atom, mol: Chem.Mol):
    if at.GetSymbol() == 'C' and at.IsInRing():
        single_bonded_inring_n = find_atom_bond_around(at, "N", Chem.BondType.AROMATIC, True, True)
        double_bonded_nonring_n = find_atom_bond_around(at, "N", Chem.BondType.DOUBLE, False, False)
        if len(single_bonded_inring_n) in {1, 2} and len(double_bonded_nonring_n) == 1:
            # WHF?
            # mol.GetBondBetweenAtoms(at.GetIdx(), double_bonded_nonring_n[0].GetIdx()).SetBondType(Chem.BondType.SINGLE)
            # mol.GetBondBetweenAtoms(at.GetIdx(), single_bonded_inring_n[1].GetIdx()).SetBondType(Chem.BondType.DOUBLE)
            at.SetFormalCharge(0)
            double_bonded_nonring_n[0].SetFormalCharge(1)
            
            
def fix_aromatic_n_pocket(at: Chem.Atom, mol: Chem.Mol):
    if at.GetSymbol() == 'N':
        single_bonded = find_atom_bond_around(at, None, Chem.BondType.SINGLE, None, None)
        double_bonded = find_atom_bond_around(at, None, Chem.BondType.DOUBLE, None, None)
        aromatic_bonded = find_atom_bond_around(at, None, Chem.BondType.AROMATIC, None, None)
        if len(single_bonded) + 1.5 * len(aromatic_bonded) + 2 * len(double_bonded) == 4:
            at.SetFormalCharge(1)
        if len(single_bonded) + 1.5 * len(aromatic_bonded) + 2 * len(double_bonded) == 4 and len(double_bonded):
            mol.GetBondBetweenAtoms(at.GetIdx(), double_bonded[0].GetIdx()).SetBondType(Chem.BondType.SINGLE)
        
def fix_pocket_o_charge(at: Chem.Atom, mol: Chem.Mol):
    if at.GetSymbol() == 'O':
        single_bonded = find_atom_bond_around(at, None, Chem.BondType.SINGLE, None, None)
        double_bonded = find_atom_bond_around(at, None, Chem.BondType.DOUBLE, None, None)
        aromatic_bonded = find_atom_bond_around(at, None, Chem.BondType.AROMATIC, None, None)
        if len(single_bonded) + 1.5 * len(aromatic_bonded) + 2 * len(double_bonded) == 2:
            at.SetFormalCharge(0)
        if len(single_bonded) + 1.5 * len(aromatic_bonded) + 2 * len(double_bonded) == 3:
            at.SetFormalCharge(1)
        if len(single_bonded) + 1.5 * len(aromatic_bonded) + 2 * len(double_bonded) == 4:
            at.SetFormalCharge(2)

def fix_pocket_c_charge(at: Chem.Atom, mol: Chem.Mol):
    if at.GetSymbol() == 'C':
        single_bonded = find_atom_bond_around(at, None, Chem.BondType.SINGLE, None, None)
        double_bonded = find_atom_bond_around(at, None, Chem.BondType.DOUBLE, None, None)
        aromatic_bonded = find_atom_bond_around(at, None, Chem.BondType.AROMATIC, None, None)
        if len(single_bonded) + 1.5 * len(aromatic_bonded) + 2 * len(double_bonded) == 4 and at.GetFormalCharge() != 0:
            at.SetFormalCharge(0)
        if len(single_bonded) + 1.5 * len(aromatic_bonded) + 2 * len(double_bonded) > 4 and len(double_bonded):
            mol.GetBondBetweenAtoms(at.GetIdx(), double_bonded[0].GetIdx()).SetBondType(Chem.BondType.SINGLE)

def fix_non_ring_aromatic(mol):
    for atom in mol.GetAtoms():
        if not atom.IsInRing() and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() == Chem.BondType.AROMATIC and not (a1.IsInRing() and a2.IsInRing()):
            bond.SetBondType(Chem.BondType.SINGLE)


def get_ring_atoms(rings, idx_lst):
    for r in rings:
        if all([i in r for i in idx_lst]):
            return r
    return None

def all_aromatic(mol, idx_lst):
    return all([mol.GetAtomWithIdx(i).GetIsAromatic() for i in idx_lst]) if idx_lst is not None else False

def correct_sanitize_v2(mol):
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
        return mol
    except:
        pass
    # partial sanitize, or valence is not available
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True,)
    rinfo = mol.GetRingInfo().AtomRings()
    
    # COO
    frag1 = Chem.MolFromSmarts('[C;-1](=O)(-,=O)')
    if mol.HasSubstructMatch(frag1):
        
        hits = mol.GetSubstructMatches(frag1)
        for hit in hits:
            c, o1 = mol.GetAtomWithIdx(hit[0]), mol.GetAtomWithIdx(hit[1])
            c.SetFormalCharge(0)
            o1.SetFormalCharge(-1)
            mol.GetBondBetweenAtoms(hit[0], hit[1]).SetBondType(Chem.BondType.SINGLE)
    # rare case
    # frag11 = Chem.MolFromSmarts('[C;v5;-1](=O)(=O)')
    
    
    # -C(=N)N
    frag2 = Chem.MolFromSmarts('[NX3;!H0;!R]=[CX3;+1;!R]([!N])=[NX3;!H0;!R]')
    if mol.HasSubstructMatch(frag2):
        
        hits = mol.GetSubstructMatches(frag2)
        for hit in hits:
            n, c = mol.GetAtomWithIdx(hit[0]), mol.GetAtomWithIdx(hit[1])
            n.SetFormalCharge(1)
            c.SetFormalCharge(0)
            mol.GetBondBetweenAtoms(hit[1], hit[3]).SetBondType(Chem.BondType.SINGLE)
            
    # N=C(=N)=N
    frag3 = Chem.MolFromSmarts('[NX3;!H0;!R]=[CX3;+1;!R](=[NX3])=[NX3]')
    if mol.HasSubstructMatch(frag3):
        
        hits = mol.GetSubstructMatches(frag3)
        for hit in hits:
            n1, c, n2, n3 = mol.GetAtomWithIdx(hit[0]), mol.GetAtomWithIdx(hit[1]), mol.GetAtomWithIdx(hit[2]), mol.GetAtomWithIdx(hit[3])
            c.SetFormalCharge(0)
            n1.SetFormalCharge(1)
            mol.GetBondBetweenAtoms(c.GetIdx(), n2.GetIdx()).SetBondType(Chem.BondType.SINGLE)
            mol.GetBondBetweenAtoms(c.GetIdx(), n3.GetIdx()).SetBondType(Chem.BondType.SINGLE)
    
    frag9 = Chem.MolFromSmarts('[NX3;!H0;!R]=[CX3+1,cX3+1](~[!#7])-,=,:[NX3,nX3]')
    if mol.HasSubstructMatch(frag9):
        
        hits = mol.GetSubstructMatches(frag9)
        for hit in hits:
            n1, c, n2 = mol.GetAtomWithIdx(hit[0]), mol.GetAtomWithIdx(hit[1]), mol.GetAtomWithIdx(hit[3])
            c.SetFormalCharge(0)
            n1.SetIsAromatic(False)
            mol.GetBondBetweenAtoms(c.GetIdx(), n1.GetIdx()).SetBondType(Chem.BondType.SINGLE)
            mol.GetBondBetweenAtoms(c.GetIdx(), n2.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
            n2.SetFormalCharge(1)
            
            
    
    
    frag7 = Chem.MolFromSmarts('[NX3;!H0;!R]=[CX3+1,cX3+1](-,=,:[NX3,nX3,NX2,nX2])-,=,:[NX3,nX3]')
    if mol.HasSubstructMatch(frag7):
        hits = mol.GetSubstructMatches(frag7)
        for hit in hits:
            n1, c, n2, n3 = mol.GetAtomWithIdx(hit[0]), mol.GetAtomWithIdx(hit[1]), mol.GetAtomWithIdx(hit[2]), mol.GetAtomWithIdx(hit[3])
            c.SetFormalCharge(0)
            ring_idx = get_ring_atoms(rinfo, hit[1:])
            if all_aromatic(mol, ring_idx):
                mol.GetBondBetweenAtoms(c.GetIdx(), n1.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(c.GetIdx(), n2.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                # mol.GetBondBetweenAtoms(c.GetIdx(), n3.GetIdx()).SetBondType(Chem.BondType.SINGLE)
            else:
                # c.SetIsAromatic(False)
                n1.SetIsAromatic(False)
                mol.GetBondBetweenAtoms(c.GetIdx(), n1.GetIdx()).SetBondType(Chem.BondType.DOUBLE)        
                mol.GetBondBetweenAtoms(c.GetIdx(), n2.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(c.GetIdx(), n3.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                n1.SetFormalCharge(1)
    
    
    
    
    frag5 = Chem.MolFromSmarts('[NX3v4;H0;+0]')
    if mol.HasSubstructMatch(frag5):
        
        hits = mol.GetSubstructMatches(frag5)
        for hit in hits:
            n = mol.GetAtomWithIdx(hit[0])
            n.SetFormalCharge(1)
    
    frag6 = Chem.MolFromSmarts('*:[nX3r6;+0](-,=[!O;!S]):*')
    if mol.HasSubstructMatch(frag6):
        
        hits = mol.GetSubstructMatches(frag6)
        for hit in hits:
            n = mol.GetAtomWithIdx(hit[1])
            n.SetFormalCharge(1)
    
    frag7 = Chem.MolFromSmarts('*:[nX3r6;+0](-,=[O,S]):*')
    if mol.HasSubstructMatch(frag7):
        
        hits = mol.GetSubstructMatches(frag7)
        for hit in hits:
            n, o = mol.GetAtomWithIdx(hit[1]), mol.GetAtomWithIdx(hit[2])
            n.SetFormalCharge(1)
            if "H" not in [a.GetSymbol() for a in o.GetNeighbors()]:
                o.SetFormalCharge(-1)
            mol.GetBondBetweenAtoms(n.GetIdx(), o.GetIdx()).SetBondType(Chem.BondType.SINGLE)
    
    
    frag8 = Chem.MolFromSmarts('[NR+0,nR+0,CR+0,cR+0]-,=,:[NR+0,nR+0](:[CR+0,cR+0]1)-,=,:[CR+0,cR+0,NR+0,nR+0][CR,cR,NR,nR][CR,cR,NR,nR][CR,cR,NR,nR]1')
    if mol.HasSubstructMatch(frag8):
        
        hits = mol.GetSubstructMatches(frag8)
        for hit in hits:
            n = mol.GetAtomWithIdx(hit[1])
            n.SetFormalCharge(1)

    # SO3
    frag4 = Chem.MolFromSmarts('[SX4;v7](=[OX1])(=[OX1])(=[OX1])')
    if mol.HasSubstructMatch(frag4):
        
        hits = mol.GetSubstructMatches(frag4)
        for hit in hits:
            s, o = mol.GetAtomWithIdx(hit[0]), mol.GetAtomWithIdx(hit[1])
            o.SetFormalCharge(-1)
            mol.GetBondBetweenAtoms(s.GetIdx(), o.GetIdx()).SetBondType(Chem.BondType.SINGLE)
    
    frag11 = Chem.MolFromSmarts('[CX3]1[NX2][CX3][CX3][CX3]1')
    if mol.HasSubstructMatch(frag11):
        hits = mol.GetSubstructMatches(frag11)
        for hit in hits:
            n = mol.GetAtomWithIdx(hit[1])
            n.SetNumExplicitHs(1)
    
    frag12 = Chem.MolFromSmarts('[NX2]1[CX3][NX2][CX3][CX3]1')
    if mol.HasSubstructMatch(frag12):
        hits = mol.GetSubstructMatches(frag12)
        for hit in hits:
            n = mol.GetAtomWithIdx(hit[0])
            n.SetNumExplicitHs(1)
    
    frag13 = Chem.MolFromSmarts('[NX2]1[NX2][CX3][CX3][CX3]1')
    if mol.HasSubstructMatch(frag13):
        hits = mol.GetSubstructMatches(frag13)
        for hit in hits:
            n = mol.GetAtomWithIdx(hit[0])
            n.SetNumExplicitHs(1)
    
    frag14 = Chem.MolFromSmarts('[NX3R][CX3R](=[N!R])=,:[NR]')
    if mol.HasSubstructMatch(frag14):
        hits = mol.GetSubstructMatches(frag14)
        for hit in hits:
            c, n = mol.GetAtomWithIdx(hit[1]), mol.GetAtomWithIdx(hit[2])
            mol.GetBondBetweenAtoms(hit[1], hit[2]).SetBondType(Chem.BondType.SINGLE)
            n.SetNumExplicitHs(n.GetNumExplicitHs()+1)
    
    frag15 = Chem.MolFromSmarts('[CX3R][NX3R]1[NX2R,CX3R][CX3R][CX3R][CX3R]1')
    if mol.HasSubstructMatch(frag15):
        hits = mol.GetSubstructMatches(frag15)
        for hit in hits:
            n, nc = mol.GetAtomWithIdx(hit[1]), mol.GetAtomWithIdx(hit[2])
            n.SetFormalCharge(1)
            if nc.GetSymbol() == "N":
                nc.SetNumExplicitHs(1)
    
    frag16 = Chem.MolFromSmarts('[NX3R+0]1[CX3R][CX3R][CX3R][CX3R][CX3R]1')
    if mol.HasSubstructMatch(frag16):
        hits = mol.GetSubstructMatches(frag16)
        for hit in hits:
            n = mol.GetAtomWithIdx(hit[0])
            n.SetFormalCharge(1)
            
    
    frag10 = Chem.MolFromSmarts('[CX4r6]')
    if mol.HasSubstructMatch(frag10):
        
        hits = mol.GetSubstructMatches(frag10)
        for hit in hits:
            ring_idx = get_ring_atoms(rinfo, hit)
            ring_idx_set = set(ring_idx)
            for r in rinfo:
                if set(r) != set(ring_idx):
                    ring_idx_set = ring_idx_set - set(r)
            for idx in ring_idx_set:
                mol.GetAtomWithIdx(idx).SetIsAromatic(False)
    
    try:
        Chem.SanitizeMol(mol)
    # input file error
    except Chem.AtomKekulizeException:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Chem.KekulizeException:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Chem.AtomValenceException:
        # https://sourceforge.net/p/rdkit/mailman/message/32599798/
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
            Chem.SanitizeFlags.SANITIZE_KEKULIZE|
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
            Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
            catchErrors=True,)
    return mol



def correct_sanitize_v1(mol: Chem.Mol):
    for at in mol.GetAtoms():
        fix_phosphoryl_group(at, mol)
        fix_carboxyl_group(at, mol)
        fix_guanidine_amidine_group(at, mol)
        fix_sulfonyl_group(at, mol)
        fix_guanine_group(at, mol)
        fix_aromatic_n_pocket(at, mol)
        fix_pocket_o_charge(at, mol)
        fix_pocket_c_charge(at, mol)
    fix_non_ring_aromatic(mol)
    try:
        Chem.SanitizeMol(mol)
    # input file error
    except Chem.KekulizeException:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Chem.AtomValenceException:
        # https://sourceforge.net/p/rdkit/mailman/message/32599798/
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
            Chem.SanitizeFlags.SANITIZE_KEKULIZE|
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
            Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
            catchErrors=True,)
    return mol


def pymol_convert(in_file: Path, out_file: Path):
    import pymol
    pymol.cmd.reinitialize()
    pymol.cmd.load(f"{str(in_file)}")
    # pymol.cmd.remove("hydrogens")
    pymol.cmd.h_add()
    pymol.cmd.save(f"{str(out_file)}", "not sol.")


def obabel_pdb2mol(in_file: Path, out_file: Path):
    run(['obabel', '-ipdb', str(in_file), '-omol', f'-O{str(out_file)}', '-x3v', '-h', '--partialcharge', 'eem'], check=True)#, stdout=DEVNULL, stderr=DEVNULL)


def obabel_sdf2mol(in_file: Path, out_file: Path):
    run(['obabel', '-isdf', str(in_file), '-omol', f'-O{str(out_file)}', '-x3v', '-h', '--partialcharge', 'eem'], check=True)#, stdout=DEVNULL, stderr=DEVNULL)


def obabel_mol22mol(in_file: Path, out_file: Path):
    run(['obabel', '-imol2', str(in_file), '-omol', f'-O{str(out_file)}', '-x3v', '-h', '--partialcharge', 'eem'], check=True)#, stdout=DEVNULL, stderr=DEVNULL)


if __name__ == "__main__":
    # from sys import stderr
    from preprocess import gen_feature
    path = Path("/mnt/yaosen-data/PDBBind/refined-set-2019")
    d = path / '2epn'
    print(d.name)
    # if len(d.name) != "2epn": continue
    ligand = read_mol(d / f'{d.name}_ligand.sdf')
    pocket = read_mol(d / f"{d.name}_pocket.sdf")
    res = gen_feature(ligand, pocket, d.name)
    for atom in ligand.GetAtoms():
        if atom.GetSymbol() == "H":
            print("HHH")
    for atom in pocket.GetAtoms():
        if atom.GetSymbol() == "H":
            print("HHH")
                
