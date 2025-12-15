import numpy as np


def posreader(PosName='POSCAR'):
    """
    Read VASP POSCAR file into dictionary format.
    
    Args:
        PosName: Path to POSCAR file
    
    Returns:
        POS: Dictionary containing structure information
            - 'CellName': Structure name
            - 'LattConst': Lattice constant
            - 'Base': 3x3 lattice vectors
            - 'EleName': List of element names
            - 'AtomNum': List of atom counts per element
            - 'AtomSum': Total number of atoms
            - 'LattPnt': List of fractional coordinates
            - 'LatType': 'Direct' or 'Cartesian'
            - 'IsSel': 1 if selective dynamics, 0 otherwise
            - 'SelMat': Selective dynamics flags (if IsSel=1)
    """
    POS = {}
    with open(PosName, 'r') as Fid:
        POS['CellName'] = Fid.readline().strip()
        POS['LattConst'] = float(Fid.readline().split()[0])
        
        # Read lattice vectors
        POS['Base'] = []
        for _ in range(3):
            line = Fid.readline().split()
            POS['Base'].append([float(x) * POS['LattConst'] for x in line[:3]])
        
        # Element names and counts
        POS['EleName'] = Fid.readline().split()
        POS['EleNum'] = len(POS['EleName'])
        
        atom_nums = Fid.readline().split()
        POS['AtomNum'] = [int(x) for x in atom_nums]
        POS['AtomSum'] = sum(POS['AtomNum'])
        
        # Check for selective dynamics
        line = Fid.readline().split()
        FL = line[0][0].upper()
        
        if FL == 'S':
            POS['IsSel'] = 1
            POS['SelMat'] = [['X']*3 for _ in range(POS['AtomSum'])]
            line = Fid.readline().split()
            FL = line[0][0].upper()
        else:
            POS['IsSel'] = 0
        
        # Coordinate type and positions
        POS['LatType'] = 'Direct' if FL == 'D' else 'Cartesian'
        POS['LattPnt'] = []
        
        if POS['LatType'] == 'Direct':
            for i in range(POS['AtomSum']):
                line = Fid.readline().split()
                POS['LattPnt'].append([float(line[j]) for j in range(3)])
                if POS['IsSel']:
                    POS['SelMat'][i] = [line[j+3] for j in range(3)]
        else:
            # Convert Cartesian to Direct
            BaseInv = np.linalg.inv(POS['Base'])
            for i in range(POS['AtomSum']):
                line = Fid.readline().split()
                cart_coord = [float(line[j]) for j in range(3)]
                POS['LattPnt'].append(list(np.dot(BaseInv, cart_coord)))
                if POS['IsSel']:
                    POS['SelMat'][i] = [line[j+3] for j in range(3)]
    
    return POS


def poswriter(PosName, POS):
    """
    Write structure dictionary to VASP POSCAR file.
    
    Args:
        PosName: Output file path
        POS: Structure dictionary from posreader
    """
    with open(PosName, 'w') as Fid:
        Fid.write(f"{POS['CellName']}\n")
        Fid.write(f"{POS['LattConst']:.6f}\n")
        
        for base_vec in POS['Base']:
            Fid.write(f"{base_vec[0]:.6f} {base_vec[1]:.6f} {base_vec[2]:.6f}\n")
        
        Fid.write(' '.join(POS['EleName']) + '\n')
        Fid.write(' '.join(map(str, POS['AtomNum'])) + '\n')
        
        if POS['IsSel']:
            Fid.write('Selective Dynamics\n')
        
        Fid.write(f"{POS['LatType']}\n")
        
        for i, pnt in enumerate(POS['LattPnt']):
            Fid.write(f"{pnt[0]:.6f} {pnt[1]:.6f} {pnt[2]:.6f}")
            if POS['IsSel']:
                Fid.write(f" {POS['SelMat'][i][0]} {POS['SelMat'][i][1]} {POS['SelMat'][i][2]}")
            Fid.write('\n')


def dismatcreate(POS):
    """
    Create distance matrix with periodic boundary conditions.
    
    For each pair of atoms, computes the minimum image distance
    considering periodic boundary conditions.
    
    Args:
        POS: Structure dictionary from posreader
    
    Returns:
        POS: Same dictionary with added 'dismat' field (NxN distance matrix)
    """
    N = POS['AtomSum']
    POS['dismat'] = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            # Fractional coordinate difference
            delta = np.array(POS['LattPnt'][i]) - np.array(POS['LattPnt'][j])
            
            # Apply minimum image convention
            delta = np.where(delta > 0.5, delta - 1, delta)
            delta = np.where(delta <= -0.5, delta + 1, delta)
            delta = np.abs(delta)
            
            # Convert to Cartesian and compute distance
            cart_delta = np.dot(delta, POS['Base'])
            POS['dismat'][i][j] = np.linalg.norm(cart_delta)
    
    return POS


def dismatswap(dismat, Ind1, Ind2):
    """
    Update distance matrix after swapping two atoms.
    
    Efficiently updates distance matrix by swapping rows and columns
    instead of recomputing entire matrix.
    
    Args:
        dismat: NxN distance matrix
        Ind1: First atom index
        Ind2: Second atom index
    
    Returns:
        dismat: Updated distance matrix
    """
    dismat[[Ind1, Ind2]] = dismat[[Ind2, Ind1]]
    dismat[:, [Ind1, Ind2]] = dismat[:, [Ind2, Ind1]]
    return dismat


def create_atom_type_mapping(poscar):
    """
    Create atom type list from POSCAR structure.
    
    Converts POSCAR's element-count format to a flat list of atom types
    where each atom is labeled by its element index.
    
    Args:
        poscar: Structure dictionary from posreader
    
    Returns:
        atom_types: List of length AtomSum, each element is the atom type index
        
    Example:
        poscar['EleName'] = ['Sr', 'Ti', 'La', 'O']
        poscar['AtomNum'] = [60, 64, 4, 192]
        → atom_types = [0]*60 + [1]*64 + [2]*4 + [3]*192
    """
    atom_types = []
    for atom_type, count in enumerate(poscar['AtomNum']):
        atom_types.extend([atom_type] * count)
    return atom_types