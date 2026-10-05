# DASH trees in the RDKit

DASH (Dynamic Attention-based Substructure Hierarchy) assigns partial charges
and other atomic properties by walking a precomputed decision tree: each level
adds one neighbouring atom to the matched substructure, and the node where the
descent stops carries the value. It is the method of the python
[DASH-tree](https://github.com/rinikerlab/DASH-tree) package
([JCIM 2023](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c00800),
[J. Chem. Phys. 2024](https://doi.org/10.1063/5.0218154)), reimplemented here
in C++ with the same results to the last bit, a few hundred times faster, and
reading its tree from a memory-mapped file instead of 3 GB of heap.

The tree is data, not code. It lives in a `.dash` container, RDKit's own format,
whose layout is defined by the structs in `DASHTreeImpl.h` and mirrored in
`tools/dash_container.py`. The python package's trees come in a different, older
layout, called the *legacy* format here, which `tools/dash_convert.py` turns
into a container.

## Quick start

```python
from rdkit import Chem
from rdkit.Chem.DASHTree import GetDASHTree

mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))  # hydrogens carry charge: make them explicit
tree = GetDASHTree()                                       # the tree shipped with the RDKit
charges = tree.GetPartialCharges(mol)                      # one float per atom, summing to the formal charge
```

With no argument `GetDASHTree()` uses `$RDKIT_DASH_TREE` if set and otherwise
the tree in `Data/DASHTree`, and says so once. That tree is the published MBIS
charge tree pruned to 38 MB; its charges stay within 0.044 e of the full tree's
(0.0065 e RMS), see [Data/DASHTree/README.md](../../../Data/DASHTree/README.md).
For the full tree, or for properties other than MBIS charges, convert one and
pass it in, as a path or an http(s) URL that is fetched once into a cache:

```python
tree = GetDASHTree("default.dash", ["result", "std"])      # resolve only the columns you need
```

## Getting a legacy tree

**Convert it.** A legacy tree is a folder of `0.gz, 0.h5, 1.gz, 1.h5, ...`, one
pair per atom-feature branch, as the DASH-tree package stores them. In a
checkout of its repository the MBIS charge tree is
`serenityff/charge/data/default_dash_tree`; the tree with every property
(AM1BCC, RESP, Mulliken, DFT-D4 C6 and polarizability, dual descriptors,
dipoles) is downloaded by the package into
`serenityff/charge/data/additional_data/dashProps`. The converter needs
`numpy`, `pandas` and `pytables`, not the DASH-tree package:

```
python Code/GraphMol/DASHTree/tools/dash_convert.py <tree_folder> default.dash
python Code/GraphMol/DASHTree/tools/dash_convert.py <tree_folder> charges.dash --props result,std --no-source-ids
```

The first keeps every column plus the map back to the legacy tree's node
numbering (226 MB for the MBIS tree, ~100 s); the second is the 151 MB
charges-only container to hand around. Further options: `--float32` halves the
float64 property columns at the cost of bit-exactness against python,
`--features FILE` gives the atom-feature list of a tree not built on the
published 122 classes (a JSON list of `[Z, degree, charge, conjugated, numHs]`),
and `--hydrogen-branch N` overrides the hydrogen class, otherwise the one with
Z = 1. A list that does not fit the data is refused.

**Prune one.** `tools/dash_prune.py` writes a smaller container. Dropping a node
drops its subtree; an atom whose descent reaches the cut reads the deepest node
still there, so a pruned tree answers for every molecule the full one accepts,
with a more general substructure where nodes are missing.

```
python Code/GraphMol/DASHTree/tools/dash_prune.py default.dash small.dash --tolerance 0.02 --props result,std --drop-source-ids
```

| rule | keeps |
|---|---|
| `--tolerance EPS` | a subtree only if some value in it differs from the parent's by at least EPS (column `--value-property`, default `result`): exactly the nodes that change the answer |
| `--max-level L` | levels 0..L |
| `--min-size S` | nodes with at least S training atoms behind them (`--size-property`, default `size`) |
| `--molecules FILE` | every node on the descent paths of these molecules (`.sdf` or `.smi`); on its own, nothing else, so the result stays exact for them |

Rules combine with "and". Roots and the children of the hydrogen root always
stay, and a removed child that a kept sibling follows stays as a value-less
placeholder so the descent stops where the subtree was cut instead of wandering
to the sibling. Measured on the MBIS tree over 118 molecules:

| `--tolerance` | container | RMSD | max error |
|---|---|---|---|
| 0.01 e | 63.7 MB | 0.0034 e | 0.033 e |
| 0.02 e (shipped) | 38.2 MB | 0.0065 e | 0.044 e |
| 0.05 e | 14.3 MB | 0.014 e | 0.059 e |
| 0.1 e | 4.2 MB | 0.024 e | 0.13 e |

**Train one.** The DASH-tree package builds trees from a trained attention
model and writes them in the legacy format; a tree built on its own
atom-feature list converts with `--features`.

## Using a tree from python

```python
from rdkit import Chem
from rdkit.Chem import rdDASHTree
from rdkit.Chem.DASHTree import GetDASHTree, GetAtomMatchSmarts

tree = GetDASHTree("charges.dash", ["result", "std"])
mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))

# charges, and what they were made from
charges = tree.GetPartialCharges(mol)
details = tree.GetPartialChargesDetails(mol)        # 'charges', 'raw', 'std', 'match_depth'

# how the charges are made to sum to the formal charge
options = rdDASHTree.ChargeOptions()
options.normalization = rdDASHTree.Normalization.SYMMETRIC   # or NONE, or STD_WEIGHTED (default)
charges = tree.GetPartialCharges(mol, options)

# many molecules at once; 0 means every core, Ctrl-C cancels
batch = tree.GetPartialChargesBatch(mols, numThreads=0)

# the substructure an atom's descent matched, as SMARTS with the atom mapped as :1
smarts = GetAtomMatchSmarts(tree, mol, 0)           # [#6X4H3+0:1]-[#6X3H0+0](~[#8X1H0+0])~[#7X3H1+0]~...
path = tree.GetAtomNodePath(mol, 0)                 # branch index, then node ids
node = tree.GetNode(path[0], path[-1])              # GetFeature(), GetValue('result'), iterable children
```

A tree with more columns works the same way; name the columns you want when you
open it and only those are read:

```python
props = GetDASHTree("props.dash", ["AM1BCC", "AM1BCC_std", "DFTD4:C6"])
c6 = props.GetMolProperty(mol, "DFTD4:C6")          # one value per atom, no normalisation

# any per-atom property with a molecule-wide total normalises like the charges do
options = rdDASHTree.NormalizationOptions()
options.valueProperty, options.stdProperty = "AM1BCC", "AM1BCC_std"
am1bcc = props.GetNormalizedMolProperty(mol, Chem.GetFormalCharge(mol), options)
```

Two things to know. Hydrogens have charges of their own, so molecules need
explicit hydrogens (`Chem.AddHs`). The tree knows a fixed set of atom classes,
122 in the published trees; a molecule with an atom outside them (silicon, a
metal, an unusual valence) is refused with the atom named.

## Using a tree from C++

```cpp
#include <GraphMol/DASHTree/DASHTree.h>

RDKit::DASH::DASHTree tree("charges.dash", {"result", "std"});
std::vector<double> charges;
tree.getPartialCharges(*mol, charges);
tree.getNormalizedMolProperty(*mol, target, values, options);
tree.getPartialChargesBatch(mols, batch, RDKit::DASH::ChargeOptions(), 0);
```

The library target is `DASHTree`, gated on `RDK_USE_BOOST_IOSTREAMS` for the
memory mapping. A tree is immutable once opened and safe to query from any
number of threads.

## Layout and tests

```
AtomFeatures.{h,cpp}    the packed match key; classification against a tree's feature table
DASHTree.{h,cpp}        the public API; mapping, validating and descending a container
DASHTreeImpl.h          container structs and the matcher (not installed)
Charges.cpp             normalisation and the threaded batch entry points
Wrap/, nbWrap/          boost::python and nanobind bindings
test_data/              a 1 kB synthetic tree; 100 molecules with the python package's values
tools/                  dash_convert.py (legacy tree -> container), dash_prune.py, the format module they share
```

`catch_tests.cpp` and `rdkit/Chem/UnitTestDASHTree.py` run against the
synthetic tree and the shipped one. Point `DASH_TREE_FILE` at a converted full
tree to also run the tests that compare against the python package exactly.
