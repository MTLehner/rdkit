# DASHTree

Partial charges and other atomic properties from a DASH tree (Dynamic
Attention-based Substructure Hierarchy), the method of the python
[DASH-tree](https://github.com/rinikerlab/DASH-tree) package, reimplemented in
C++.

The tree is a data file, not part of the library. The library reads RDKit's own
`.dash` container, and `tools/dash_convert.py` produces one from a DASH-tree
distribution.

## Converting a tree

A DASH-tree distribution is a folder holding `0.gz, 0.h5, 1.gz, 1.h5, ...`, one
pair per atom-feature branch. In a checkout of the DASH-tree repository the
default MBIS charge tree is `serenityff/charge/data/default_dash_tree`, and the
tree with all properties lands in `serenityff/charge/data/additional_data/dashProps`
when the package downloads it.

The converter needs `numpy`, `pandas` and `pytables`, not the DASH-tree package.
The container carries the tree's atom-feature table, one class per branch; the
published trees' 122 classes are embedded in the converter as the default, and a
tree built on another feature list is given with `--features` (a JSON list of
`[Z, degree, charge, conjugated, numHs]`). The data has to agree with the table
on which branch is the hydrogen class, so a wrong list is caught.

```
python Code/GraphMol/DASHTree/tools/dash_convert.py <tree_folder> default.dash
```

keeps every property column, the attentions and the map back to the python
package's node numbering (226 MB for the default tree, ~100 s). Options:

| option | effect |
|---|---|
| `--props result,std` | keep only these columns; the default tree's `size` column is read by nobody |
| `--no-source-ids` | omit the map back to the python numbering; only the exactness tests read it |
| `--float32` | narrow float64 columns (AM1BCC, RESP, DFTD4, ...) to float32: half the size, no longer bit-exact against python |
| `--features FILE` | the atom-feature list the tree was built on, if not the published one |
| `--hydrogen-branch N` | override the hydrogen class, which is otherwise the one with Z = 1 |

The charges-only container to distribute:

```
python Code/GraphMol/DASHTree/tools/dash_convert.py <tree_folder> charges.dash --props result,std --no-source-ids
```

is 151 MB for the default tree. A successful run reports the branch count, the
hydrogen branch, the node count, every column with its dtype and the blocks it
wrote; `nodes past the default attention threshold` is 0 on the published trees.

## Pruning a container

`tools/dash_prune.py` writes a smaller container holding a subset of the nodes.
Dropping a node drops its subtree, and an atom whose descent reaches the cut
reads the deepest node still there, so a pruned tree answers for every molecule
the full tree accepts, with a shallower, more general substructure where nodes
are missing. Roots and the children of the hydrogen root always stay.

```
python Code/GraphMol/DASHTree/tools/dash_prune.py default.dash small.dash --tolerance 0.05
```

| rule | keeps |
|---|---|
| `--tolerance EPS` | a subtree only if some value in it differs from the parent's by at least EPS (column `--value-property`, default `result`): the nodes that change the answer |
| `--max-level L` | levels 0..L |
| `--min-size S` | nodes whose `--size-property` (default `size`, the training support) is at least S |
| `--molecules FILE` | additionally every node on the descent paths of these molecules (`.sdf` or `.smi`), so the result stays exact for them |

The rules combine with "and", and `--molecules` on its own keeps just those
paths; `--props` and `--drop-source-ids` trim columns as in the converter. A
removed child that a kept sibling follows stays as a value-less leaf, so the
descent still stops where the subtree was cut instead of wandering to the
sibling; those placeholders are about half as many again as the kept nodes.

Measured on the default MBIS tree against the full tree, over 118 test
molecules (5,218 atoms), std-weighted charges:

| `--tolerance` | nodes kept | container | RMSD | max error |
|---|---|---|---|---|
| 0.01 e | 36 % + placeholders | 63.7 MB | 0.0034 e | 0.033 e |
| 0.02 e | 20 % + placeholders | 38.2 MB | 0.0065 e | 0.044 e |
| 0.05 e | 6.5 % + placeholders | 14.3 MB | 0.014 e | 0.06 e |
| 0.1 e | 1.7 % + placeholders | 4.2 MB | 0.024 e | 0.13 e |
| 0.2 e | 0.2 % + placeholders | 0.7 MB | 0.037 e | 0.20 e |

The tree shipped in `Data/DASHTree` is the 0.02 e one; its README has the
full comparison and the reason.

## Using a container

```python
from rdkit import Chem
from rdkit.Chem import rdDASHTree
from rdkit.Chem.DASHTree import GetDASHTree, GetAtomMatchSmarts

tree = GetDASHTree("charges.dash", ["result", "std"])   # a path or an http(s) URL
tree = GetDASHTree()   # $RDKIT_DASH_TREE, else the pruned tree in Data/DASHTree
mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))  # hydrogens must be explicit
charges = tree.GetPartialCharges(mol)                    # sum to the formal charge
details = tree.GetPartialChargesDetails(mol)             # raw values, stds, match depths
smarts = GetAtomMatchSmarts(tree, mol, 0)                # the substructure it matched

props = rdDASHTree.DASHTree("props.dash", ["AM1BCC", "AM1BCC_std", "DFTD4:C6"])
c6 = props.GetMolProperty(mol, "DFTD4:C6")

# any per-atom property with a molecule-wide total normalises like the charges do
options = rdDASHTree.NormalizationOptions()
options.valueProperty, options.stdProperty = "AM1BCC", "AM1BCC_std"
am1bcc = props.GetNormalizedMolProperty(mol, Chem.GetFormalCharge(mol), options)
batch = tree.GetPartialChargesBatch(mols, numThreads=0)  # 0 = every core
```

## Tests

`catch_tests.cpp` and `rdkit/Chem/UnitTestDASHTree.py` run against the small
synthetic tree in `test_data/`. Point `DASH_TREE_FILE` at a converted published
tree to also run the tests that compare against the python package.
