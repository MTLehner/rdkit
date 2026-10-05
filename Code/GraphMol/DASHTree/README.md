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

The converter needs `numpy`, `pandas` and `pytables`. It does not need the
DASH-tree package: the one fact it has to know about the tree, which branch is
the hydrogen class, it reads off the data and only cross-checks against the
package when that is importable.

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
| `--hydrogen-branch N` | override the branch read off the data |

The charges-only container to distribute:

```
python Code/GraphMol/DASHTree/tools/dash_convert.py <tree_folder> charges.dash --props result,std --no-source-ids
```

is 151 MB for the default tree. A successful run reports the branch count, the
hydrogen branch, the node count, every column with its dtype and the blocks it
wrote; `nodes past the default attention threshold` is 0 on the published trees.

## Using a container

```python
from rdkit import Chem
from rdkit.Chem import rdDASHTree
from rdkit.Chem.DASHTree import GetDASHTree, GetAtomMatchSmarts

tree = GetDASHTree("charges.dash", ["result", "std"])   # a path or an http(s) URL
mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))  # hydrogens must be explicit
charges = tree.GetPartialCharges(mol)                    # sum to the formal charge
details = tree.GetPartialChargesDetails(mol)             # raw values, stds, match depths
smarts = GetAtomMatchSmarts(tree, mol, 0)                # the substructure it matched

props = rdDASHTree.DASHTree("props.dash", ["AM1BCC", "AM1BCC_std", "DFTD4:C6"])
c6 = props.GetMolProperty(mol, "DFTD4:C6")
batch = tree.GetPartialChargesBatch(mols, numThreads=0)  # 0 = every core
```

## Tests

`catch_tests.cpp` and `rdkit/Chem/UnitTestDASHTree.py` run against the small
synthetic tree in `test_data/`. Point `DASH_TREE_FILE` at a converted published
tree to also run the tests that compare against the python package.
