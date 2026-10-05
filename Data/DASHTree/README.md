# Data/DASHTree

`default_pruned.dash` is the DASH tree `rdkit.Chem.DASHTree.GetDASHTree()`
falls back to when no container and no `RDKIT_DASH_TREE` are given: the
published default MBIS charge tree of the
[DASH-tree](https://github.com/rinikerlab/DASH-tree) package, pruned to the
nodes that change a charge by at least 0.02 e. It carries the `result` and
`std` columns only.

Made with the tools in `Code/GraphMol/DASHTree/tools/`, from the legacy tree in
the `serenityff/charge/data/default_dash_tree` folder of a DASH-tree checkout:

```
python dash_convert.py <DASH-tree>/serenityff/charge/data/default_dash_tree default.dash
python dash_prune.py default.dash default_pruned.dash --tolerance 0.02 --props result,std --drop-source-ids
```

Both steps are deterministic, so the file is reproducible byte for byte.

## Why 0.02 e

The full tree is 9,439,485 nodes and 151 MB with these two columns, far too
big for the repository. Pruning drops a subtree when every value in it lies
within the tolerance of the parent's, and keeps a value-less placeholder where
a removed child has a kept sibling after it, so the descent stops where the
subtree was cut instead of wandering to the sibling. Measured against the full
tree over 118 test molecules (5,218 atoms; the 100 in
`Code/GraphMol/DASHTree/test_data/dash_ref_mols.sdf` and 18 small ones),
std-weighted charges:

| tolerance | nodes kept | placeholders | on disk | in git | RMSD | max error | mean match depth |
|---|---|---|---|---|---|---|---|
| full tree | 9,439,485 | — | 151 MB | — | — | — | 11.1 |
| 0.01 e | 3,368,081 | 613,484 | 63.7 MB | 38 MB | 0.0034 e | 0.033 e | 9.1 |
| **0.02 e** | **1,879,766** | **505,427** | **38.2 MB** | **22 MB** | **0.0065 e** | **0.044 e** | **8.1** |
| 0.05 e | 612,447 | 278,628 | 14.3 MB | 8.2 MB | 0.014 e | 0.059 e | 6.3 |
| 0.1 e | 155,734 | 103,541 | 4.2 MB | 2.3 MB | 0.024 e | 0.13 e | 5.1 |
| 0.2 e | 22,598 | 21,887 | 0.7 MB | 0.4 MB | 0.037 e | 0.20 e | 3.6 |

0.02 e is the most accurate tree that stays under GitHub's 50 MB per-file
warning. Every molecule the full tree assigns is assigned, and the charges
still sum to the formal charge exactly. For the full tree, convert one and
pass it to `GetDASHTree()` or set `RDKIT_DASH_TREE`.
