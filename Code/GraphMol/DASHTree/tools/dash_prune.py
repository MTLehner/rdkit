#
#  Copyright (C) 2026 Marc Lehner and other RDKit contributors
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
"""Prune a .dash container into a smaller one.

A published DASH tree is 9.4 million nodes and 150 MB even with nothing but
the charges in it -- far too much for a repository, and more than a user who
wants a quick number needs. This writes a container holding a subset of the
nodes, renumbered breadth-first again so the C++ reads it like any other.

Dropping a node drops its whole subtree, and an atom whose descent reaches the
cut reads the value of the deepest node that is still there, exactly as it
would at the end of a full path. So a pruned tree never fails on a molecule the
full tree accepts; it answers with a shallower, more general substructure.

For that to hold the descent has to stop where the cut is. It takes the first
child in tree order that matches any candidate, so if that child were simply
gone a later sibling might match instead and the path would wander off. A
removed child that a kept sibling follows therefore stays as a value-less leaf:
the descent still takes it, stops there, and the property walk reads the parent.
Those placeholders are about half as many again as the kept nodes; their float
columns are NaN and their integer columns 0.

Four rules, combined with "and", choose what stays:

  --tolerance EPS   drop a subtree when every value in it lies within EPS of the
                    parent's value (column --value-property, default 'result').
                    This removes precisely the nodes that do not change the
                    answer. On the default MBIS tree, 0.05 e keeps 6.5 % of the
                    nodes and moves charges by 0.014 e RMS; see ../README.md.
  --max-level L     keep levels 0..L only
  --min-size S      drop nodes whose --size-property (default 'size', the
                    training support) is below S
  --molecules FILE  additionally keep every node on the descent path of every
                    atom of these molecules (.sdf or .smi; hydrogens are added
                    when missing), so the result stays exact for them. On its
                    own it keeps nothing else. Needs rdkit.Chem.rdDASHTree.

Roots are always kept, and so are the children of the hydrogen root: a hydrogen
is matched through its heavy neighbour, and the C++ refuses a hydrogen whose
heavy-neighbour node is missing rather than reading the root. The hydrogen
branch is the class with Z = 1 in the file's feature table (--hydrogen-branch
overrides it). Stop flags are
copied unchanged, since every ancestor of a kept node is kept and the
cumulative attention along its path is therefore the same.

Usage:
    python dash_prune.py default.dash small.dash --tolerance 0.05
    python dash_prune.py default.dash tests.dash --molecules mols.sdf --max-level 2
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dash_container import (Container, key_ranges_of, unpack_key, write_container)  # noqa: E402


def tolerance_mask(container, parent, level, values, eps):
  """True where a node changes the value read by at least eps.

  err(n) = max |value(x) - value(parent(n))| over n and its subtree. A node
  whose parent carries no value is always kept, since dropping it would lose
  the only value on that path.
  """
  vals = values.astype(np.float64)
  smin = np.where(np.isnan(vals), np.inf, vals)
  smax = np.where(np.isnan(vals), -np.inf, vals)
  for depth in range(int(level.max()), 0, -1):
    nodes = np.nonzero(level == depth)[0]
    np.minimum.at(smin, parent[nodes], smin[nodes])
    np.maximum.at(smax, parent[nodes], smax[nodes])
  parent_val = np.where(parent >= 0, vals[np.maximum(parent, 0)], np.nan)
  err = np.maximum(np.abs(smax - parent_val), np.abs(smin - parent_val))
  err[np.isnan(parent_val)] = np.inf
  return err >= eps


def molecule_paths(container, path):
  """Global node ids on the descent path of every atom of every molecule."""
  from rdkit import Chem
  from rdkit.Chem import rdDASHTree
  tree = rdDASHTree.DASHTree(container.path, [])
  if path.lower().endswith(".sdf"):
    mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
  else:
    mols = [m for m in Chem.SmilesMolSupplier(path, titleLine=False) if m is not None]
  roots = container.branch_root.astype(np.int64)
  keep = np.zeros(container.n_nodes, dtype=bool)
  n_atoms = 0
  for mol in mols:
    if any(a.GetTotalNumHs() for a in mol.GetAtoms()):
      mol = Chem.AddHs(mol)
    for atom in range(mol.GetNumAtoms()):
      p = tree.GetAtomNodePath(mol, atom)
      if len(p) > 1:
        keep[roots[p[0]] + np.asarray(p[1:], dtype=np.int64)] = True
      n_atoms += 1
  return keep, len(mols), n_atoms


def renumber(container, keep, parent, level):
  """Breadth-first numbering of the kept nodes, branch by branch.

  Returns old ids in new order, and per new node its child count and first
  child. Within a level the nodes are ordered by their parent's new index, so
  each node's children form one contiguous run, in the order they had before.
  """
  roots = container.branch_root.astype(np.int64)
  ends = np.append(roots[1:], container.n_nodes)
  new_of_old = np.full(container.n_nodes, -1, dtype=np.int64)
  order, n_children, first_child, new_roots = [], [], [], []
  base = 0
  for b in range(container.n_branches):
    ids = np.arange(roots[b], ends[b])
    kept = ids[keep[roots[b]:ends[b]]]
    if kept.size == 0 or kept[0] != roots[b]:
      raise ValueError(f"branch {b}: its root was pruned")
    seq = [np.array([roots[b]])]
    new_of_old[roots[b]] = base
    count = 1
    lv = level[kept]
    for depth in range(1, int(lv.max()) + 1):
      nodes = kept[lv == depth]
      # by the parent's new index first, then by old id: keeps each node's
      # children together and in their original order
      nodes = nodes[np.lexsort((nodes, new_of_old[parent[nodes]]))]
      new_of_old[nodes] = base + count + np.arange(nodes.size)
      seq.append(nodes)
      count += nodes.size
    branch_order = np.concatenate(seq)
    children = branch_order[1:]
    parent_new = new_of_old[parent[children]] - base
    cnt = np.bincount(parent_new, minlength=count)
    first = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, parent_new, new_of_old[children])
    first[cnt == 0] = 0
    order.append(branch_order)
    n_children.append(cnt)
    first_child.append(first)
    new_roots.append(base)
    base += count
  return (np.concatenate(order), np.concatenate(n_children), np.concatenate(first_child),
          np.asarray(new_roots, dtype=np.int64))


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("inp")
  ap.add_argument("out")
  ap.add_argument("--tolerance", type=float, default=None,
                  help="drop subtrees whose values all lie within this of the parent's")
  ap.add_argument("--value-property", default="result",
                  help="the column --tolerance looks at (default: result)")
  ap.add_argument("--max-level", type=int, default=None, help="keep levels 0..L")
  ap.add_argument("--min-size", type=int, default=None,
                  help="drop nodes whose --size-property is below this")
  ap.add_argument("--size-property", default="size")
  ap.add_argument("--molecules", default=None,
                  help=".sdf or .smi whose atoms' descent paths are kept in full")
  ap.add_argument("--hydrogen-branch", type=int, default=None,
                  help="branch matched through the heavy neighbour (default: the class with Z = 1)")
  ap.add_argument("--props", default=None,
                  help="comma separated property columns to keep (default: all)")
  ap.add_argument("--drop-source-ids", action="store_true",
                  help="omit the map back to the source tree's node numbering")
  args = ap.parse_args(argv)
  if (args.tolerance is None and args.max_level is None and args.min_size is None and
      args.molecules is None):
    ap.error("give at least one of --tolerance, --max-level, --min-size, --molecules")

  src = Container(args.inp)
  print(f"{args.inp}: {src.n_branches} branches, {src.n_nodes:,} nodes, "
        f"properties {list(src.props)}")
  parent, level = src.parent_and_level()
  print(f"  levels 0..{int(level.max())}")

  if args.hydrogen_branch is not None:
    h_branch = args.hydrogen_branch
    if not 0 <= h_branch < src.n_branches:
      sys.exit(f"--hydrogen-branch {h_branch} is not a branch of this tree")
  else:
    try:
      h_branch = src.hydrogen_branch()
    except ValueError as e:
      sys.exit(f"{e}; pass --hydrogen-branch")
  print(f"  hydrogen branch: {h_branch}")

  # ---- the generic rules: a node stays if it passes all of them ------------
  # with none given, --molecules alone keeps just the paths it names
  generic = (args.tolerance is not None or args.max_level is not None or
             args.min_size is not None)
  passes = np.full(src.n_nodes, generic, dtype=bool)
  if args.tolerance is not None:
    if args.value_property not in src.props:
      sys.exit(f"no property '{args.value_property}' in the file; available: {list(src.props)}")
    changed = tolerance_mask(src, parent, level, src.props[args.value_property][1],
                             args.tolerance)
    print(f"  tolerance {args.tolerance} on '{args.value_property}': "
          f"{int((~changed).sum()):,} nodes change the value by less")
    passes &= changed
  if args.max_level is not None:
    passes &= level <= args.max_level
    print(f"  max level {args.max_level}: {int((level > args.max_level).sum()):,} nodes deeper")
  if args.min_size is not None:
    if args.size_property not in src.props:
      sys.exit(f"no property '{args.size_property}' in the file; available: {list(src.props)}")
    small = src.props[args.size_property][1] < args.min_size
    passes &= ~small
    print(f"  min size {args.min_size}: {int(small.sum()):,} nodes below")

  roots = src.branch_root.astype(np.int64)
  passes[roots] = True
  h_root = roots[h_branch]
  h_children = np.nonzero(parent == h_root)[0]
  passes[h_children] = True

  # closure: a node stays only if every ancestor stays; walk down the levels
  keep = np.zeros(src.n_nodes, dtype=bool)
  keep[roots] = True
  for depth in range(1, int(level.max()) + 1):
    nodes = np.nonzero(level == depth)[0]
    keep[nodes] = keep[parent[nodes]] & passes[nodes]
  n_rules = int(keep.sum())

  if args.molecules is not None:
    on_path, n_mols, n_atoms = molecule_paths(src, args.molecules)
    keep |= on_path  # ancestors of a path node are on the path, so this stays closed
    print(f"  molecules: {n_mols} molecules, {n_atoms:,} atoms, "
          f"{int(on_path.sum()):,} nodes on their paths, {int(keep.sum()) - n_rules:,} added")
  n_real = int(keep.sum())

  # ---- placeholders: a removed child that a kept sibling follows stays as a
  # value-less leaf, so the descent still takes it and stops there ----------
  # children are contiguous in the old numbering, so a child's position among
  # its siblings is its distance from the parent's first child
  first = src.records["firstChild"].astype(np.int64)
  kids = np.nonzero(parent >= 0)[0]
  kids = kids[keep[parent[kids]]]
  pos = kids - first[parent[kids]]
  kept_kid = keep[kids]
  last_kept = np.full(src.n_nodes, -1, dtype=np.int64)
  np.maximum.at(last_kept, parent[kids[kept_kid]], pos[kept_kid])
  placeholder = np.zeros(src.n_nodes, dtype=bool)
  placeholder[kids[~kept_kid & (pos < last_kept[parent[kids]])]] = True
  keep |= placeholder
  print(f"  keeping {n_real:,} of {src.n_nodes:,} nodes "
        f"({100.0 * n_real / src.n_nodes:.2f} %) plus {int(placeholder.sum()):,} "
        f"placeholders")

  # ---- renumber and gather ----------------------------------------------
  old, n_children, first_child, new_roots = renumber(src, keep, parent, level)
  records = np.zeros(old.size, dtype=src.records.dtype)
  records["key"] = src.records["key"][old]
  records["flags"] = src.records["flags"][old]
  records["nChildren"] = n_children
  records["firstChild"] = first_child
  attn = src.attn[old]
  source = None if (args.drop_source_ids or src.source is None) else src.source[old]
  names = list(src.props)
  if args.props:
    names = [p.strip() for p in args.props.split(",") if p.strip()]
    missing = [p for p in names if p not in src.props]
    if missing:
      sys.exit(f"requested columns not in the file: {missing}; available: {list(src.props)}")
  blank = placeholder[old]
  props = []
  for name in names:
    tag, arr = src.props[name]
    arr = arr[old].copy()
    arr[blank] = np.nan if np.issubdtype(arr.dtype, np.floating) else 0
    props.append((name, tag, arr))

  blocks = write_container(args.out, new_roots, src.features, records, attn, source, props,
                           src.threshold, key_ranges_of(records, new_roots))
  new_level = level[old]
  print(f"\nwrote {args.out}  {os.path.getsize(args.out) / 1e6:.1f} MB")
  for name, arr, off in blocks:
    print(f"    {name:<28} n={arr.size:>10,}  {arr.nbytes / 1e6:>7.1f} MB @ {off}")
  print("  nodes per level, before -> after:")
  for depth in range(int(level.max()) + 1):
    print(f"    {depth:>2}  {int((level == depth).sum()):>10,} -> {int((new_level == depth).sum()):>9,}")


if __name__ == "__main__":
  main()
