#
#  Copyright (C) 2026 Marc Lehner and other RDKit contributors
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
"""Convert a DASH tree into RDKit's .dash container.

The DASH-tree python package stores a tree as a pair of files per atom-feature
branch:

  <b>.gz   gzipped pickle of a list of node tuples
           (id, atom_type, con_atom, con_type, attention, [child ids])
  <b>.h5   pandas HDF5 table (Fixed format), one row per node

Most of that is redundant, which this script verifies rather than assumes: the
h5 columns ``atom_type``, ``con_atom`` and ``con_type`` are identical to node
fields 1/2/3, ``level`` is the node's depth, ``max_attention`` is node field 4
rounded to float16, and the node tuple's leading id is always its own index.
Only the child lists and the value columns carry information.

Two decisions shape the container it writes:

  * Nodes are renumbered in breadth-first order across the whole forest. That
    makes each node's children a contiguous run of ids, so no child-index array
    is needed at all, and it puts the shallow levels every descent passes
    through into one small region at the front of the file -- which is what
    keeps the working set to a couple of pages per atom instead of ten.
  * A node's match key, child range and stop flag live in one 8-byte record, so
    a descent step is a single dependent load.

The container also carries the tree's atom-feature table, one class per branch,
which is what the C++ classifies atoms with. The published trees were built on
the 122 classes of the DASH-tree package's ``AtomFeatures.feature_list``, which
is embedded below as the default; a tree built on another list needs
``--features`` (a JSON list of ``[Z, degree, charge, conjugated, numHs]``).

The hydrogen branch matters: the first descent step out of its root walks to the
heavy neighbour without accumulating attention, and the precomputed stop flags
depend on that. It is the class with Z = 1, and the data has to agree: the
hydrogen root is the only root whose children attach with con_atom = -1, so a
tree built on a different feature list than the one given is caught here.

Requires numpy, pandas and pytables; nothing from the DASH-tree package. The
container layout lives in dash_container.py, shared with dash_prune.py.

Usage:
    python dash_convert.py <tree_folder> <out.dash> [--props result,std,...]
"""

import argparse
import gzip
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dash_container import (  # noqa: E402
    DEFAULT_ATTENTION_THRESHOLD, DTYPE_F16, DTYPE_F32, DTYPE_F64, DTYPE_I32,
    FEATURE_DTYPE, KEY_CON_ATOM_BITS, KEY_CON_TYPE_BITS, MAX_CHILDREN, MAX_CON_ATOM,
    MAX_CON_TYPE, MAX_NODES, NODE_FLAG_STOP, NP_DTYPE, RECORD_DTYPE, pack_key,
    write_container)

_NP_DTYPE = {tag: dt.type for tag, dt in NP_DTYPE.items()}

# (atomicNum, degree, formalCharge, conjugated, numHs) of the published trees'
# 122 classes, in branch order: serenityff.charge.tree.atom_features
# .AtomFeatures.feature_list. The order is part of those trees' data.
STANDARD_FEATURES = [
    (5, 1, 0, 0, 2), (5, 3, -1, 1, 0), (5, 3, 0, 0, 0), (5, 3, 0, 0, 2),
    (5, 4, -1, 0, 0), (5, 4, -1, 0, 2), (35, 1, 0, 0, 0), (6, 1, -1, 0, 0),
    (6, 1, -1, 1, 0), (6, 1, 0, 0, 1), (6, 1, 0, 0, 2), (6, 1, 0, 0, 3),
    (6, 1, 0, 1, 1), (6, 2, -1, 1, 1), (6, 2, 0, 0, 0), (6, 2, 0, 0, 1),
    (6, 2, 0, 0, 2), (6, 2, 0, 1, 0), (6, 2, 0, 1, 1), (6, 3, -1, 0, 0),
    (6, 3, -1, 1, 0), (6, 3, -1, 0, 1), (6, 3, -1, 1, 1), (6, 3, 0, 0, 0),
    (6, 3, 0, 0, 1), (6, 3, 0, 0, 2), (6, 3, 0, 1, 0), (6, 3, 0, 1, 1),
    (6, 3, 0, 1, 2), (6, 3, 1, 0, 0), (6, 3, 1, 1, 0), (6, 4, 0, 0, 0),
    (6, 4, 0, 0, 1), (6, 4, 0, 0, 2), (6, 4, 0, 0, 3), (17, 1, 0, 0, 0),
    (9, 1, 0, 0, 0), (1, 1, 0, 0, 0), (53, 1, 0, 0, 0), (53, 3, 0, 0, 0),
    (53, 3, 0, 0, 1), (53, 4, 0, 0, 0), (7, 1, -1, 0, 0), (7, 1, -1, 1, 0),
    (7, 1, 0, 0, 0), (7, 1, 0, 1, 0), (7, 1, 0, 1, 1), (7, 1, 0, 1, 2),
    (7, 1, 1, 0, 3), (7, 2, -1, 0, 0), (7, 2, -1, 0, 1), (7, 2, -1, 1, 0),
    (7, 2, -1, 1, 1), (7, 2, 0, 0, 0), (7, 2, 0, 0, 1), (7, 2, 0, 1, 0),
    (7, 2, 0, 1, 1), (7, 2, 1, 0, 0), (7, 2, 1, 0, 1), (7, 2, 1, 0, 2),
    (7, 2, 1, 1, 0), (7, 2, 1, 1, 1), (7, 3, 0, 0, 0), (7, 3, 0, 0, 1),
    (7, 3, 0, 0, 2), (7, 3, 0, 1, 0), (7, 3, 0, 1, 1), (7, 3, 0, 1, 2),
    (7, 3, 1, 0, 0), (7, 3, 1, 0, 1), (7, 3, 1, 1, 0), (7, 3, 1, 1, 1),
    (7, 4, 1, 0, 0), (7, 4, 1, 0, 1), (7, 4, 1, 0, 2), (7, 4, 1, 0, 3),
    (8, 1, -1, 0, 0), (8, 1, -1, 1, 0), (8, 1, 0, 0, 0), (8, 1, 0, 0, 1),
    (8, 1, 0, 1, 0), (8, 1, 0, 1, 1), (8, 2, 0, 0, 0), (8, 2, 0, 0, 1),
    (8, 2, 0, 0, 2), (8, 2, 0, 1, 0), (8, 2, 0, 1, 1), (8, 2, 0, 1, 2),
    (8, 2, 1, 0, 0), (8, 2, 1, 0, 1), (8, 2, 1, 1, 0), (8, 3, 1, 0, 0),
    (8, 3, 1, 0, 1), (15, 1, 0, 0, 1), (15, 2, 0, 0, 0), (15, 2, 0, 0, 1),
    (15, 2, 0, 1, 0), (15, 3, 0, 0, 0), (15, 4, 0, 0, 0), (15, 4, 0, 0, 1),
    (15, 4, 0, 1, 0), (15, 4, 0, 1, 1), (15, 4, 1, 0, 0), (15, 5, 0, 0, 0),
    (15, 5, 0, 0, 1), (16, 1, -1, 0, 0), (16, 1, -1, 1, 0), (16, 1, 0, 0, 0),
    (16, 1, 0, 1, 0), (16, 2, 0, 0, 0), (16, 2, 0, 1, 0), (16, 2, 0, 0, 1),
    (16, 2, 1, 1, 0), (16, 3, 0, 0, 0), (16, 3, 0, 1, 0), (16, 3, 1, 1, 0),
    (16, 3, 1, 0, 0), (16, 4, 0, 0, 0), (16, 4, 0, 0, 1), (16, 4, 0, 1, 0),
    (16, 4, 1, 0, 0), (16, 4, 1, 1, 0),
]
# h5 columns that merely duplicate the topology and are therefore never stored
REDUNDANT_COLUMNS = ("level", "atom_type", "con_atom", "con_type", "max_attention")

# atomic number of the feature class whose atoms are matched through their heavy
# neighbour; the descent spends its first level getting there and does not
# accumulate that level's attention, which the stop flag has to account for
HYDROGEN = 1


def load_branch(folder, b):
    with gzip.open(os.path.join(folder, f"{b}.gz"), "rb") as f:
        return pickle.load(f)


def looks_like_hydrogen_root(tree):
    """Does every child of this branch's root attach with con_atom = -1?

    A hydrogen's descent starts at its heavy neighbour, and the node for that
    step has no attachment position. Every other candidate the descent ever
    generates carries a real position, so no other root has such children.
    """
    children = [c for c in tree[0][5] if c != 0]
    return bool(children) and all(int(tree[c][2]) == -1 for c in children)


def load_features(path):
    """A feature list from a JSON file: [[Z, degree, charge, conjugated, numHs], ...]."""
    with open(path) as f:
        raw = json.load(f)
    features = []
    for i, row in enumerate(raw):
        if len(row) != 5:
            sys.exit(f"{path}: feature {i} is not a 5-tuple: {row}")
        z, degree, charge, conjugated, num_hs = row
        if not (0 <= z <= 255 and 0 <= degree <= 255 and -128 <= charge <= 127
                and conjugated in (0, 1, False, True) and 0 <= num_hs <= 255):
            sys.exit(f"{path}: feature {i} is out of range: {row}")
        features.append((int(z), int(degree), int(charge), int(bool(conjugated)), int(num_hs)))
    return features


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree_folder")
    ap.add_argument("out")
    ap.add_argument("--props", default=None,
                    help="comma separated property columns to keep "
                         "(default: every non-redundant column in the h5 files)")
    ap.add_argument("--float32", action="store_true",
                    help="narrow every float column to float32. Halves the "
                         "float64 columns at the cost of reproducing them exactly.")
    ap.add_argument("--no-source-ids", action="store_true",
                    help="omit the map back to the python package's node numbering. "
                         "It is never read unless a caller asks for it, and the "
                         "bit-exactness tests do.")
    ap.add_argument("--features", default=None,
                    help="JSON file with the atom-feature list the tree was built "
                         "on, one [Z, degree, charge, conjugated, numHs] per branch "
                         "(default: the 122 classes of the published trees)")
    ap.add_argument("--hydrogen-branch", type=int, default=None,
                    help="branch index of the hydrogen feature class "
                         "(default: the class with Z = 1)")
    args = ap.parse_args(argv)

    import pandas as pd

    folder = args.tree_folder
    if not os.path.isdir(folder):
        sys.exit(f"{folder} is not a directory")
    gz = sorted(int(f[:-3]) for f in os.listdir(folder)
                if f.endswith(".gz") and f[:-3].isdigit())
    n_branches = len(gz)
    if n_branches == 0:
        sys.exit(f"no <n>.gz files in {folder}: a DASH-tree distribution is 0.gz, "
                 f"0.h5, 1.gz, 1.h5, ... directly inside the folder you pass")
    if gz != list(range(n_branches)):
        sys.exit(f"the .gz files in {folder} are not numbered 0..{n_branches - 1} "
                 f"without gaps")
    missing_h5 = [b for b in gz if not os.path.exists(os.path.join(folder, f"{b}.h5"))]
    if missing_h5:
        sys.exit(f"no .h5 data file for branches {missing_h5[:10]}")
    print(f"branches: {n_branches}")

    # ---- pass 1: sizes, the hydrogen branch and the property column list ----
    branch_n = np.zeros(n_branches, dtype=np.int64)
    h_candidates = []
    for b in range(n_branches):
        tree = load_branch(folder, b)
        branch_n[b] = len(tree)
        if looks_like_hydrogen_root(tree):
            h_candidates.append(b)
        del tree
    n_nodes = int(branch_n.sum())

    features = load_features(args.features) if args.features else STANDARD_FEATURES
    if len(features) != n_branches:
        sys.exit(f"the feature list has {len(features)} classes but the tree has "
                 f"{n_branches} branches; pass --features with the list the tree "
                 f"was built on")
    if len(set(features)) != len(features):
        sys.exit("the feature list contains the same class twice")
    h_from_table = [i for i, f in enumerate(features) if f[0] == HYDROGEN]
    if args.hydrogen_branch is not None:
        h_branch = args.hydrogen_branch
        if not 0 <= h_branch < n_branches:
            sys.exit(f"--hydrogen-branch {h_branch} is not a branch of this tree")
        if h_branch not in h_from_table or (h_candidates and h_branch not in h_candidates):
            print(f"WARNING: --hydrogen-branch {h_branch}, but the feature table "
                  f"puts hydrogen at {h_from_table} and the data at "
                  f"{h_candidates}", file=sys.stderr)
    elif len(h_from_table) == 1:
        h_branch = h_from_table[0]
        if h_candidates != [h_branch]:
            sys.exit(f"the feature table puts hydrogen at branch {h_branch}, but "
                     f"the roots whose children attach with con_atom = -1 are "
                     f"{h_candidates}: the tree was built on a different feature "
                     f"list than the one given; pass --features")
    else:
        sys.exit(f"the feature list has {len(h_from_table)} classes with Z = 1 "
                 f"({h_from_table}); pass --hydrogen-branch (it decides where "
                 f"the attention accumulation starts, so it must be right)")
    print(f"hydrogen branch: {h_branch}")
    feature_table = np.zeros(n_branches, dtype=FEATURE_DTYPE)
    for i, (z, degree, charge, conjugated, num_hs) in enumerate(features):
        feature_table[i] = (z, degree, charge, conjugated, num_hs, (0, 0, 0))
    if n_nodes > MAX_NODES:
        sys.exit(f"{n_nodes} nodes exceeds the container's 32-bit node ids")
    branch_root = np.zeros(n_branches, dtype=np.uint32)
    acc = 0
    for b in range(n_branches):
        branch_root[b] = acc
        acc += int(branch_n[b])
    print(f"nodes: {n_nodes:,}")

    df0 = pd.read_hdf(os.path.join(folder, "0.h5"), key="df", mode="r")
    if args.props:
        props = [p.strip() for p in args.props.split(",") if p.strip()]
        missing = [p for p in props if p not in df0.columns]
        if missing:
            sys.exit(f"requested columns not in the data: {missing}\n"
                     f"available: {list(df0.columns)}")
        redundant = [p for p in props if p in REDUNDANT_COLUMNS]
        if redundant:
            sys.exit(f"these columns duplicate the topology and are not stored: "
                     f"{redundant}")
    else:
        props = [c for c in df0.columns if c not in REDUNDANT_COLUMNS]
    print(f"properties: {props}")
    print(f"  (skipped as redundant: "
          f"{[c for c in df0.columns if c in REDUNDANT_COLUMNS]})")

    def out_dtype(col):
        """Keep each column at the width its source actually needs.

        The MBIS charges are float16 and round-trip through it exactly; the
        DASH-properties columns (AM1BCC, DFTD4, ...) are genuine float64 and do
        not survive narrowing, so storing them as float32 would quietly lose
        bits. --float32 narrows them anyway, for when size matters more than
        reproducing the source exactly.
        """
        d = df0[col].dtype
        if np.issubdtype(d, np.integer):
            return DTYPE_I32
        if args.float32:
            return DTYPE_F32
        if d == np.float16:
            return DTYPE_F16
        if d == np.float32:
            return DTYPE_F32
        return DTYPE_F64

    prop_dtypes = [out_dtype(c) for c in props]
    for c, dt in zip(props, prop_dtypes):
        print(f"    {c:<28} {df0[c].dtype} -> {_NP_DTYPE[dt].__name__}")

    # ---- pass 2: renumber breadth-first, one branch at a time --------------
    # Within a branch the roots come first and then every node's children in
    # tree order, which is exactly breadth-first: processing nodes in id order
    # and appending their children makes each node's children a contiguous run,
    # so no child-index array is needed, and it gathers the shallow levels every
    # descent passes through into a small region at the start of the branch.
    node_key = np.zeros(n_nodes, dtype=np.uint16)
    node_children = np.zeros(n_nodes, dtype=np.uint8)
    node_first = np.zeros(n_nodes, dtype=np.uint32)
    node_flags = np.zeros(n_nodes, dtype=np.uint8)
    node_attn = np.zeros(n_nodes, dtype=np.float32)
    node_source = np.zeros(n_nodes, dtype=np.uint32)
    prop_arrays = [np.empty(n_nodes, dtype=_NP_DTYPE[dt]) for dt in prop_dtypes]

    max_atom_type = max_con_atom = max_con_type = -(10 ** 9)
    min_con_atom = min_con_type = 10 ** 9
    max_children = 0
    n_stop = 0

    for b in range(n_branches):
        tree = load_branch(folder, b)
        base = int(branch_root[b])
        count = len(tree)
        cum = np.zeros(count, dtype=np.float64)

        root = tree[0]
        node_key[base] = pack_key(int(root[1]), int(root[2]), int(root[3]))
        node_source[base] = 0
        node_attn[base] = np.float32(root[4])
        cum[0] = 0.0  # a root's own attention is never accumulated

        next_local = 1
        for local in range(count):
            src = int(node_source[base + local])
            # The root of every branch lists itself as its own first child. That
            # edge can never be matched: its key carries conAtom = -1, while
            # every candidate the descent generates carries a real position, and
            # the one candidate that does use -1 -- a hydrogen's heavy neighbour
            # -- cannot carry the hydrogen feature class. So it is dropped
            # instead of encoded.
            children = [c for c in tree[src][5] if c != src]
            if src == 0 and len(children) != len(tree[0][5]) - 1:
                sys.exit(f"branch {b}: root does not list itself exactly once")
            if src != 0 and len(children) != len(tree[src][5]):
                sys.exit(f"branch {b}: node {src} references itself")
            if len(children) > MAX_CHILDREN:
                sys.exit(f"node {src} of branch {b} has {len(children)} children, "
                         f"more than the container's {MAX_CHILDREN}")
            max_children = max(max_children, len(children))
            node_children[base + local] = len(children)
            node_first[base + local] = base + next_local

            # the first step out of a hydrogen root only walks to the heavy
            # neighbour, and the descent does not accumulate its attention
            accumulate = not (src == 0 and b == h_branch)
            for c in children:
                cn = tree[c]
                at, ca, ct = int(cn[1]), int(cn[2]), int(cn[3])
                max_atom_type = max(max_atom_type, at)
                max_con_atom = max(max_con_atom, ca)
                min_con_atom = min(min_con_atom, ca)
                max_con_type = max(max_con_type, ct)
                min_con_type = min(min_con_type, ct)
                if not 0 <= at < n_branches:
                    sys.exit(f"atom_type {at} is not one of the {n_branches} classes")
                if not -1 <= ca <= MAX_CON_ATOM:
                    sys.exit(f"con_atom {ca} does not fit {KEY_CON_ATOM_BITS} bits")
                if not -1 <= ct <= MAX_CON_TYPE:
                    sys.exit(f"con_type {ct} does not fit {KEY_CON_TYPE_BITS} bits")

                attention = np.float32(cn[4])
                node_key[base + next_local] = pack_key(at, ca, ct)
                node_source[base + next_local] = c
                node_attn[base + next_local] = attention
                cum[next_local] = cum[local] + (float(attention) if accumulate
                                                else 0.0)
                next_local += 1

        if next_local != count:
            sys.exit(f"branch {b}: renumbering covered {next_local} of {count} "
                     f"nodes -- the tree is not a single rooted tree")

        stop = cum > DEFAULT_ATTENTION_THRESHOLD
        node_flags[base:base + count] = np.where(stop, NODE_FLAG_STOP,
                                                 0).astype(np.uint8)
        n_stop += int(stop.sum())
        del tree, cum

        # the property columns, indexed by the new numbering
        d = pd.read_hdf(os.path.join(folder, f"{b}.h5"), key="df", mode="r")
        if len(d) != count:
            sys.exit(f"branch {b}: {count} nodes but {len(d)} data rows")
        rows = node_source[base:base + count]
        for arr, col, dt in zip(prop_arrays, props, prop_dtypes):
            arr[base:base + count] = d[col].to_numpy()[rows].astype(
                _NP_DTYPE[dt], copy=False)
        del d

    print(f"  nodes past the default attention threshold: {n_stop:,}")
    print(f"key ranges: atom_type <= {max_atom_type}   "
          f"con_atom [{min_con_atom},{max_con_atom}]   "
          f"con_type [{min_con_type},{max_con_type}]   "
          f"max children {max_children}")

    # ---- write -------------------------------------------------------------
    records = np.zeros(n_nodes, dtype=RECORD_DTYPE)
    records["key"] = node_key[:n_nodes]
    records["nChildren"] = node_children[:n_nodes]
    records["flags"] = node_flags[:n_nodes]
    records["firstChild"] = node_first[:n_nodes]

    try:
        blocks = write_container(
            args.out, branch_root, feature_table, records, node_attn[:n_nodes],
            None if args.no_source_ids else node_source[:n_nodes],
            list(zip(props, prop_dtypes, prop_arrays)),
            DEFAULT_ATTENTION_THRESHOLD, (max_atom_type, max_con_atom, max_con_type))
    except ValueError as e:
        sys.exit(str(e))

    print(f"\nwrote {args.out}  {os.path.getsize(args.out)/1e6:.1f} MB")
    for name, arr, off in blocks:
        print(f"    {name:<28} n={arr.size:>10,}  {arr.nbytes/1e6:>7.1f} MB @ {off}")


if __name__ == "__main__":
    main()
