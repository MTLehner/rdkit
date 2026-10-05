#
#  Copyright (C) 2026 Marc Lehner and other RDKit contributors
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
"""The .dash container as the tools see it: constants, a reader and the writer.

dash_convert.py and dash_prune.py both produce containers, so the layout lives
here once; the C++ side of it is the structs in ../DASHTreeImpl.h and the
reader in ../DASHTree.cpp. Little-endian throughout, every array 64-byte
aligned.
"""

import struct

import numpy as np

MAGIC = b"DASHTREE"
FORMAT_VERSION = 3
ENDIAN_ID = 0xDEADBEEF

HEADER_SIZE = 128
HEADER_FORMAT = "<8sIIIIIIdQQQQQQ"
KEY_RANGE_FORMAT = "<hhh"
FEATURE_TABLE_FORMAT = "<HQ"  # numFeatures, offFeatureTable
PROP_ENTRY_SIZE = 48
PROP_NAME_LEN = 32
ALIGN = 64

RECORD_DTYPE = np.dtype([("key", "<u2"), ("nChildren", "u1"), ("flags", "u1"),
                         ("firstChild", "<u4")])
NODE_RECORD_SIZE = RECORD_DTYPE.itemsize
FEATURE_DTYPE = np.dtype([("atomicNum", "u1"), ("degree", "u1"), ("formalCharge", "i1"),
                          ("conjugated", "u1"), ("numHs", "u1"), ("reserved", "3u1")])

DTYPE_F16, DTYPE_F32, DTYPE_I32, DTYPE_F64 = 1, 2, 3, 4
NP_DTYPE = {
  DTYPE_F16: np.dtype("<f2"),
  DTYPE_F32: np.dtype("<f4"),
  DTYPE_I32: np.dtype("<i4"),
  DTYPE_F64: np.dtype("<f8"),
}

# key packing, mirroring AtomFeatures.h
KEY_ATOM_BITS, KEY_CON_ATOM_BITS, KEY_CON_TYPE_BITS = 8, 4, 3
MAX_ATOM_TYPE = (1 << KEY_ATOM_BITS) - 1  # 255
MAX_CON_ATOM = (1 << KEY_CON_ATOM_BITS) - 2  # 14, stored as conAtom + 1
MAX_CON_TYPE = (1 << KEY_CON_TYPE_BITS) - 2  # 6,  stored as conType + 1

MAX_CHILDREN = 255  # numChildren is one byte
MAX_NODES = 0xFFFFFFFF  # firstChild is four bytes

NODE_FLAG_STOP = 0x01

# the cumulative-attention threshold the stop flag is precomputed for
DEFAULT_ATTENTION_THRESHOLD = 10.0


def pack_key(atom_type, con_atom, con_type):
  return (atom_type | ((con_atom + 1) << KEY_ATOM_BITS) |
          ((con_type + 1) << (KEY_ATOM_BITS + KEY_CON_ATOM_BITS)))


def unpack_key(key):
  """(atomType, conAtom, conType) of a packed key; works on arrays too."""
  key = np.asarray(key, dtype=np.int64)
  atom = key & MAX_ATOM_TYPE
  con_atom = ((key >> KEY_ATOM_BITS) & ((1 << KEY_CON_ATOM_BITS) - 1)) - 1
  con_type = ((key >> (KEY_ATOM_BITS + KEY_CON_ATOM_BITS)) & ((1 << KEY_CON_TYPE_BITS) - 1)) - 1
  return atom, con_atom, con_type


def align_up(n):
  return (n + ALIGN - 1) // ALIGN * ALIGN


class Container:
  """A mapped .dash file, its arrays exposed as read-only numpy views.

  Attributes: n_branches, n_nodes, max_children, threshold, key_ranges,
  branch_root (uint32), features (FEATURE_DTYPE, one per branch), records
  (RECORD_DTYPE), attn (float32), source (uint32 or None) and props, an ordered
  dict name -> (dtype tag, array).
  """

  def __init__(self, path):
    self.path = path
    self._mm = np.memmap(path, dtype=np.uint8, mode="r")
    if self._mm.size < HEADER_SIZE:
      raise ValueError(f"{path}: too small to be a DASH container")
    (magic, version, endian, self.n_branches, n_props, self.n_nodes, self.max_children,
     self.threshold, off_roots, off_rec, off_attn, off_source, off_dir,
     file_size) = struct.unpack_from(HEADER_FORMAT, self._mm, 0)
    if magic != MAGIC or endian != ENDIAN_ID:
      raise ValueError(f"{path}: not a DASH container")
    if version != FORMAT_VERSION:
      raise ValueError(f"{path}: container version {version}, these tools read {FORMAT_VERSION}")
    if file_size != self._mm.size:
      raise ValueError(f"{path}: header says {file_size} bytes, file has {self._mm.size}")
    self.key_ranges = struct.unpack_from(KEY_RANGE_FORMAT, self._mm,
                                         struct.calcsize(HEADER_FORMAT))
    n_features, off_features = struct.unpack_from(
      FEATURE_TABLE_FORMAT, self._mm,
      struct.calcsize(HEADER_FORMAT) + struct.calcsize(KEY_RANGE_FORMAT))
    if n_features != self.n_branches:
      raise ValueError(f"{path}: {n_features} atom-feature classes for {self.n_branches} branches")
    self.branch_root = self._view(off_roots, np.dtype("<u4"), self.n_branches)
    self.features = self._view(off_features, FEATURE_DTYPE, n_features)
    self.records = self._view(off_rec, RECORD_DTYPE, self.n_nodes)
    self.attn = self._view(off_attn, np.dtype("<f4"), self.n_nodes)
    self.source = self._view(off_source, np.dtype("<u4"), self.n_nodes) if off_source else None
    self.props = {}
    for i in range(n_props):
      entry = off_dir + i * PROP_ENTRY_SIZE
      name = bytes(self._mm[entry:entry + PROP_NAME_LEN]).rstrip(b"\0").decode("utf-8")
      tag, off = struct.unpack_from("<B7xQ", self._mm, entry + PROP_NAME_LEN)
      if tag not in NP_DTYPE:
        raise ValueError(f"{path}: property '{name}' has unknown dtype tag {tag}")
      self.props[name] = (tag, self._view(off, NP_DTYPE[tag], self.n_nodes))

  def _view(self, offset, dtype, count):
    end = offset + dtype.itemsize * count
    if offset % ALIGN or end > self._mm.size:
      raise ValueError(f"{self.path}: array at {offset} is misaligned or out of range")
    return np.frombuffer(self._mm, dtype=dtype, count=count, offset=offset)

  def parent_and_level(self):
    """parent[n] (-1 at a root) and level[n] (0 at a root) for every node."""
    rec = self.records
    parent = np.full(self.n_nodes, -1, dtype=np.int64)
    level = np.full(self.n_nodes, -1, dtype=np.int16)
    frontier = self.branch_root.astype(np.int64)
    level[frontier] = 0
    depth = 0
    while frontier.size:
      depth += 1
      has = rec["nChildren"][frontier] > 0
      parents = frontier[has]
      counts = rec["nChildren"][parents].astype(np.int64)
      starts = rec["firstChild"][parents].astype(np.int64)
      children = (np.repeat(starts, counts) + np.arange(counts.sum()) -
                  np.repeat(np.cumsum(counts) - counts, counts))
      parent[children] = np.repeat(parents, counts)
      level[children] = depth
      frontier = children
    if (level < 0).any():
      raise ValueError(f"{self.path}: {int((level < 0).sum())} nodes are reachable from no root")
    return parent, level

  def hydrogen_branch(self):
    """The branch matched through the heavy neighbour: the class with Z = 1.

    Its root's children attach with conAtom = -1, the "no bond" position, and
    no other node in a published tree does; a table that disagrees with the
    data on this is an error, since the tree was then built on another table.
    """
    from_table = [b for b in range(self.n_branches) if self.features["atomicNum"][b] == 1]
    from_data = []
    for b in range(self.n_branches):
      root = self.records[self.branch_root[b]]
      n = int(root["nChildren"])
      if n:
        first = int(root["firstChild"])
        _, con_atom, _ = unpack_key(self.records["key"][first:first + n])
        if (con_atom == -1).all():
          from_data.append(b)
    if len(from_table) != 1:
      raise ValueError(f"{self.path}: {len(from_table)} atom-feature classes with Z = 1")
    if from_data and from_data != from_table:
      raise ValueError(f"{self.path}: the feature table puts hydrogen at branch "
                       f"{from_table[0]}, the data at {from_data}")
    return from_table[0]


def write_container(path, branch_root, features, records, attn, source, props, threshold,
                    key_ranges):
  """Writes a version 3 container.

  features has one FEATURE_DTYPE row per branch; props is a list of (name,
  dtype tag, array); source may be None to omit the map back to the source
  numbering. Returns the blocks written as a list of (name, array, offset), in
  file order, for reporting.
  """
  branch_root = np.ascontiguousarray(branch_root, dtype="<u4")
  features = np.ascontiguousarray(features, dtype=FEATURE_DTYPE)
  if features.size != branch_root.size:
    raise ValueError(f"{features.size} atom-feature classes for {branch_root.size} branches")
  records = np.ascontiguousarray(records, dtype=RECORD_DTYPE)
  attn = np.ascontiguousarray(attn, dtype="<f4")
  n_nodes = records.size
  if branch_root.size == 0 or n_nodes == 0:
    raise ValueError("a container needs at least one branch and one node")
  if attn.size != n_nodes or (source is not None and len(source) != n_nodes):
    raise ValueError("every per-node array must have one entry per node")
  if int(records["nChildren"].max()) > MAX_CHILDREN:
    raise ValueError(f"a node has more than {MAX_CHILDREN} children")

  blocks = [("branchRoot", branch_root), ("featureTable", features), ("nodeRecord", records),
            ("nodeAttn", attn)]
  if source is not None:
    blocks.append(("nodeSourceId", np.ascontiguousarray(source, dtype="<u4")))
  for name, tag, arr in props:
    if len(name.encode("utf-8")) >= PROP_NAME_LEN:
      raise ValueError(f"property name too long: {name}")
    if tag not in NP_DTYPE:
      raise ValueError(f"unknown dtype tag {tag} for property {name}")
    arr = np.ascontiguousarray(arr, dtype=NP_DTYPE[tag])
    if arr.size != n_nodes:
      raise ValueError(f"property {name} has {arr.size} values for {n_nodes} nodes")
    blocks.append((f"prop:{name}", arr))

  prop_dir_off = HEADER_SIZE
  off = align_up(HEADER_SIZE + PROP_ENTRY_SIZE * len(props))
  offsets = {}
  for name, arr in blocks:
    offsets[name] = off
    off = align_up(off + arr.nbytes)
  file_size = off

  with open(path, "wb") as f:
    f.write(
      struct.pack(HEADER_FORMAT, MAGIC, FORMAT_VERSION, ENDIAN_ID, branch_root.size, len(props),
                  n_nodes, int(records["nChildren"].max()), float(threshold),
                  offsets["branchRoot"], offsets["nodeRecord"], offsets["nodeAttn"],
                  offsets.get("nodeSourceId", 0), prop_dir_off, file_size))
    f.write(struct.pack(KEY_RANGE_FORMAT, *[int(k) for k in key_ranges]))
    f.write(struct.pack(FEATURE_TABLE_FORMAT, features.size, offsets["featureTable"]))
    f.write(b"\0" * (HEADER_SIZE - f.tell()))
    for name, tag, _ in props:
      nm = name.encode("utf-8")
      f.write(nm + b"\0" * (PROP_NAME_LEN - len(nm)))
      f.write(struct.pack("<B7xQ", tag, offsets[f"prop:{name}"]))
    for name, arr in blocks:
      f.write(b"\0" * (offsets[name] - f.tell()))
      f.write(arr.tobytes())
    f.write(b"\0" * (file_size - f.tell()))
  return [(name, arr, offsets[name]) for name, arr in blocks]


def key_ranges_of(records, branch_root):
  """(maxAtomType, maxConAtom, maxConType) over every node that is not a root,
  which is what the converter records in the header."""
  is_child = np.ones(records.size, dtype=bool)
  is_child[np.asarray(branch_root, dtype=np.int64)] = False
  if not is_child.any():
    return -1, -1, -1
  atom, con_atom, con_type = unpack_key(records["key"][is_child])
  return int(atom.max()), int(con_atom.max()), int(con_type.max())
