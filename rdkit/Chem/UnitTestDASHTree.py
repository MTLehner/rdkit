#
#  Copyright (C) 2026 Marc Lehner and other RDKit contributors
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
"""Tests for rdkit.Chem.DASHTree and rdkit.Chem.rdDASHTree.

Most of these run against a small synthetic tree, test_data/dash_test_tree.dash,
written by _writeSyntheticTree() below and checked in so the C++ tests can share
it. It knows ethanol and nothing else, and its values are made up, so what it
exercises is the machinery: the descent, the property fallback, the stop flag,
the normalisations, the node API and the pattern writer.

The published tree is a few hundred megabytes and is not in the repository; the
last class runs against it when DASH_TREE_FILE points at a converted container
and is skipped otherwise.
"""

import contextlib
import math
import os
import shutil
import struct
import tempfile
import unittest

from rdkit import Chem, RDConfig
from rdkit.Chem import DASHTree, rdDASHTree

# ethanol with explicit hydrogens is C0 H3 H4 H5, C1 H6 H7, O2 H8. The synthetic
# tree carries exactly its four atom classes, so these are their branch indices;
# in the published trees the same classes are branches 34, 33, 83 and 37.
M, P, O, H = 0, 1, 2, 3
FEATURES = [(6, 4, 0, False, 3), (6, 4, 0, False, 2), (8, 2, 0, False, 1), (1, 1, 0, False, 0)]
STOP = 0x01
NAN = float("nan")

# The synthetic tree, one branch per class. A node is (atomFeature, conAtom,
# conType, firstChild, numChildren, attention, flags, result, std), with
# firstChild relative to the branch root. The values are arbitrary but distinct,
# so a wrong node shows up as a wrong number.
BRANCHES = {
  # methyl C: C0 -> C1 -> O2 -> H8, with no value on the deepest node so the
  # property walk has to fall back to the O node
  M: [(M, -1, -1, 1, 1, 0.0, 0, -0.30, 0.05), (P, 0, 1, 2, 1, 1.0, 0, -0.40, 0.04),
      (O, 1, 1, 3, 1, 1.0, 0, -0.45, 0.03), (H, 2, 1, 0, 0, 1.0, 0, NAN, NAN)],
  # CH2: C1 -> C0 -> O2; the root's second child is never taken
  P: [(P, -1, -1, 1, 2, 0.0, 0, -0.10, 0.10), (M, 0, 1, 3, 1, 1.0, 0, -0.15, 0.10),
      (H, 0, 1, 0, 0, 1.0, 0, -0.99, 0.10), (O, 0, 1, 0, 0, 1.0, 0, -0.18, 0.02)],
  # O: the root's first child is the hydrogen, so a descent that scanned the
  # candidates rather than the children would take C1 instead. Its attention
  # crosses the default threshold, so a default descent stops there and only a
  # caller raising the threshold reaches the node below it.
  O: [(O, -1, -1, 1, 2, 0.0, 0, -0.60, 0.06), (H, 0, 1, 3, 1, 11.0, STOP, -0.65, 0.05),
      (P, 0, 1, 0, 0, 1.0, 0, -0.99, 0.05), (P, 0, 1, 0, 0, 1.0, STOP, -0.70, 0.04)],
  # H: matched through the heavy neighbour, which attaches to nothing
  H: [(H, -1, -1, 1, 3, 0.0, 0, 0.05, 0.02), (M, -1, -1, 4, 1, 1.0, 0, 0.10, 0.02),
      (P, -1, -1, 0, 0, 1.0, 0, 0.08, 0.02), (O, -1, -1, 5, 1, 1.0, 0, 0.30, 0.02),
      (P, 0, 1, 0, 0, 1.0, 0, 0.12, 0.02), (P, 0, 1, 0, 0, 1.0, 0, 0.31, 0.02)],
}


def _writeSyntheticTree(path):
  """Writes BRANCHES as a version 3 '.dash' container, float64 columns."""
  numBranches, roots, nodes = len(FEATURES), [], []
  for b in range(numBranches):
    base = len(nodes)
    roots.append(base)
    for feat, conAtom, conType, first, nChildren, attn, flags, result, std in BRANCHES[b]:
      key = feat | ((conAtom + 1) << 8) | ((conType + 1) << 12)
      nodes.append((key, nChildren, flags, base + first if nChildren else 0, attn, result, std))

  def alignUp(n):
    return (n + 63) // 64 * 64

  header, entry, n = 128, 48, len(nodes)
  offRoots = alignUp(header + 2 * entry)
  offFeatures = alignUp(offRoots + 4 * numBranches)
  offNodes = alignUp(offFeatures + 8 * numBranches)
  offAttn = alignUp(offNodes + 8 * n)
  offResult = alignUp(offAttn + 4 * n)
  offStd = alignUp(offResult + 8 * n)
  size = alignUp(offStd + 8 * n)
  out = bytearray(size)
  struct.pack_into("<8sIIIIIIdQQQQQQhhhHQ", out, 0, b"DASHTREE", 3, 0xDEADBEEF, numBranches, 2, n,
                   3, 10.0, offRoots, offNodes, offAttn, 0, header, size, 3, 2, 1, numBranches,
                   offFeatures)
  for i, (name, off) in enumerate((("result", offResult), ("std", offStd))):
    struct.pack_into("<32sB7xQ", out, header + i * entry, name.encode(), 4, off)
  struct.pack_into(f"<{numBranches}I", out, offRoots, *roots)
  for i, (z, degree, charge, conjugated, numHs) in enumerate(FEATURES):
    struct.pack_into("<BBbBB3x", out, offFeatures + 8 * i, z, degree, charge, conjugated, numHs)
  for i, (key, nChildren, flags, first, attn, result, std) in enumerate(nodes):
    struct.pack_into("<HBBI", out, offNodes + 8 * i, key, nChildren, flags, first)
    struct.pack_into("<f", out, offAttn + 4 * i, attn)
    struct.pack_into("<d", out, offResult + 8 * i, result)
    struct.pack_into("<d", out, offStd + 8 * i, std)
  with open(path, "wb") as f:
    f.write(out)


def _fixturePath():
  return os.path.join(RDConfig.RDBaseDir, "Code", "GraphMol", "DASHTree", "test_data",
                      "dash_test_tree.dash")


class TestSyntheticTree(unittest.TestCase):

  def setUp(self):
    self.tree = rdDASHTree.DASHTree(_fixturePath(), ["result", "std"])
    self.mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))

  def testFixtureIsReproducible(self):
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "tree.dash")
      _writeSyntheticTree(path)
      with open(path, "rb") as f, open(_fixturePath(), "rb") as g:
        self.assertEqual(f.read(), g.read())

  def testFeatureIndices(self):
    # the classes come from the table in the file, not from the code
    self.assertEqual(self.tree.GetNumAtomFeatures(), 4)
    got = [self.tree.GetAtomFeatureIndex(a) for a in self.mol.GetAtoms()]
    self.assertEqual(got, [M, P, O, H, H, H, H, H, H])
    self.assertEqual([self.tree.GetAtomFeature(i) for i in range(4)], FEATURES)
    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    self.assertEqual(self.tree.GetAtomFeatureIndex(benzene.GetAtomWithIdx(0)), -1)
    with self.assertRaises(ValueError):
      self.tree.GetAtomFeature(4)

  def testDescent(self):
    self.assertEqual(self.tree.GetNumBranches(), 4)
    self.assertEqual(self.tree.GetNumNodes(), 18)
    expected = {
      0: [M, 0, 1, 2, 3],
      1: [P, 0, 1, 3],
      2: [O, 0, 1],
      3: [H, 0, 1, 4],
      6: [H, 0, 2],
      8: [H, 0, 3, 5]
    }
    for atom, path in expected.items():
      self.assertEqual(self.tree.GetAtomNodePath(self.mol, atom), path)
    self.assertEqual(self.tree.GetMatchedSubstructure(self.mol, 0), (0, 1, 2, 8))
    self.assertEqual(self.tree.GetMatchedSubstructure(self.mol, 3), (0, 1))
    # the O descent stops at the flagged node; told to read the attentions
    # instead of the flag, it carries on to the node below
    params = rdDASHTree.DASHParams()
    params.attentionThreshold = 100.0
    self.assertEqual(self.tree.GetAtomNodePath(self.mol, 2, params), [O, 0, 1, 3])
    self.assertAlmostEqual(self.tree.GetAtomProperty(self.mol, 2, "result"), -0.65)
    self.assertAlmostEqual(self.tree.GetAtomProperty(self.mol, 2, "result", params), -0.70)

  def testCharges(self):
    raw = [-0.45, -0.18, -0.65, 0.12, 0.12, 0.12, 0.08, 0.08, 0.31]  # deepest node with a value
    std = [0.03, 0.02, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    deficit = 0.0 - sum(raw)
    details = self.tree.GetPartialChargesDetails(self.mol)
    for got, want in zip(details["raw"], raw):
      self.assertAlmostEqual(got, want)
    for got, want in zip(details["std"], std):
      self.assertAlmostEqual(got, want)
    self.assertEqual(details["match_depth"], [4, 3, 2, 3, 3, 3, 2, 2, 3])
    N = rdDASHTree.ChargeNormalization
    for normalization, want in ((N.NONE, raw), (N.SYMMETRIC, [r + deficit / 9 for r in raw]),
                                (N.STD_WEIGHTED,
                                 [r + deficit * s / sum(std) for r, s in zip(raw, std)])):
      options = rdDASHTree.ChargeOptions()
      options.normalization = normalization
      for got, w in zip(self.tree.GetPartialCharges(self.mol, options), want):
        self.assertAlmostEqual(got, w)
    # a batch agrees with the single-molecule call, and None yields an empty row
    batch = self.tree.GetPartialChargesBatch([self.mol, None, self.mol], numThreads=0)
    self.assertEqual(batch[1], [])
    self.assertEqual(batch[0], self.tree.GetPartialCharges(self.mol))
    self.assertEqual(batch[0], batch[2])

  def testRefusals(self):
    with self.assertRaises(ValueError):  # methane's carbon is not a DASH class
      self.tree.GetPartialCharges(Chem.AddHs(Chem.MolFromSmiles("C")))
    with self.assertRaises(ValueError):  # a class the synthetic tree has no data for
      self.tree.GetPartialCharges(Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")))
    with self.assertRaises(ValueError):
      rdDASHTree.DASHTree(_fixturePath(), ["result", "definitely_absent"])

  def testNodes(self):
    root = self.tree.GetRoot(O)
    self.assertEqual((root.GetBranch(), root.GetId(), root.GetAtomFeatureIndex()), (O, 0, O))
    self.assertEqual((root.GetConAtom(), root.GetConType()), (-1, -1))
    self.assertEqual(root.GetFeature(), (8, 2, 0, False, 1))
    self.assertEqual(len(root), 2)
    first, second = root
    self.assertEqual((first.GetAtomFeatureIndex(), first.GetConAtom(), first.GetConType()),
                     (H, 0, 1))
    self.assertAlmostEqual(first.GetAttention(), 11.0)
    self.assertTrue(first.Stops())
    self.assertFalse(second.Stops())
    self.assertAlmostEqual(second.GetValue("result"), -0.99)
    self.assertEqual(root[-1].GetId(), second.GetId())
    self.assertEqual(first[0].GetId(), 3)
    self.assertTrue(math.isnan(self.tree.GetNode(M, 3).GetValue("result")))
    with self.assertRaises(IndexError):
      root[2]
    with self.assertRaises(ValueError):
      root.GetChild(2)
    with self.assertRaises(ValueError):
      self.tree.GetNode(4, 0)
    with self.assertRaises(ValueError):
      root.GetValue("no_such_property")

  def testPatterns(self):
    self.assertEqual(DASHTree.GetAtomMatchSmarts(self.tree, self.mol, 0),
                     "[#6X4H3+0:1]-[#6X4H2+0]-[#8X2H1+0]")
    path = self.tree.GetAtomNodePath(self.mol, 0)
    self.assertEqual(DASHTree.NodePathToSmarts(self.tree, path, foldHydrogens=False),
                     "[#6X4H3+0:1]-[#6X4H2+0]-[#8X2H1+0]-[#1X1H0+0]")
    self.assertEqual(DASHTree.GetAtomMatchSmarts(self.tree, self.mol, 8),
                     "[#1X1H0+0:1]-[#8X2H1+0]-[#6X4H2+0]")
    unfolded = DASHTree.NodePathToQueryMol(self.tree, path, foldHydrogens=False)
    self.assertIn((0, 1, 2, 8), self.mol.GetSubstructMatches(unfolded, uniquify=False))
    folded = DASHTree.NodePathToQueryMol(self.tree, path)
    self.assertTrue(Chem.MolFromSmiles("CCO").HasSubstructMatch(folded))


class TestPruneTool(unittest.TestCase):
  """tools/dash_prune.py, on the synthetic tree."""

  @classmethod
  def setUpClass(cls):
    import importlib.util
    toolPath = os.path.join(RDConfig.RDBaseDir, "Code", "GraphMol", "DASHTree", "tools",
                            "dash_prune.py")
    spec = importlib.util.spec_from_file_location("dash_prune", toolPath)
    cls.prune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cls.prune)
    cls.mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cls.full = rdDASHTree.DASHTree(_fixturePath(), ["result", "std"])
    # a mapped container cannot be deleted while a tree holds it, so the
    # directory outlives the tests rather than each one
    cls.tmpDir = tempfile.mkdtemp()

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.tmpDir, ignore_errors=True)

  def _pruned(self, *args):
    out = os.path.join(self.tmpDir, f"{self._testMethodName}.dash")
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
      self.prune.main([_fixturePath(), out, *args])
    return rdDASHTree.DASHTree(out, ["result", "std"])

  def testDepthCut(self):
    tree = self._pruned("--max-level", "1")
    self.assertEqual(tree.GetNumNodes(), 12)
    self.assertEqual(tree.GetNumAtomFeatures(), 4)
    # every descent now ends one level below the root; a hydrogen still finds
    # its heavy neighbour there, and the charges still sum to the formal charge
    self.assertEqual(tree.GetPartialChargesDetails(self.mol)["match_depth"], [2] * 9)
    self.assertEqual([c.GetConAtom() for c in tree.GetRoot(H)], [-1, -1, -1])
    self.assertAlmostEqual(sum(tree.GetPartialCharges(self.mol)), 0.0, places=12)

  def testMoleculePathsStayExact(self):
    sdf = os.path.join(self.tmpDir, "ethanol.sdf")
    with Chem.SDWriter(sdf) as w:
      w.write(self.mol)
    tree = self._pruned("--molecules", sdf)
    # only the nodes ethanol descends through survive, and they are enough
    self.assertEqual(tree.GetNumNodes(), 15)
    self.assertEqual(tree.GetPartialCharges(self.mol), self.full.GetPartialCharges(self.mol))
    for atom in range(self.mol.GetNumAtoms()):
      self.assertEqual(len(tree.GetAtomNodePath(self.mol, atom)),
                       len(self.full.GetAtomNodePath(self.mol, atom)))

  def testToleranceCut(self):
    # Under the O root (-0.60) the H child's subtree (-0.65, -0.70) is within
    # 0.3 e and goes; the P child (-0.99) is not and stays. Were the H child
    # simply gone, the O atom's descent would take the P child instead -- its
    # key matches the neighbouring C -- and read -0.99. The placeholder left in
    # its place makes the descent stop there and read the root.
    tree = self._pruned("--tolerance", "0.3")
    self.assertLess(tree.GetNumNodes(), 18)
    self.assertEqual(tree.GetAtomNodePath(self.mol, 2), [O, 0, 1])
    self.assertTrue(math.isnan(tree.GetNode(O, 1).GetValue("result")))
    self.assertAlmostEqual(tree.GetAtomProperty(self.mol, 2, "result"), -0.60)


treeFile = os.environ.get("DASH_TREE_FILE")


@unittest.skipUnless(treeFile and os.path.isfile(treeFile),
                     "set DASH_TREE_FILE to a converted published tree")
class TestPublishedTree(unittest.TestCase):

  def setUp(self):
    self.tree = rdDASHTree.DASHTree(treeFile, ["result", "std"])
    self.mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))

  def testChargesSumToFormalCharge(self):
    self.assertAlmostEqual(sum(self.tree.GetPartialCharges(self.mol)), 0.0, places=12)

  def testEveryPathNodeIsAChildOfTheOneBefore(self):
    path = self.tree.GetAtomNodePath(self.mol, 0)
    node = self.tree.GetRoot(path[0])
    for nodeId in path[2:]:
      self.assertIn(nodeId, [child.GetId() for child in node])
      node = self.tree.GetNode(path[0], nodeId)

  def testPatternRefindsItsOwnAtom(self):
    for atom in self.mol.GetAtoms():
      smarts = DASHTree.GetAtomMatchSmarts(self.tree, self.mol, atom.GetIdx())
      query = Chem.MolFromSmarts(smarts)
      mapped = next(a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum() == 1)
      landings = {m[mapped] for m in self.mol.GetSubstructMatches(query, uniquify=False)}
      self.assertIn(atom.GetIdx(), landings, smarts)


if __name__ == '__main__':
  unittest.main()
