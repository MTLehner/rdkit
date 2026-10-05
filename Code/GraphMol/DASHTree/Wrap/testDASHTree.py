#
#  Copyright (C) 2026 Marc Lehner and other RDKit contributors
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
"""Tests for rdkit.Chem.rdDASHTree.

Most run against test_data/dash_test_tree.dash, a synthetic tree written by
rdkit/Chem/UnitTestDASHTree.py that knows ethanol and nothing else: its values
are made up, so what it exercises is the machinery -- the descent, the property
fallback, the stop flag, the normalisations, the node API. A published tree is
a couple of hundred megabytes and not in the repository, so the tests that need
one are skipped unless DASH_TREE_FILE points at a container.
"""

import math
import os
import unittest

from rdkit import Chem, RDConfig
from rdkit.Chem import rdDASHTree

TEST_DATA_DIR = os.path.join(RDConfig.RDBaseDir, 'Code', 'GraphMol', 'DASHTree',
                             'test_data')
FIXTURE = os.path.join(TEST_DATA_DIR, 'dash_test_tree.dash')

# ethanol with explicit hydrogens is C0 H3 H4 H5, C1 H6 H7, O2 H8; the synthetic
# tree carries exactly its four atom classes, in this order
M, P, O, H = 0, 1, 2, 3
FEATURES = [(6, 4, 0, False, 3), (6, 4, 0, False, 2), (8, 2, 0, False, 1), (1, 1, 0, False, 0)]


def treeFile():
  path = os.environ.get('DASH_TREE_FILE', '')
  return path if path and os.path.exists(path) else None


def molWithHs(smiles):
  mol = Chem.MolFromSmiles(smiles)
  assert mol is not None, smiles
  return Chem.AddHs(mol)


needsTree = unittest.skipIf(
  treeFile() is None,
  'set DASH_TREE_FILE to a container built by tools/dash_convert.py')


class TestAtomFeatures(unittest.TestCase):
  """The atom classification, read from the table in the synthetic tree."""

  @classmethod
  def setUpClass(cls):
    cls.tree = rdDASHTree.DASHTree(FIXTURE, ['result'])

  def testFeatureTable(self):
    self.assertEqual(self.tree.GetNumAtomFeatures(), 4)
    # hydrogen is the only class with atomic number 1, and has degree one
    hydrogens = [
      i for i in range(self.tree.GetNumAtomFeatures()) if self.tree.GetAtomFeature(i)[0] == 1
    ]
    self.assertEqual(len(hydrogens), 1)
    self.assertEqual(self.tree.GetAtomFeature(hydrogens[0]), (1, 1, 0, False, 0))
    with self.assertRaises(ValueError):
      self.tree.GetAtomFeature(1000)

  def testAtomFeatureIndex(self):
    mol = molWithHs('CCO')
    for atom in mol.GetAtoms():
      self.assertGreaterEqual(self.tree.GetAtomFeatureIndex(atom), 0)
    # silicon is outside the classes any tree covers
    silane = molWithHs('C[Si](C)(C)C')
    self.assertEqual(min(self.tree.GetAtomFeatureIndex(a) for a in silane.GetAtoms()), -1)

  def testFeatureTupleMatchesTheAtom(self):
    mol = molWithHs('CCO')
    idx = self.tree.GetAtomFeatureIndex(mol.GetAtomWithIdx(2))  # the oxygen
    self.assertEqual(self.tree.GetAtomFeature(idx), (8, 2, 0, False, 1))


class TestBadFiles(unittest.TestCase):

  def testMissingFile(self):
    with self.assertRaises(Exception):
      rdDASHTree.DASHTree('no_such_file_at_all.dash')


class TestSyntheticTree(unittest.TestCase):
  """The machinery, on the fixture: every value it holds is known."""

  def setUp(self):
    self.tree = rdDASHTree.DASHTree(FIXTURE, ["result", "std"])
    self.mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))

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

  def testNormalizedMolProperty(self):
    # charges are the normalised assignment with the formal charge as target
    charges = self.tree.GetPartialCharges(self.mol)
    self.assertEqual(self.tree.GetNormalizedMolProperty(self.mol, 0.0), charges)
    self.assertAlmostEqual(sum(self.tree.GetNormalizedMolProperty(self.mol, 1.0)), 1.0, places=12)
    details = self.tree.GetNormalizedMolPropertyDetails(self.mol, 1.0)
    self.assertEqual(sorted(details), ["match_depth", "raw", "std", "values"])
    self.assertEqual(details["raw"], self.tree.GetPartialChargesDetails(self.mol)["raw"])
    batch = self.tree.GetNormalizedMolPropertyBatch([self.mol, None, self.mol], [0.0, 0.0, 1.0])
    self.assertEqual(batch[0], charges)
    self.assertEqual(batch[1], [])
    self.assertAlmostEqual(sum(batch[2]), 1.0, places=12)
    with self.assertRaises(ValueError):  # one target per molecule
      self.tree.GetNormalizedMolPropertyBatch([self.mol], [0.0, 1.0])
    # the options class and the normalisation enum keep their charge names too
    self.assertIs(rdDASHTree.ChargeOptions, rdDASHTree.NormalizationOptions)
    self.assertIs(rdDASHTree.ChargeNormalization, rdDASHTree.Normalization)
    options = rdDASHTree.NormalizationOptions()
    options.normalization = rdDASHTree.Normalization.NONE
    self.assertEqual(self.tree.GetNormalizedMolProperty(self.mol, 1.0, options), details["raw"])
    # a column without a deviation column still normalises, except by weight
    options.stdProperty = "no_such_column"
    options.normalization = rdDASHTree.Normalization.SYMMETRIC
    self.assertAlmostEqual(sum(self.tree.GetNormalizedMolProperty(self.mol, 1.0, options)), 1.0,
                           places=12)
    options.normalization = rdDASHTree.Normalization.STD_WEIGHTED
    with self.assertRaises(ValueError):
      self.tree.GetNormalizedMolProperty(self.mol, 1.0, options)
    options.stdProperty = "std"
    options.defaultStdValue = 0.0
    with self.assertRaises(ValueError):  # would divide by the deviation total
      self.tree.GetNormalizedMolProperty(self.mol, 1.0, options)

  def testRefusals(self):
    with self.assertRaises(ValueError):  # methane's carbon is not a DASH class
      self.tree.GetPartialCharges(Chem.AddHs(Chem.MolFromSmiles("C")))
    with self.assertRaises(ValueError):  # a class the synthetic tree has no data for
      self.tree.GetPartialCharges(Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")))
    with self.assertRaises(ValueError):
      rdDASHTree.DASHTree(FIXTURE, ["result", "definitely_absent"])

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


@needsTree
class TestDASHTree(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tree = rdDASHTree.DASHTree(treeFile(), ['result', 'std'])

  def testIntrospection(self):
    self.assertEqual(self.tree.GetNumBranches(), self.tree.GetNumAtomFeatures())
    self.assertGreater(self.tree.GetNumNodes(), 0)
    self.assertGreater(self.tree.GetMappedSize(), 0)
    self.assertEqual(sorted(self.tree.GetPropertyNames()), ['result', 'std'])
    self.assertTrue(self.tree.HasProperty('result'))
    self.assertFalse(self.tree.HasProperty('nope'))
    self.assertIn('result', self.tree.GetAvailablePropertyNames())
    self.assertEqual(self.tree.GetFileName(), treeFile())

  def testMissingPropertyIsRefused(self):
    with self.assertRaises(ValueError):
      rdDASHTree.DASHTree(treeFile(), ['result', 'definitely_absent'])
    with self.assertRaises(ValueError):
      self.tree.GetAtomProperty(molWithHs('CCO'), 0, 'nope')

  def testEthanolCharges(self):
    charges = self.tree.GetPartialCharges(molWithHs('CCO'))
    self.assertEqual(len(charges), 9)
    self.assertAlmostEqual(sum(charges), 0.0, places=12)
    # the carbinol carbon is the least negative of the two carbons
    self.assertLess(charges[2], charges[0])  # O more negative than C

  def testChargesSumToFormalCharge(self):
    for smiles in ('CCO', 'CC(=O)Nc1ccc(O)cc1', 'CC(=O)[O-]', 'C[NH3+]',
                   'CN1C=NC2=C1C(=O)N(C)C(=O)N2C'):
      mol = molWithHs(smiles)
      formal = sum(a.GetFormalCharge() for a in mol.GetAtoms())
      for norm in (rdDASHTree.ChargeNormalization.SYMMETRIC,
                   rdDASHTree.ChargeNormalization.STD_WEIGHTED):
        options = rdDASHTree.ChargeOptions()
        options.normalization = norm
        charges = self.tree.GetPartialCharges(mol, options)
        self.assertAlmostEqual(sum(charges), formal, places=12,
                               msg=f'{smiles} {norm}')

  def testUnnormalisedIsRaw(self):
    mol = molWithHs('CC(=O)Nc1ccc(O)cc1')
    options = rdDASHTree.ChargeOptions()
    options.normalization = rdDASHTree.ChargeNormalization.NONE
    details = self.tree.GetPartialChargesDetails(mol, options)
    self.assertEqual(list(details['charges']), list(details['raw']))
    self.assertEqual(list(details['raw']),
                     list(self.tree.GetMolProperty(mol, 'result')))
    self.assertTrue(all(s > 0 for s in details['std']))
    self.assertTrue(all(2 <= d <= 17 for d in details['match_depth']))

  def testSymmetryEquivalentAtoms(self):
    charges = self.tree.GetPartialCharges(molWithHs('c1ccccc1'))
    self.assertEqual(len(set(charges[:6])), 1)
    self.assertEqual(len(set(charges[6:])), 1)

  def testNodePath(self):
    mol = molWithHs('CCO')
    path = self.tree.GetAtomNodePath(mol, 0)
    self.assertGreater(len(path), 1)
    self.assertEqual(path[0], self.tree.GetAtomFeatureIndex(mol.GetAtomWithIdx(0)))
    # a hydrogen descends through its heavy neighbour, so its path is longer
    # than just the root
    hydrogenPath = self.tree.GetAtomNodePath(mol, 3)
    self.assertGreaterEqual(len(hydrogenPath), 3)

  def testEveryPathNodeIsAChildOfTheOneBefore(self):
    mol = molWithHs('CC(=O)Nc1ccc(O)cc1')
    path = self.tree.GetAtomNodePath(mol, 0)
    node = self.tree.GetRoot(path[0])
    for nodeId in path[2:]:
      self.assertIn(nodeId, [child.GetId() for child in node])
      node = self.tree.GetNode(path[0], nodeId)

  def testSourceNodeIdsNeedTheColumn(self):
    if self.tree.HasSourceNodeIds():
      path = self.tree.GetAtomNodePath(molWithHs('CCO'), 0,
                                       rdDASHTree.DASHParams(), True)
      self.assertGreater(len(path), 1)
    else:
      with self.assertRaises(ValueError):
        self.tree.GetAtomNodePath(molWithHs('CCO'), 0, rdDASHTree.DASHParams(),
                                  True)

  def testMaxDepth(self):
    mol = molWithHs('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
    previous = 0
    for maxDepth in (0, 1, 2, 3, 5, 16):
      params = rdDASHTree.DASHParams()
      params.maxDepth = maxDepth
      deepest = 0
      for i in range(mol.GetNumAtoms()):
        path = self.tree.GetAtomNodePath(mol, i, params)
        # a hydrogen spends its first node reaching the heavy atom, so two
        # nodes are always possible however small maxDepth is
        self.assertLessEqual(len(path), 1 + max(2, maxDepth))
        deepest = max(deepest, len(path))
      self.assertGreaterEqual(deepest, previous)
      previous = deepest

  def testStopFlagAgreesWithTheAttentions(self):
    # the default path takes a precomputed flag; nudging the threshold off its
    # default forces the arithmetic, and the two must agree
    for smiles in ('CC(C)Cc1ccc(cc1)C(C)C(=O)O',
                   'CN(C)CCOC(c1ccccc1)c1ccccc1'):
      mol = molWithHs(smiles)
      slow = rdDASHTree.ChargeOptions()
      slow.params.attentionThreshold = 10.000000001
      self.assertEqual(list(self.tree.GetPartialCharges(mol)),
                       list(self.tree.GetPartialCharges(mol, slow)))

  def testUnsupportedAtoms(self):
    for smiles in ('C[Si](C)(C)C', '[Na+].CC(=O)[O-]', 'C'):
      with self.assertRaises(ValueError, msg=smiles):
        self.tree.GetPartialCharges(molWithHs(smiles))

  def testAtomWithNoDataIsRefused(self):
    # this bridging [N+] classifies into a branch holding one node with no value
    mol = molWithHs(
      'COC(=O)C1=CO[C@@H](C)[C@H]2C=[N+]3CCc4c([nH]c5ccccc45)[C@@H]3C[C@H]12')
    with self.assertRaises(ValueError):
      self.tree.GetPartialCharges(mol)
    self.assertTrue(math.isnan(self.tree.GetAtomProperty(mol, 11, 'result')))

  def testStalePropertyCacheIsRefused(self):
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(8))
    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    with self.assertRaises(ValueError):
      self.tree.GetPartialCharges(mol)

  def testOutOfRangeAtomIndex(self):
    mol = molWithHs('CCO')
    with self.assertRaises(ValueError):
      self.tree.GetAtomNodePath(mol, mol.GetNumAtoms())
    with self.assertRaises(ValueError):
      self.tree.GetAtomProperty(mol, mol.GetNumAtoms(), 'result')


@needsTree
class TestBatch(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tree = rdDASHTree.DASHTree(treeFile(), ['result', 'std'])
    cls.mols = [
      molWithHs(s) for s in (
        'CCO', 'CC(=O)Nc1ccc(O)cc1', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        'CN1C=NC2=C1C(=O)N(C)C(=O)N2C', 'CC(=O)Oc1ccccc1C(=O)O',
        'c1ccc2c(c1)[nH]c1ccccc12', 'OC(=O)CCc1c[nH]c2ccccc12',
        'CN(C)CCOC(c1ccccc1)c1ccccc1')
    ]

  def testEveryThreadCountAgrees(self):
    single = [list(self.tree.GetPartialCharges(m)) for m in self.mols]
    for numThreads in (1, 2, 4, 8, 0, -1):
      batch = self.tree.GetPartialChargesBatch(self.mols,
                                               rdDASHTree.ChargeOptions(),
                                               numThreads)
      self.assertEqual([list(r) for r in batch], single, msg=str(numThreads))

  def testNoneEntryGivesAnEmptyRow(self):
    mols = list(self.mols)
    mols.insert(2, None)
    batch = self.tree.GetPartialChargesBatch(mols, rdDASHTree.ChargeOptions(), 4)
    self.assertEqual(len(batch), len(mols))
    self.assertEqual(len(batch[2]), 0)
    self.assertEqual(len(batch[3]), mols[3].GetNumAtoms())

  def testEmptyBatch(self):
    self.assertEqual(
      len(self.tree.GetPartialChargesBatch([], rdDASHTree.ChargeOptions(), 4)), 0)

  def testPropertyBatch(self):
    batch = self.tree.GetMolPropertyBatch(self.mols, 'result',
                                          rdDASHTree.DASHParams(), 4)
    self.assertEqual(len(batch), len(self.mols))
    for row, mol in zip(batch, self.mols):
      self.assertEqual(list(row), list(self.tree.GetMolProperty(mol, 'result')))

  def testAcceptsAnyPythonSequence(self):
    asTuple = self.tree.GetPartialChargesBatch(tuple(self.mols))
    asList = self.tree.GetPartialChargesBatch(self.mols)
    self.assertEqual([list(r) for r in asTuple], [list(r) for r in asList])


@needsTree
class TestReferenceAgreement(unittest.TestCase):
  """Exact agreement with the values the DASH-tree python package produces."""

  def testAgainstStoredReference(self):
    sdf = os.path.join(TEST_DATA_DIR, 'dash_ref_mols.sdf')
    values = os.path.join(TEST_DATA_DIR, 'dash_ref_values.txt')
    if not (os.path.exists(sdf) and os.path.exists(values)):
      self.skipTest('reference fixtures not present')

    tree = rdDASHTree.DASHTree(treeFile(), ['result', 'std'])
    mols = [m for m in Chem.SDMolSupplier(sdf, removeHs=False) if m is not None]
    self.assertGreater(len(mols), 0)

    cache = {}
    for i, mol in enumerate(mols):
      options = rdDASHTree.ChargeOptions()
      options.normalization = rdDASHTree.ChargeNormalization.NONE
      details = tree.GetPartialChargesDetails(mol, options)
      entry = dict(details)
      for name, norm in (('symmetric', rdDASHTree.ChargeNormalization.SYMMETRIC),
                         ('std_weighted',
                          rdDASHTree.ChargeNormalization.STD_WEIGHTED)):
        options.normalization = norm
        entry[name] = tree.GetPartialCharges(mol, options)
      cache[i] = entry

    nChecked = 0
    with open(values) as f:
      for line in f:
        if line.startswith('#'):
          continue
        parts = line.split()
        if not parts:
          continue
        molIdx, atomIdx = int(parts[0]), int(parts[1])
        rawResult, stdUsed = float(parts[2]), float(parts[3])
        depth = int(parts[4])
        qNone, qSym, qStd = float(parts[5]), float(parts[6]), float(parts[7])
        entry = cache[molIdx]
        # the stored values are table lookups, so these must be exact
        self.assertEqual(entry['raw'][atomIdx], rawResult)
        self.assertEqual(entry['std'][atomIdx], stdUsed)
        self.assertEqual(entry['match_depth'][atomIdx], depth)
        self.assertEqual(entry['charges'][atomIdx], qNone)
        # the normalised ones go through a division; allow the last bit
        self.assertAlmostEqual(entry['symmetric'][atomIdx], qSym, delta=1e-15)
        self.assertAlmostEqual(entry['std_weighted'][atomIdx], qStd, delta=1e-15)
        nChecked += 1
    self.assertGreater(nChecked, 0)


if __name__ == '__main__':  # pragma: nocover
  unittest.main()
