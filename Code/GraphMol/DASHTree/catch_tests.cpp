//
//  Copyright (C) 2026 Marc Lehner and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
#include <catch2/catch_all.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <GraphMol/DASHTree/AtomFeatures.h>
#include <GraphMol/DASHTree/DASHTree.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <RDGeneral/BadFileException.h>
#include <RDGeneral/Exceptions.h>

using namespace RDKit;

namespace {
std::unique_ptr<ROMol> molWithHs(const std::string &smiles) {
  std::unique_ptr<ROMol> mol(SmilesToMol(smiles));
  REQUIRE(mol);
  return std::unique_ptr<ROMol>(MolOps::addHs(*mol));
}
}  // namespace

TEST_CASE("DASH atom features", "[DASHTree]") {
  // the classes are a property of the tree, read from the table it carries;
  // the synthetic tree has four of them
  const char *rdbase = std::getenv("RDBASE");
  REQUIRE(rdbase != nullptr);
  DASH::DASHTree tree(
      std::string(rdbase) +
          "/Code/GraphMol/DASHTree/test_data/dash_test_tree.dash",
      {"result"});
  REQUIRE(tree.numAtomFeatures() == 4);
  auto ethanol = molWithHs("CCO");
  CHECK(tree.getAtomFeatureIndex(ethanol->getAtomWithIdx(0)) ==
        tree.getAtomFeatureIndex(6, 4, 0, false, 3));
  CHECK(tree.getAtomFeatureIndex(ethanol->getAtomWithIdx(2)) ==
        tree.getAtomFeatureIndex(8, 2, 0, false, 1));
  CHECK(tree.getAtomFeature(2).atomicNum == 8);
  // an atom the tree does not cover must say so rather than misclassify
  CHECK(tree.getAtomFeatureIndex(14, 4, 0, false, 0) == -1);  // silicon
  CHECK(tree.getAtomFeatureIndex(6, 3, 0, true, 1) == -1);    // aromatic CH
  CHECK_THROWS_AS(tree.getAtomFeature(4), ValueErrorException);
  CHECK(DASH::dashBondType(molWithHs("c1ccccc1")->getBondBetweenAtoms(0, 1)) ==
        4);
}

TEST_CASE("DASH packed match key", "[DASHTree]") {
  // the two ways of building a key must agree, and a component out of range
  // must flag rather than alias onto some real node's key
  CHECK(DASH::packMatchKey(27, 2, 1) ==
        DASH::withConAtom(DASH::packPartialMatchKey(27, 1), 2));
  CHECK(DASH::packMatchKey(27, 0, 17) == DASH::noMatchKey);  // DATIVE
  CHECK(DASH::packMatchKey(DASH::maxKeyAtomType + 1, 0, 1) == DASH::noMatchKey);
  CHECK(DASH::withConAtom(DASH::packPartialMatchKey(27, 1),
                          DASH::maxKeyConAtom + 1) == DASH::noMatchKey);
  CHECK(DASH::packMatchKey(27, 0, 12) != DASH::packMatchKey(27, 0, 4));
  // and a key unpacks to what it was packed from, -1 included
  const std::uint16_t key = DASH::packMatchKey(27, 2, 4);
  CHECK(DASH::keyAtomType(key) == 27);
  CHECK(DASH::keyConAtom(key) == 2);
  CHECK(DASH::keyConType(key) == 4);
  CHECK(DASH::keyConAtom(DASH::packMatchKey(27, -1, -1)) == -1);
  CHECK(DASH::keyConType(DASH::packMatchKey(27, -1, -1)) == -1);
}

TEST_CASE("DASH assignment", "[DASHTree]") {
  CHECK_THROWS_AS(DASH::DASHTree("no_such_file_at_all.dash"), BadFileException);

  // A DASH tree is a couple of hundred megabytes of published data, far too
  // big for the repository. Point DASH_TREE_FILE at a container built by
  // tools/dash_convert.py to run the rest; without one it warns and passes.
  const char *treePath = std::getenv("DASH_TREE_FILE");
  if (!treePath || !*treePath) {
    WARN("set DASH_TREE_FILE to a container to run the DASH assignment tests");
    return;
  }
  const char *rdbase = std::getenv("RDBASE");
  REQUIRE(rdbase != nullptr);
  DASH::DASHTree tree(treePath, {"result", "std"});

  SDMolSupplier suppl(std::string(rdbase) +
                      "/Code/GraphMol/DASHTree/test_data/dash_test_mols.sdf");
  std::vector<std::unique_ptr<ROMol>> mols;
  std::vector<const ROMol *> ptrs;
  while (!suppl.atEnd()) {
    std::unique_ptr<ROMol> mol(suppl.next());
    REQUIRE(mol);
    mols.emplace_back(MolOps::addHs(*mol));
    ptrs.push_back(mols.back().get());
  }
  REQUIRE(mols.size() == 3);

  // the charges of a molecule have to add back up to what it started with
  for (const auto *mol : ptrs) {
    std::vector<double> charges;
    tree.getPartialCharges(*mol, charges);
    REQUIRE(charges.size() == mol->getNumAtoms());
    double total = 0.0;
    for (const auto charge : charges) {
      total += charge;
    }
    CHECK_THAT(total, Catch::Matchers::WithinAbs(MolOps::getFormalCharge(*mol),
                                                 1e-12));
  }

  // and threading may not change an answer
  std::vector<double> single;
  tree.getPartialCharges(*ptrs.back(), single);
  for (const int numThreads : {1, 0}) {  // 0 means every core
    std::vector<std::vector<double>> batch;
    tree.getPartialChargesBatch(ptrs, batch, DASH::ChargeOptions(), numThreads);
    REQUIRE(batch.size() == ptrs.size());
    CHECK(batch.back() == single);
  }

  // the nodes of a match are reachable by the ids the path reports, each is a
  // child of the one before, and the matched atoms line up with them
  const ROMol &ethanol = *ptrs.front();
  std::vector<std::uint32_t> path;
  tree.getAtomNodePath(ethanol, 0, path);
  REQUIRE(path.size() >= 3);
  DASH::DASHTreeNode node = tree.getRoot(path[0]);
  CHECK(node.getId() == path[1]);
  CHECK(node.getAtomFeatureIndex() == static_cast<int>(path[0]));
  CHECK(node.getConAtom() == -1);
  CHECK(node.getConType() == -1);
  for (std::size_t i = 2; i < path.size(); ++i) {
    bool isChild = false;
    for (unsigned int c = 0; c < node.getNumChildren() && !isChild; ++c) {
      isChild = node.getChild(c).getId() == path[i];
    }
    CHECK(isChild);
    node = tree.getNode(path[0], path[i]);
    CHECK(node.getBranch() == path[0]);
    CHECK(node.getConAtom() >= 0);
    CHECK(node.getConType() >= 1);
  }
  CHECK_THROWS_AS(node.getChild(node.getNumChildren()), ValueErrorException);
  CHECK_THROWS_AS(tree.getNode(tree.numBranches(), 0), ValueErrorException);
  CHECK_THROWS_AS(node.getValue("no_such_property"), ValueErrorException);
  // the deepest node on the path carrying a value is what the atom gets
  double deepest = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t i = path.size(); i-- > 1 && std::isnan(deepest);) {
    deepest = tree.getNode(path[0], path[i]).getValue("result");
  }
  CHECK(deepest == tree.getAtomProperty(ethanol, 0, "result"));
  std::vector<unsigned int> atoms;
  tree.getMatchedSubstructure(ethanol, 0, atoms);
  REQUIRE(atoms.size() == path.size() - 1);
  CHECK(atoms.front() == 0);
}

TEST_CASE("DASH integration on the synthetic tree", "[DASHTree]") {
  // test_data/dash_test_tree.dash is written by rdkit/Chem/UnitTestDASHTree.py
  // and checked in: the four atom classes of ethanol and nothing else. Its
  // values are made up, so this checks the machinery -- the descent, the
  // property fallback, the stop flag, the normalisations and the node API --
  // rather than any chemistry.
  const char *rdbase = std::getenv("RDBASE");
  REQUIRE(rdbase != nullptr);
  DASH::DASHTree tree(
      std::string(rdbase) +
          "/Code/GraphMol/DASHTree/test_data/dash_test_tree.dash",
      {"result", "std"});
  CHECK(tree.numBranches() == 4);
  CHECK(tree.numNodes() == 18);
  auto mol = molWithHs("CCO");  // C0 H3 H4 H5, C1 H6 H7, O2 H8
  const std::uint32_t M = 0, P = 1, O = 2, H = 3;  // its four atom classes
  REQUIRE(tree.getAtomFeatureIndex(mol->getAtomWithIdx(2)) ==
          static_cast<int>(O));

  // every atom descends where the tree says, hydrogens through their neighbour
  const std::vector<std::pair<unsigned int, std::vector<std::uint32_t>>> paths =
      {{0, {M, 0, 1, 2, 3}}, {1, {P, 0, 1, 3}}, {2, {O, 0, 1}},
       {3, {H, 0, 1, 4}},    {6, {H, 0, 2}},    {8, {H, 0, 3, 5}}};
  std::vector<std::uint32_t> path;
  for (const auto &[atom, expected] : paths) {
    tree.getAtomNodePath(*mol, atom, path);
    CHECK(path == expected);
  }
  std::vector<unsigned int> atoms;
  tree.getMatchedSubstructure(*mol, 0, atoms);
  CHECK(atoms == std::vector<unsigned int>{0, 1, 2, 8});

  // the O descent stops at the flagged node; told to read the attentions
  // instead, it carries on. Atom 0's deepest node has no value, so its walk
  // falls back to the node above.
  DASH::DASHParams deeper;
  deeper.attentionThreshold = 100.0;
  tree.getAtomNodePath(*mol, 2, path, deeper);
  CHECK(path == std::vector<std::uint32_t>{O, 0, 1, 3});
  CHECK(tree.getAtomProperty(*mol, 2, "result") == -0.65);
  CHECK(tree.getAtomProperty(*mol, 2, "result", deeper) == -0.70);
  CHECK(tree.getAtomProperty(*mol, 0, "result") == -0.45);

  // the raw values, the deviations and the three normalisations
  const std::vector<double> raw = {-0.45, -0.18, -0.65, 0.12, 0.12,
                                   0.12,  0.08,  0.08,  0.31};
  const std::vector<double> dev = {0.03, 0.02, 0.05, 0.02, 0.02,
                                   0.02, 0.02, 0.02, 0.02};
  std::vector<double> charges, rawOut, devOut;
  std::vector<unsigned int> depths;
  tree.getPartialCharges(*mol, charges, rawOut, devOut, depths);
  CHECK(rawOut == raw);
  CHECK(devOut == dev);
  CHECK(depths == std::vector<unsigned int>{4, 3, 2, 3, 3, 3, 2, 2, 3});
  double deficit = 0.0, devTotal = 0.0;
  for (std::size_t i = 0; i < raw.size(); ++i) {
    deficit -= raw[i];
    devTotal += dev[i];
  }
  for (std::size_t i = 0; i < raw.size(); ++i) {
    CHECK_THAT(charges[i], Catch::Matchers::WithinAbs(
                               raw[i] + deficit * dev[i] / devTotal, 1e-12));
  }
  DASH::ChargeOptions symmetric;
  symmetric.normalization = DASH::ChargeNormalization::SYMMETRIC;
  tree.getPartialCharges(*mol, charges, symmetric);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    CHECK_THAT(charges[i],
               Catch::Matchers::WithinAbs(raw[i] + deficit / 9.0, 1e-12));
  }

  // the node API reads the same records the descent used
  DASH::DASHTreeNode root = tree.getRoot(O);
  REQUIRE(root.getNumChildren() == 2);
  DASH::DASHTreeNode first = root.getChild(0);
  CHECK(first.getAtomFeatureIndex() == static_cast<int>(H));
  CHECK(first.getConAtom() == 0);
  CHECK(first.getConType() == 1);
  CHECK(first.getAttention() == 11.0f);
  CHECK(first.stops());
  CHECK(!root.getChild(1).stops());
  CHECK(root.getChild(1).getValue("result") == -0.99);
  CHECK(first.getChild(0).getId() == 3);
  CHECK(std::isnan(tree.getNode(M, 3).getValue("result")));
  CHECK_THROWS_AS(root.getChild(2), ValueErrorException);
  CHECK_THROWS_AS(tree.getNode(4, 0), ValueErrorException);

  // a threaded batch agrees with the single call, and a null entry is empty
  std::vector<std::vector<double>> batch;
  tree.getPartialChargesBatch({mol.get(), nullptr, mol.get()}, batch,
                              DASH::ChargeOptions(), 0);
  REQUIRE(batch.size() == 3);
  CHECK(batch[1].empty());
  tree.getPartialCharges(*mol, charges);
  CHECK(batch[0] == charges);
  CHECK(batch[2] == charges);

  // an atom outside the DASH classes, and a class the tree has no data for
  CHECK_THROWS_AS(tree.getPartialCharges(*molWithHs("C"), charges),
                  ValueErrorException);
  CHECK_THROWS_AS(tree.getPartialCharges(*molWithHs("c1ccccc1"), charges),
                  ValueErrorException);
}
