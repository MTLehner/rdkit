//
//  Copyright (C) 2026 Marc Lehner and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
//! \file Charges.cpp
/// \brief charge normalisation and the threaded batch entry points

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <numeric>

#include <RDGeneral/ControlCHandler.h>
#include <RDGeneral/Exceptions.h>
#include <RDGeneral/RDThreads.h>
#include <GraphMol/Atom.h>
#include <GraphMol/ROMol.h>

#include "DASHTreeImpl.h"

#ifdef RDK_BUILD_THREADSAFE_SSS
#include <mutex>
#include <thread>
#endif

namespace RDKit {
namespace DASH {

using namespace detail;

namespace {

//! \brief Neumaier (compensated) summation
/*!
  The normalisation divides by these sums, so the last bits of the result depend
  on how they were accumulated. Compensated summation tracks the rounding error
  a naive loop throws away, which both tightens the result and makes it agree
  exactly with the reference python implementation -- CPython's \c sum() has
  used the same algorithm for floats since 3.12.
*/
double compensatedSum(const std::vector<double> &values) {
  double sum = 0.0;
  double compensation = 0.0;
  for (const double value : values) {
    const double running = sum + value;
    if (std::abs(sum) >= std::abs(value)) {
      compensation += (sum - running) + value;
    } else {
      compensation += (value - running) + sum;
    }
    sum = running;
  }
  return sum + compensation;
}

//! \brief spreads the difference to a molecule-wide target over the atoms
/*!
  The tree stores each atom's value independently, so the raw values do not
  quite add up to what the molecule as a whole should carry -- its formal
  charge, for partial charges. SYMMETRIC hands every atom the same share of the
  difference; STD_WEIGHTED gives more of it to the atoms the tree is least
  certain about, which is the published default for charges.
*/
void normalizeValues(double target, const std::vector<double> &rawValues,
                     const std::vector<double> &stds,
                     Normalization normalization, std::vector<double> &res) {
  const std::size_t nAtoms = rawValues.size();
  if (normalization == Normalization::NONE || nAtoms == 0) {
    res = rawValues;
    return;
  }
  res.resize(nAtoms);

  const double deficit = target - compensatedSum(rawValues);

  if (normalization == Normalization::SYMMETRIC) {
    const double share = deficit / static_cast<double>(nAtoms);
    for (std::size_t i = 0; i < nAtoms; ++i) {
      res[i] = rawValues[i] + share;
    }
    return;
  }

  const double stdTotal = compensatedSum(stds);
  for (std::size_t i = 0; i < nAtoms; ++i) {
    res[i] = rawValues[i] + deficit * (stds[i] / stdTotal);
  }
}

double formalCharge(const ROMol &mol) {
  int total = 0;
  for (const auto atom : mol.atoms()) {
    total += atom->getFormalCharge();
  }
  return total;
}

void checkOptions(const NormalizationOptions &options) {
  if (options.normalization == Normalization::STD_WEIGHTED &&
      !(options.defaultStdValue > 0.0)) {
    throw ValueErrorException(
        "defaultStdValue must be positive: it stands in for a missing "
        "deviation, and STD_WEIGHTED divides by the sum of the deviations");
  }
}

//! \brief the deviation column \p options asks for
/*!
  STD_WEIGHTED cannot do without it; the other normalisations report it when
  the file has it and fall back to \c defaultStdValue when it does not, so a
  column without a deviation column of its own can still be normalised.
*/
const PropertyColumn *stdColumnFor(const DASHTree::Impl &impl,
                                   const NormalizationOptions &options) {
  if (options.normalization == Normalization::STD_WEIGHTED ||
      impl.propertyIndex(options.stdProperty) >= 0) {
    return &impl.property(options.stdProperty);
  }
  return nullptr;
}

//! the per-molecule normalised assignment, shared by every entry point
void assignNormalized(MolMatcher &matcher, const ROMol &mol, double target,
                      const PropertyColumn &valueColumn,
                      const PropertyColumn *stdColumn,
                      const NormalizationOptions &options,
                      std::vector<double> &res, std::vector<double> &rawValues,
                      std::vector<double> &stds,
                      std::vector<unsigned int> &matchDepths) {
  const unsigned int nAtoms = mol.getNumAtoms();
  matcher.setMolecule(mol);
  rawValues.resize(nAtoms);
  stds.resize(nAtoms);
  matchDepths.resize(nAtoms);

  for (unsigned int i = 0; i < nAtoms; ++i) {
    matcher.match(i, options.params);
    rawValues[i] = matcher.pathValue(valueColumn);
    if (std::isnan(rawValues[i])) {
      // No node on this atom's path carries a value, so there is nothing to
      // normalise and a silent NaN would spread to every other atom. This is
      // the case where the reference python implementation raises too, though
      // it does so from inside pandas.
      const Atom *atom = mol.getAtomWithIdx(i);
      throw ValueErrorException(
          "the DASH tree holds no '" + options.valueProperty + "' for atom " +
          std::to_string(i) + " (" + atom->getSymbol() + ", degree " +
          std::to_string(atom->getDegree()) + ", charge " +
          std::to_string(atom->getFormalCharge()) +
          "): its atom type is in the tree but carries no data");
    }
    // a zero deviation would give the atom no share of the normalisation, and
    // a missing one (NaN) would poison the whole molecule
    const double deviation =
        stdColumn ? matcher.pathValue(*stdColumn) : options.defaultStdValue;
    stds[i] = (deviation > 0.0) ? deviation : options.defaultStdValue;
    matchDepths[i] = static_cast<unsigned int>(matcher.path().size());
  }
  normalizeValues(target, rawValues, stds, options.normalization, res);
}

// ---------------------------------------------------------------------------
//  Work is handed out one molecule at a time through an atomic counter rather
//  than sliced up in advance: real batches vary in molecule size by an order of
//  magnitude, and a fixed slicing leaves threads idle at the end of the run.
//  Each thread keeps one matcher for the whole batch, so the per-molecule
//  tables are the only thing reallocated, and writes straight into its own slot
//  of the result, so nothing has to be merged afterwards.
// ---------------------------------------------------------------------------

//! \brief runs \c work(threadIdx, i) over [0, \p n) on \p nThreads threads
/*!
  Stops early if a Ctrl-C arrives, which the python wrapper turns into a
  KeyboardInterrupt once it has the GIL back. Nothing else looks at the
  handler's flag, so a C++ caller that has not installed the handler simply
  never sees it set.
*/
template <typename Work>
void forEachIndex(std::size_t n, unsigned int nThreads, Work work) {
  if (nThreads <= 1 || n <= 1) {
    for (std::size_t i = 0; i < n; ++i) {
      if (ControlCHandler::getGotSignal()) {
        return;
      }
      work(0u, i);
    }
    return;
  }
#ifdef RDK_BUILD_THREADSAFE_SSS
  std::atomic<std::size_t> next(0);
  std::vector<std::thread> threads;
  threads.reserve(nThreads);
  std::exception_ptr failure;
  std::mutex failureMutex;
  for (unsigned int t = 0; t < nThreads; ++t) {
    threads.emplace_back([&, t]() {
      try {
        for (;;) {
          const std::size_t i = next.fetch_add(1);
          if (i >= n || ControlCHandler::getGotSignal()) {
            break;
          }
          work(t, i);
        }
      } catch (...) {
        const std::lock_guard<std::mutex> lock(failureMutex);
        if (!failure) {
          failure = std::current_exception();
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  if (failure) {
    std::rethrow_exception(failure);
  }
#else
  for (std::size_t i = 0; i < n; ++i) {
    work(0u, i);
  }
#endif
}

//! how many threads to actually spawn: never more than there is work for
unsigned int resolveThreadCount(int numThreads, std::size_t nWork) {
  const unsigned int wanted = getNumThreadsToUse(numThreads);
  return static_cast<unsigned int>(
      std::min<std::size_t>(wanted, std::max<std::size_t>(nWork, 1)));
}

//! one reusable matcher per thread
std::vector<std::unique_ptr<MolMatcher>> makeMatchers(
    const DASHTree::Impl &tree, unsigned int nThreads) {
  std::vector<std::unique_ptr<MolMatcher>> matchers(nThreads);
  for (auto &matcher : matchers) {
    matcher.reset(new MolMatcher(tree));
  }
  return matchers;
}

}  // namespace

void DASHTree::getNormalizedMolProperty(
    const ROMol &mol, double target, std::vector<double> &res,
    std::vector<double> &rawValues, std::vector<double> &stds,
    std::vector<unsigned int> &matchDepths,
    const NormalizationOptions &options) const {
  checkOptions(options);
  const PropertyColumn &valueColumn = d_impl->property(options.valueProperty);
  const PropertyColumn *stdColumn = stdColumnFor(*d_impl, options);
  MolMatcher matcher(*d_impl);
  assignNormalized(matcher, mol, target, valueColumn, stdColumn, options, res,
                   rawValues, stds, matchDepths);
}

void DASHTree::getNormalizedMolProperty(
    const ROMol &mol, double target, std::vector<double> &res,
    const NormalizationOptions &options) const {
  std::vector<double> rawValues, stds;
  std::vector<unsigned int> matchDepths;
  getNormalizedMolProperty(mol, target, res, rawValues, stds, matchDepths,
                           options);
}

void DASHTree::getPartialCharges(const ROMol &mol, std::vector<double> &res,
                                 std::vector<double> &rawValues,
                                 std::vector<double> &stds,
                                 std::vector<unsigned int> &matchDepths,
                                 const ChargeOptions &options) const {
  getNormalizedMolProperty(mol, formalCharge(mol), res, rawValues, stds,
                           matchDepths, options);
}

void DASHTree::getPartialCharges(const ROMol &mol, std::vector<double> &res,
                                 const ChargeOptions &options) const {
  getNormalizedMolProperty(mol, formalCharge(mol), res, options);
}

void DASHTree::getMolPropertyBatch(const std::vector<const ROMol *> &mols,
                                   const std::string &property,
                                   std::vector<std::vector<double>> &res,
                                   const DASHParams &params,
                                   int numThreads) const {
  const PropertyColumn &column = d_impl->property(property);
  res.clear();
  res.resize(mols.size());

  const unsigned int nThreads = resolveThreadCount(numThreads, mols.size());
  auto matchers = makeMatchers(*d_impl, nThreads);
  forEachIndex(mols.size(), nThreads, [&](unsigned int t, std::size_t i) {
    if (!mols[i]) {
      return;
    }
    MolMatcher &matcher = *matchers[t];
    const ROMol &mol = *mols[i];
    matcher.setMolecule(mol);
    res[i].resize(mol.getNumAtoms());
    for (unsigned int a = 0; a < mol.getNumAtoms(); ++a) {
      matcher.match(a, params);
      res[i][a] = matcher.pathValue(column);
    }
  });
}

void DASHTree::getNormalizedMolPropertyBatch(
    const std::vector<const ROMol *> &mols, const std::vector<double> &targets,
    std::vector<std::vector<double>> &res, const NormalizationOptions &options,
    int numThreads) const {
  if (targets.size() != mols.size()) {
    throw ValueErrorException(
        "one target per molecule: " + std::to_string(targets.size()) +
        " targets for " + std::to_string(mols.size()) + " molecules");
  }
  checkOptions(options);
  const PropertyColumn &valueColumn = d_impl->property(options.valueProperty);
  const PropertyColumn *stdColumn = stdColumnFor(*d_impl, options);
  res.clear();
  res.resize(mols.size());

  const unsigned int nThreads = resolveThreadCount(numThreads, mols.size());
  auto matchers = makeMatchers(*d_impl, nThreads);
  std::vector<std::vector<double>> rawScratch(nThreads), stdScratch(nThreads);
  std::vector<std::vector<unsigned int>> depthScratch(nThreads);
  forEachIndex(mols.size(), nThreads, [&](unsigned int t, std::size_t i) {
    if (!mols[i]) {
      return;
    }
    assignNormalized(*matchers[t], *mols[i], targets[i], valueColumn, stdColumn,
                     options, res[i], rawScratch[t], stdScratch[t],
                     depthScratch[t]);
  });
}

void DASHTree::getPartialChargesBatch(const std::vector<const ROMol *> &mols,
                                      std::vector<std::vector<double>> &res,
                                      const ChargeOptions &options,
                                      int numThreads) const {
  std::vector<double> targets(mols.size(), 0.0);
  for (std::size_t i = 0; i < mols.size(); ++i) {
    if (mols[i]) {
      targets[i] = formalCharge(*mols[i]);
    }
  }
  getNormalizedMolPropertyBatch(mols, targets, res, options, numThreads);
}

}  // namespace DASH
}  // namespace RDKit
