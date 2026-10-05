//
//  Copyright (C) 2026 Marc Lehner and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
//! \file AtomFeatures.h
/// \brief the DASH atom-feature classification and the packed node match key

#include <RDGeneral/export.h>

#ifndef RD_DASH_ATOMFEATURES_H
#define RD_DASH_ATOMFEATURES_H

#include <cstdint>
#include <vector>

namespace RDKit {
class Atom;
class Bond;
class ROMol;

namespace DASH {

//! \brief one atom-feature class, laid out as the container stores it
/*!
  The 5-tuple DASH classifies atoms by. Which tuples exist is a property of
  the tree, not of the code: the container carries its table, one entry per
  branch, and an atom whose tuple is not in it cannot be assigned by that tree
  at all. The published trees have 122 classes.
*/
struct RDKIT_DASHTREE_EXPORT AtomFeature {
  std::uint8_t atomicNum;
  std::uint8_t degree;
  std::int8_t formalCharge;
  std::uint8_t conjugated;  //!< is any bond of the atom conjugated
  std::uint8_t numHs;       //!< total Hs, explicit neighbours included
  std::uint8_t reserved[3];
};
static_assert(sizeof(AtomFeature) == 8, "AtomFeature is an 8-byte file record");

//! \brief classifies atoms against the feature table of one tree
/*!
  O(1): a direct-lookup table over the 14 bits the five components fit into
  (Z <= 63, degree <= 7, charge -1..2, numHs <= 3), built once when the tree is
  mapped. A table holding a tuple outside that budget is refused.
*/
class RDKIT_DASHTREE_EXPORT AtomFeatureLookup {
 public:
  AtomFeatureLookup() = default;
  //! \throws ValueErrorException if a tuple cannot be encoded or occurs twice
  AtomFeatureLookup(const AtomFeature *features, unsigned int numFeatures);

  unsigned int size() const { return d_numFeatures; }
  //! \pre index < size()
  const AtomFeature &feature(unsigned int index) const {
    return d_features[index];
  }
  //! branch index of a tuple, or -1 if the table has no such class
  int index(unsigned int atomicNum, unsigned int degree, int formalCharge,
            bool conjugated, unsigned int numHs) const;
  //! branch index of an atom in a molecule, or -1
  int index(const Atom *atom) const;

 private:
  const AtomFeature *d_features = nullptr;
  unsigned int d_numFeatures = 0;
  std::vector<std::int16_t> d_index;
};

//! \brief the DASH bond descriptor: 4 if conjugated, else the bond order, -1 if
//! unbonded
RDKIT_DASHTREE_EXPORT int dashBondType(const Bond *bond);

// ---------------------------------------------------------------------------
//  The packed match key
// ---------------------------------------------------------------------------
//  A tree node is identified relative to its parent by three small integers:
//  the feature class of the atom it adds, the position of the already-matched
//  atom it attaches to, and the bond type between them. All three fit in one
//  16-bit word, which makes a match a single integer comparison and keeps the
//  match key inside the node record the descent already has to load.
//
//  Bit 15 is not part of any field: it marks a key that cannot describe a node
//  and so can never match one. Two different situations need that, and both are
//  correct rather than approximate:
//
//   * A query throws up a component outside the stored range -- a bond order
//     RDKit knows but the trees do not (DATIVE is 17, ZERO is 21), or an
//     attachment position beyond the deepest one any node uses, which a caller
//     can reach by raising maxDepth. Aliasing those into a neighbouring field
//     would silently match the wrong node; flagging them is exact, because the
//     converter refuses to write a tree holding a key it cannot encode, so
//     "unrepresentable" really does imply "matches nothing here".
//   * A candidate that has already been consumed by the match, which must not
//     be bound a second time.
//
//  Setting bit 15 also survives the bit-or that grafts an attachment position
//  onto a half-built key, so one sentinel serves both.

const unsigned int keyAtomTypeBits = 8;  //!< bits 0-7:   atom feature class
const unsigned int keyConAtomBits = 4;  //!< bits 8-11:  attachment position + 1
const unsigned int keyConTypeBits = 3;  //!< bits 12-14: bond descriptor + 1
const unsigned int keyConAtomShift = keyAtomTypeBits;
const unsigned int keyConTypeShift = keyAtomTypeBits + keyConAtomBits;

const int maxKeyAtomType = (1 << keyAtomTypeBits) - 1;  // 255
const int maxKeyConAtom = (1 << keyConAtomBits) - 2;    // 14, stored as +1
const int maxKeyConType = (1 << keyConTypeBits) - 2;    // 6,  stored as +1

//! \brief bit 15: this key describes no node, so it matches nothing
const std::uint16_t noMatchKeyFlag = 0x8000;
//! \brief a key that can never match, and stays that way under \c withConAtom
const std::uint16_t noMatchKey = 0xFFFF;

//! packs a match key, returning \c noMatchKey if any component is out of range
inline std::uint16_t packMatchKey(int atomType, int conAtom, int conType) {
  if (atomType < 0 || atomType > maxKeyAtomType || conAtom < -1 ||
      conAtom > maxKeyConAtom || conType < -1 || conType > maxKeyConType) {
    return noMatchKey;
  }
  return static_cast<std::uint16_t>(atomType |
                                    ((conAtom + 1) << keyConAtomShift) |
                                    ((conType + 1) << keyConTypeShift));
}

//! \brief packs the part of a match key that does not depend on the match state
/*!
  The feature class and the bond descriptor are fixed for a given (atom,
  neighbour) pair; only the position of the attachment atom changes as the match
  grows. Precomputing this half per bond turns candidate generation into a
  bit-or.
*/
inline std::uint16_t packPartialMatchKey(int atomType, int conType) {
  if (atomType < 0 || atomType > maxKeyAtomType || conType < -1 ||
      conType > maxKeyConType) {
    return noMatchKey;
  }
  return static_cast<std::uint16_t>(atomType |
                                    ((conType + 1) << keyConTypeShift));
}

//! grafts an attachment position onto a partial key
inline std::uint16_t withConAtom(std::uint16_t partialKey, int conAtom) {
  if (conAtom < 0 || conAtom > maxKeyConAtom) {
    return noMatchKey;
  }
  return static_cast<std::uint16_t>(partialKey |
                                    ((conAtom + 1) << keyConAtomShift));
}

//! \name the components of a packed key
//! The inverse of packMatchKey. A component the key does not have -- the
//! attachment of a branch root, or of the heavy-atom child of a hydrogen
//! root -- comes back as -1, which is what packMatchKey was given for it.
//@{
inline int keyAtomType(std::uint16_t key) {
  return key & ((1 << keyAtomTypeBits) - 1);
}
inline int keyConAtom(std::uint16_t key) {
  return static_cast<int>((key >> keyConAtomShift) &
                          ((1 << keyConAtomBits) - 1)) -
         1;
}
inline int keyConType(std::uint16_t key) {
  return static_cast<int>((key >> keyConTypeShift) &
                          ((1 << keyConTypeBits) - 1)) -
         1;
}
//@}

}  // namespace DASH
}  // namespace RDKit

#endif
