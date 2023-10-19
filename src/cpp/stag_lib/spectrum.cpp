//
// This file is provided as part of the STAG library and released under the MIT
// license.
//
#include "spectrum.h"

#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <cstdlib>


stag::EigenSystem stag::compute_eigensystem(
    const SprsMat* mat, stag_int num, Spectra::SortRule sort) {
  stag::SprsMatOp op(*mat);

  // Construct eigen solver object, requesting the smallest k eigenvalues
  Spectra::SymEigsSolver<SprsMatOp> eigs(
      op, num, fmin(2*num, mat->rows()));

  // Initialize and compute
  eigs.init();
  stag_int converged = eigs.compute(sort);
  assert(eigs.info() == Spectra::CompInfo::Successful);

  // Retrieve results
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
  eigenvalues = eigs.eigenvalues();
  eigenvectors = eigs.eigenvectors();

  return {eigenvalues, eigenvectors};
}

stag::EigenSystem stag::compute_eigensystem(const SprsMat *mat, stag_int num) {
  return stag::compute_eigensystem(mat, num, Spectra::SortRule::SmallestMagn);
}

Eigen::VectorXd stag::compute_eigenvalues(const SprsMat *mat, stag_int num,
                                          Spectra::SortRule sort) {
  return get<0>(stag::compute_eigensystem(mat, num, sort));
}

Eigen::VectorXd stag::compute_eigenvalues(const SprsMat *mat, stag_int num) {
  return stag::compute_eigenvalues(mat, num, Spectra::SortRule::SmallestMagn);
}

Eigen::MatrixXd stag::compute_eigenvectors(const SprsMat *mat, stag_int num,
                                           Spectra::SortRule sort) {
  return get<1>(stag::compute_eigensystem(mat, num, sort));
}

Eigen::MatrixXd stag::compute_eigenvectors(const SprsMat *mat, stag_int num) {
  return stag::compute_eigenvectors(mat, num, Spectra::SortRule::SmallestMagn);
}
