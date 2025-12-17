#ifndef UTILS_H
#define UTILS_H

#include "common.h"
#include "io.h"
#include "osqp.h"

namespace utils {

/**
 * @brief Convert an Eigen::SparseMatrix to raw CSC arrays compatible with OSQP.
 *
 * Takes an Eigen::SparseMatrix in column-compressed form and produces three
 * heap-allocated arrays representing the CSC structure used by OSQP.
 * - x: non-zero values
 * - i: row indices
 * - p: column pointers
 * If @p upperTriangular is true, only the upper triangular part of the matrix
 * is extracted before conversion. This is useful when passing a symmetric
 * Hessian to OSQP, which expects only one triangle of the matrix.
 *
 * @param inputMatrix   Source Eigen sparse matrix.
 * @param upperTriangular If true, only the upper triangle is used.
 * @return Tuple (x, i, p) of newly allocated arrays; the caller is responsible
 *         for freeing them with @c delete[].
 */
inline std::tuple<OSQPFloat*, OSQPInt*, OSQPInt*> convert_eigen_to_osqpCsc(const SparseMatrix<double>& inputMatrix, bool upperTriangular = false) {
    SparseMatrix<double> matrix;
    if (upperTriangular) {
        matrix = inputMatrix.triangularView<Upper>();
    } else {
        matrix = inputMatrix;
    }
    matrix.makeCompressed(); // Ensure the matrix is in compressed format

    OSQPInt n = static_cast<OSQPInt>(matrix.cols());
    OSQPInt nnz = static_cast<OSQPInt>(matrix.nonZeros());

    OSQPFloat* x = new OSQPFloat[nnz];  // Internal type of OSQPFloat is double
    OSQPInt* i = new OSQPInt[nnz]; // Internal type of OSQPInt is long long int
    OSQPInt* p = new OSQPInt[n + 1];
    
    // Copy data
    for (OSQPInt k = 0; k < nnz; ++k) {
        x[k] = static_cast<OSQPFloat>(matrix.valuePtr()[k]);
        i[k] = static_cast<OSQPInt>(matrix.innerIndexPtr()[k]);
    }
    for (OSQPInt k = 0; k < n + 1; ++k) {
        p[k] = static_cast<OSQPInt>(matrix.outerIndexPtr()[k]);
    }
    return std::make_tuple(x, i, p);
}

/**
 * @brief Convert an OSQP CSC matrix (OSQPCscMatrix) back to Eigen::SparseMatrix.
 *
 * Constructs a new Eigen::SparseMatrix<double> from an OSQPCscMatrix in
 * column-compressed format. This is primarily useful for inspecting OSQP’s
 * internal matrices, debugging, or post-processing.
 *
 * @param input_matrix Pointer to an OSQPCscMatrix (may be nullptr).
 * @return An Eigen::SparseMatrix<double> with the same dimensions and entries
 *         as @p input_matrix; if @p input_matrix is nullptr, an empty matrix
 *         is returned.
 */
inline SparseMatrix<double> convert_osqpCsc_to_eigen(const OSQPCscMatrix* input_matrix) {
    
    if (!input_matrix) return SparseMatrix<double>(); // Check if input_matrix is not nullptr

    const OSQPInt* p = input_matrix->p;
    const OSQPInt* i = input_matrix->i;
    const OSQPFloat* x = input_matrix->x;

    OSQPInt m = input_matrix->m;
    OSQPInt n = input_matrix->n;
    OSQPInt nnz = p[n];

    std::vector<Triplet<double>> triplets;
    triplets.reserve(nnz);

    OSQPInt col = 0;
    for (OSQPInt idx = 0; idx < nnz; ++idx) {
        while (idx >= p[col + 1]) ++col;
        triplets.emplace_back(i[idx], col, x[idx]);
    }

    SparseMatrix<double> output_matrix(m, n);
    output_matrix.setFromTriplets(triplets.begin(), triplets.end());
    return output_matrix;
}
/**
 * @struct OsqpMatrixDeleter
 * @brief Custom deleter for OSQPCscMatrix pointers.
 *
 * This functor wraps @c OSQPCscMatrix_free and can be used with
 * @c std::unique_ptr to automatically release OSQP CSC matrices
 * when they go out of scope.
 */
struct OsqpMatrixDeleter {
    void operator()(OSQPCscMatrix* mat) const { if (mat) OSQPCscMatrix_free(mat); }
};
using OSQPMatrixPtr = std::unique_ptr<OSQPCscMatrix, OsqpMatrixDeleter>;

// Conversion from std::vector<double> to Eigen::VectorXi using rounding
inline VectorXi convert_stdVecdouble_to_vectorXi(const std::vector<double>& v) {
    VectorXi out(v.size());
    for (unsigned int i = 0; i < v.size(); ++i)
        out(i) = static_cast<int>(std::round(v[i]));
    return out;
}

// Conversion from std::vector<double> to Eigen::VectorXd
inline VectorXd convert_stdVecdouble_to_vectorXd(const std::vector<double>& v) {
    return Map<const VectorXd>(v.data(), v.size());
}

// Conversion from std::vector<int> to Eigen::VectorXi
inline VectorXi convert_stdVecint_to_vectorXi(const std::vector<int>& v) {
    return Map<const VectorXi>(v.data(), v.size());
}

// Conversion from std::vector<double> to std::vector<int> using rounding
inline std::vector<int> convert_stdVecdouble_to_stdVecint(const std::vector<double>& v) {
    std::vector<int> out(v.size());
    for (unsigned int i = 0; i < v.size(); ++i)
        out[i] = static_cast<int>(std::round(v[i]));
    return out;
}

/**
 * @brief Extract surface faces from a volumetric mesh (tetrahedral or hexahedral).
 * 
 * This function extracts the surface faces of a volumetric mesh defined by
 * its undeformed vertex positions and element connectivity. It supports both 
 * tetrahedral (elementDim=4) and hexahedral (elementDim=8) meshes.
 * Usage:
 * auto [surfaceGroups, faceEleGroups] = utils::extract_surfaces<vertexDim, elementDim>(V, E, verboseFlag);
 */
template<int vertexDim, int elementDim>
auto extract_surfaces(
    Eigen::Matrix<double, -1, vertexDim>& undeformedVertices,
    Eigen::Matrix<int,    -1, elementDim>& eleIdx,
    bool verboseFlag = false) {
    static_assert(vertexDim == 3, "extract_surfaces: only implemented for vertexDim == 3.");
    static_assert(elementDim == 4 || elementDim == 8,
                  "extract_surfaces: only implemented for elementDim == 4 (tet) or 8 (hex).");

    if constexpr (elementDim == 4) {
        return IO::extract_tet_surfaces(undeformedVertices, eleIdx, verboseFlag);
    } else if constexpr (elementDim == 8){ 
        return IO::extract_hex_surfaces(undeformedVertices, eleIdx, verboseFlag);
    }
}

/**
 * @brief Estimate the condition number of a sparse Hessian.
 *
 * Uses power-iteration-based estimates of the largest and smallest eigenvalues
 * (assuming a positive semi-definite Hessian) to approximate:
 *   cond(H) ≈ λ_max / λ_min.
 */
inline double compute_condition_number (const SparseMatrix<double>& hessian) {
    // SVD
    // JacobiSVD<MatrixXd> svd(hessian);
    // std::cout << "Hessian Condition Number: " << svd.singularValues()(0)/svd.singularValues()(svd.singularValues().size()-1) << std::endl;

    // Psuedo inverse
    // SimplicialLDLT<SparseMatrix<double>> solver;
    // solver.compute(hessian);
    // SparseMatrix<double> I(hessian.rows(), hessian.cols());
    // I.setIdentity();
    // SparseMatrix<double> hessian_inv = solver.solve(I);

    // std::cout << "Hessian Condition Number: " << hessian.norm() * hessian_inv.norm() << std::endl;

    // Power Iterations
    int MAX_POWER_ITER = 1000;
    VectorXd v_max = VectorXd::Random(hessian.cols());
    VectorXd v = v_max;
    for (int i = 0; i < MAX_POWER_ITER; i++) {
        v_max = hessian*v_max;
        v_max.normalize();
        // Check convergence
        if ((v - v_max).norm() < 1e-3) {
            // std::cout << "Power Iterations e_max Converged in " << i << " iterations" << std::endl;
            break;
        }
        v = v_max;
    }
    double e_max = v_max.transpose()*hessian*v_max;

    // Spectral shift for Hessian, since all eigenvalues should be positive PSD
    VectorXd v_min = VectorXd::Random(hessian.cols());
    auto shiftedHessian = -(hessian - e_max*MatrixXd::Identity(hessian.rows(), hessian.cols()));
    for (int i = 0; i < MAX_POWER_ITER; i++) {
        v_min = shiftedHessian*v_min;
        v_min.normalize();
        // Check convergence
        if ((v - v_min).norm() < 1e-3) {
            // std::cout << "Power Iterations e_min Converged in " << i << " iterations" << std::endl;
            break;
        }
        v = v_min;
    }
    double e_min = -v_min.transpose()*shiftedHessian*v_min + e_max;
    return e_max/e_min;
}

}
#endif  // UTILS_H
