#ifndef __PCA_H_
#define __PCA_H_

#include "numbers.hpp"
#include <Eigen/Dense>

template<typename Scalar, u64 rows, u64 cols>
auto covariance( Eigen::Matrix<Scalar, rows, cols> X ) -> Eigen::Matrix<Scalar, cols, cols>
{
	auto m = X.colwise().mean();
	auto M = Eigen::Matrix<Scalar, rows, 1>::Ones() * m;
	auto X_centered = X-M;
	return X_centered.transpose() * X_centered;
}

template<typename Scalar, u64 t_rows, u64 t_cols>
class PrincipleComponentAnalysis
{
	Eigen::Matrix<Scalar, t_rows, t_cols> X;
	Eigen::Matrix<Scalar, t_cols, t_cols> C;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, t_cols, t_cols>> solver;

public:
	PrincipleComponentAnalysis(Eigen::Matrix<Scalar, t_rows, t_cols> _X)
		: X(_X),
		  C(covariance<Scalar, t_rows, t_cols>(_X)),
		  solver(C)
	{
		if (solver.info() != Eigen::Success) abort();
	}

	template<u64 k>
	auto projection_matrix() -> Eigen::Matrix<Scalar, t_cols, k>
	{
		// wtf?? why do i have to write with  this weird syntax??
		return solver.eigenvectors().template block<t_cols,k>(0,0);
	}

	template<u64 k>
	auto project() -> Eigen::Matrix<Scalar, t_rows, k>
	{
		auto P = projection_matrix<k>();
		return X * P;
	}

	template<u64 k>
	auto reconstruct(Eigen::Matrix<Scalar, t_rows, k> Z) -> Eigen::Matrix<Scalar, t_rows, t_cols>
	{
		auto P = projection_matrix<k>();
		return Z * P.adjoint();
	}
};

#endif // __PCA_H_
