#include <iostream>
#include <Eigen/Dense>
#include "numbers.hpp"

using namespace Eigen;

#define SCALAR_TYPE f64

template<typename Scalar, u64 rows, u64 cols>
auto covariance( Matrix<Scalar, rows, cols> X ) -> Matrix<Scalar, cols, cols>
{
	auto m = X.colwise().mean();
	auto M = Matrix<Scalar, rows, 1>::Ones() * m;
	auto X_centered = X-M;
	return X_centered.transpose() * X_centered;
}

template<typename Scalar, u64 t_rows, u64 t_cols>
class PrincipleComponentAnalysis
{
	Matrix<Scalar, t_rows, t_cols> X;
	Matrix<Scalar, t_cols, t_cols> C;
	SelfAdjointEigenSolver<Matrix<Scalar, t_cols, t_cols>> solver;

public:
	PrincipleComponentAnalysis(Matrix<Scalar, t_rows, t_cols> _X)
		: X(_X),
		  C(covariance<Scalar, t_rows, t_cols>(_X)),
		  solver(C)
	{
		if (solver.info() != Success) abort();
	}

	template<u64 k>
	auto projection_matrix() -> Matrix<Scalar, t_cols, k>
	{
		// wtf?? why do i have to write with  this weird syntax??
		return solver.eigenvectors().template block<t_cols,k>(0,0);
	}

	template<u64 k>
	auto project() -> Matrix<Scalar, t_rows, k>
	{
		auto P = projection_matrix<k>();
		return X * P;
	}

	template<u64 k>
	auto reconstruct(Matrix<Scalar, t_rows, k> Z) -> Matrix<Scalar, t_rows, t_cols>
	{
		auto P = projection_matrix<k>();
		return Z * P.adjoint();
	}
};

int main()
{
	static constexpr int ROWS = 5;
	static constexpr int COLS = 6;
	static constexpr int COLS_REDUCED = 6;
	auto X = Matrix<SCALAR_TYPE, ROWS, COLS>{
		{ 2, 7, 3,16, 4, 5},
		{ 3, 5, 9, 6,13, 7},
		{ 6,12, 5, 3,22, 2},
		{11, 3, 2, 1, 7,11},
		{ 5,23,10, 4, 3, 6}
	};

	auto pca = PrincipleComponentAnalysis<SCALAR_TYPE, ROWS, COLS>(X);
	auto Z = pca.project<5>();
	std::cout << Z << std::endl;
	auto X_hat = pca.reconstruct<5>(Z);
	std::cout << X_hat << std::endl;
}
