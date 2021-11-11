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
	auto out = covariance<SCALAR_TYPE,ROWS,COLS>(X);
	std::cout << out << std::endl;
	auto eigensolver
		= SelfAdjointEigenSolver<Matrix<SCALAR_TYPE,COLS,COLS>>(out);
	if (eigensolver.info() != Success) abort();
	std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
	std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
			<< "corresponding to these eigenvalues:\n"
			<< eigensolver.eigenvectors() << std::endl;
	auto P = eigensolver.eigenvectors().block<COLS,COLS_REDUCED>(0,0);
	auto Z = X*P;
	auto X_hat = Z*P.adjoint();
	std::cout << X_hat << std::endl;

}
