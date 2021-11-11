#include <iostream>
#include <Eigen/Dense>
#include "numbers.hpp"

using Eigen::MatrixXd;
using Eigen::Matrix;

#define SCALAR_TYPE i32

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
	auto b = Matrix<SCALAR_TYPE, 5, 4>{
		{2,3,4,5},
		{3,5,6,7},
		{6,5,3,2},
		{3,2,1,7},
		{5,4,3,6}
	};
	auto out = covariance<SCALAR_TYPE,5,4>(b);
	std::cout << out << std::endl;
}
