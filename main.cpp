#include <iostream>
#include <Eigen/Dense>
#include "numbers.hpp"

using Eigen::MatrixXd;
using Eigen::Matrix;

template<typename Scalar, u64 rows, u64 cols>
auto mean_cols( Matrix<Scalar, rows, cols> x ) -> Matrix<Scalar, 1, cols>
{

}

template<typename Scalar, u64 rows, u64 cols>
auto covariance( Matrix<Scalar, rows, cols> x ) -> Matrix<Scalar, rows, cols>
{

}

int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
