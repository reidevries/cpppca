#include <iostream>
#include <Eigen/Dense>
#include "numbers.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
using namespace Eigen;

auto covariance( MatrixXd X ) -> MatrixXd
{
	auto m = X.colwise().mean();
	auto M = MatrixXd::Ones(X.rows(), 1) * m;
	auto X_centered = X-M;
	return X_centered.transpose() * X_centered;
}

class PrincipleComponentAnalysis
{
	MatrixXd X;
	MatrixXd C;
	SelfAdjointEigenSolver<MatrixXd> solver;

public:
	PrincipleComponentAnalysis(MatrixXd _X)
		: X(_X),
		  C(covariance(_X)),
		  solver(C)
	{
		if (solver.info() != Success) abort();
	}

	auto projection_matrix(u64 k) -> MatrixXd
	{
		return solver.eigenvectors().leftCols(k);
	}

	auto project(u64 k) -> MatrixXd
	{
		auto P = projection_matrix(k);
		return X * P;
	}

	auto reconstruct(MatrixXd Z, u64 k) -> MatrixXd
	{
		auto P = projection_matrix(k);
		return Z * P.adjoint();
	}
};

int main()
{
	static constexpr int ROWS = 5;
	static constexpr int COLS = 6;
	static constexpr int COLS_REDUCED = 6;

	auto image_path = cv::samples::findFile("starry_night.jpg");
    auto img = cv::imread(image_path, cv::IMREAD_COLOR);
	auto img_mat = MatrixXd();
	cv::cv2eigen(img, img_mat);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    cv::imshow("Display window", img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        cv::imwrite("starry_night.png", img);
    }
    return 0;
}
