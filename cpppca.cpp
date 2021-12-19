#include "numbers.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

void read_img_list(const std::string& filename, std::vector<cv::Mat>& images)
{
	std::ifstream file(filename.c_str(), std::ifstream::in);
	if (!file) {
		CV_Error(cv::Error::StsBadArg, "invalid input filename");
	}
	std::string line;
	while (std::getline(file, line)) {
		images.push_back(cv::imread(line,0));
	}
}

auto reshape_images_to_rows(const std::vector<cv::Mat> &images) -> cv::Mat
{
	cv::Mat result(
		static_cast<int>(images.size()),
		images[0].rows*images[0].cols,
		CV_32F
	);
    for(unsigned int i = 0; i < images.size(); i++)
    {
        cv::Mat img_row = images[i].clone().reshape(0,1);
        cv::Mat row_i = result.row(i);
        img_row.convertTo(row_i, CV_32F);
    }
    return result;
}

int main()
{
	std::vector<cv::Mat> images;

	try {
		read_img_list("img_list.txt", images);
	} catch (const cv::Exception& e) {
		std::cerr << "Error opening img list: " << e.msg << std::endl;
		exit(1);
	}

	auto data = reshape_images_to_rows(images);
	auto pca = cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.95);

    cv::Mat point = pca.project(data.row(0)); // project into the eigenspace, thus the image becomes a "point"
    cv::Mat reconstruction = pca.backProject(point); // re-create the image from the "point"
    reconstruction = reconstruction.reshape(images[0].channels(), images[0].rows); // reshape from a row vector into image shape

	// init highgui window
	auto window_name = "reconstruction";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    // display until user presses q
    cv::imshow(window_name, reconstruction);
    while(cv::waitKey() != 'q') {}
	return 0;
}
