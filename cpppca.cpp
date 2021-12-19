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
		images.push_back(cv::imread(line, cv::IMREAD_COLOR));
	}
}

auto reshape_img_to_row(const cv::Mat& img) -> cv::Mat
{
	cv::Mat row_channels[3];
	cv::split(img.clone(), row_channels);
	auto output = cv::Mat();
	output.push_back(row_channels[0]);
	output.push_back(row_channels[1]);
	output.push_back(row_channels[2]);
	output = output.reshape(0,1);
	return output;
}

auto reshape_row_to_img(const cv::Mat& row) -> cv::Mat
{
	return cv::Mat();
}

auto reshape_images_to_rows(const std::vector<cv::Mat> &images) -> cv::Mat
{
	auto img_size = 3*images[0].rows*images[0].cols;
	auto result = cv::Mat(
		static_cast<int>(images.size()),
		img_size,
		CV_32F
	);
	for (u64 i = 0; i < images.size(); ++i) {
		auto row_i = result.row(i);
		reshape_img_to_row(images[i]).convertTo(row_i, CV_32F);
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
	auto pca = cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1.0);

    cv::Mat point = pca.project(data.row(0));
    cv::Mat reconstruction = pca.backProject(point);
    reconstruction = reconstruction.reshape(0, images[0].rows*3);
	cv::Mat final_img;
    cv::normalize(reconstruction, final_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	// init highgui window
	auto window_name = "reconstruction";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    // display until user presses q
    cv::imshow(window_name, final_img);
    while(cv::waitKey() != 'q') {}
	return 0;
}
