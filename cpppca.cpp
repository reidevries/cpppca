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
	// split individual channels from img
	cv::Mat row_channels[3];
	cv::split(img.clone(), row_channels);

	// append each channel into the same grayscale Mat
	// then put them all on the same row
	auto output = cv::Mat();
	output.push_back(row_channels[0]);
	output.push_back(row_channels[1]);
	output.push_back(row_channels[2]);
	output = output.reshape(0,1);
	return output;
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

auto reshape_row_to_img(const cv::Mat& row, const u64 img_rows) -> cv::Mat
{
	// split row up into three equal parts, and place them as rows of a matrix
	auto input = row.reshape(0,3);
	cv::Mat channels[3];
	for (u8 i = 0; i < 3; ++i) {
		input.row(i).reshape(0, img_rows).convertTo(channels[i], CV_32F);
	}
	cv::Mat output;
	cv::merge(channels, 3, output);
	return output;
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
	cv::Mat means;
	cv::Mat eigenvecs;
	cv::PCACompute(data, means, eigenvecs, static_cast<int>(images.size()));

	cv::Mat projection;
    cv::PCAProject(data.row(1), means, eigenvecs, projection);
	cv::Mat reconstruction;
    cv::PCABackProject(projection, means, eigenvecs, reconstruction);
	cv::Mat dest = reshape_row_to_img(reconstruction, images[0].rows);
	cv::Mat final_img;
    cv::normalize(dest, final_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	// init highgui window
	auto window_name = "reconstruction";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    // display until user presses q
    cv::imshow(window_name, final_img);
    while(cv::waitKey() != 'q') {}
	return 0;
}
