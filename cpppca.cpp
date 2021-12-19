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
	cv::Mat merged;
	cv::merge(channels, 3, merged);
	cv::Mat output;
    cv::normalize(merged, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	return output;
}

class PCA {
	cv::Mat means;
	cv::Mat eigenvecs;
public:
	PCA() {}

	PCA(const cv::Mat& data)
	{
		train(data);
	}

	void train(const cv::Mat& data)
	{
		cv::PCACompute(data, means, eigenvecs, static_cast<int>(data.rows));
	}

	auto project(const cv::Mat& data_row) const -> cv::Mat
	{
		cv::Mat output;
		cv::PCAProject(data_row, means, eigenvecs, output);
		return output;
	}

	auto project(const cv::Mat& data_row, const int num_components) const
		-> cv::Mat
	{
		auto z = eigenvecs.rowRange(0, num_components);
		cv::Mat output;
		cv::PCAProject(data_row, means, z, output);
		return output;
	}

	auto reconstruct(const cv::Mat& input) const -> cv::Mat
	{
		cv::Mat output;
		cv::PCABackProject(input, means, eigenvecs, output);
		return output;
	}

	auto reconstruct(const cv::Mat& input, const int num_components) const
		-> cv::Mat
	{

		auto z = eigenvecs.rowRange(0, num_components);
		cv::Mat output;
		cv::PCABackProject(input, means, z, output);
		return output;
	}

	auto get_component(const int index) -> cv::Mat
	{
		return eigenvecs.row(index);
	}
};


struct params
{
    cv::Mat data;
    int img_xsize;
    PCA pca;
    std::string window_name;
};

void trackbar_callback(int pos, void* ptr)
{
    struct params *p = (struct params *)ptr;
	auto final_img = reshape_row_to_img(p->pca.get_component(pos), p->img_xsize);
    cv::imshow(p->window_name, final_img);
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
	auto pca = PCA(data);

	auto projection = pca.project(data.row(1), 4);
	std::cout << "projection: " << projection.rows << "x" << projection.cols << std::endl;
	auto reconstruction = pca.reconstruct(projection, 4);
	std::cout << "reconstruction: " << reconstruction.rows << "x" << reconstruction.cols << std::endl;
	auto final_img = reshape_row_to_img(pca.get_component(0), images[0].rows);

	// init highgui window
	auto window_name = "reconstruction";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	// params struct to pass to the trackbar handler
    params p;
    p.data = data;
    p.img_xsize = images[0].rows;
    p.pca = pca;
    p.window_name = window_name;
    // create the tracbar
    cv::createTrackbar("Retained Variance (%)", window_name, NULL, images.size()-1, trackbar_callback, (void*)&p);
    // display until user presses q
    cv::imshow(window_name, final_img);
    while(cv::waitKey() != 'q') {}
	return 0;
}
