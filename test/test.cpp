#include <patchmatch/patchmatch.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define DEBUG
#include "../src/debug.hpp"

bool drawing = false;
bool run = false;

void draw_circle(int event, int x, int y, int flags, void* userdata)
{
   cv::Mat& mask = *(cv::Mat*)userdata;
   switch(event) {
      case cv::EVENT_LBUTTONDOWN:
         drawing = true;
         break;

      case cv::EVENT_MOUSEMOVE:
         if (drawing) {
            cv::circle(mask, cv::Point{x, y}, 10, cv::Scalar{255, 255, 255}, -1);
         }
         break;

      case cv::EVENT_RBUTTONDOWN:
         run = true;
         break;

      case cv::EVENT_LBUTTONUP:
         drawing = false;
         break;
   }
}

#define SIMPLE_MODE 1
int main(int argc, char* argv[])
{
  cv::Mat input = cv::imread("data/patch-5.jpg", cv::IMREAD_COLOR);
  cv::cvtColor(input, input, cv::COLOR_BGR2BGRA);
#if SIMPLE_MODE
  cv::Mat mask = cv::imread("data/patch-5_mask.png", cv::IMREAD_GRAYSCALE);
  cv::Mat completedImage = imageComplete(input, mask, [](cv::Mat const& progress){
    showImage("Progress", progress);
  });
  cv::imwrite("patch-5_out.png", completedImage);
#else
   cv::Mat red = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
   cv::Mat mask_ = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
   cv::Mat render = cv::Mat::zeros(input.rows, input.cols, CV_8UC4);

   showImage("input", input);
   cv::setMouseCallback("input", draw_circle, &mask_);
   while(true)
   {
      red.setTo(cv::Scalar(0, 0, 255), mask_);
      cv::addWeighted(input, 1.0, red, 0.5, 0.0, render);
      showImage("input", render);
      cv::waitKey(1);
      if (run)
      {
         run = false;
         cv::imwrite("patch-5_mask.png", mask_);
         cv::Mat completedImage = imageComplete(input, mask_, [](cv::Mat const& progress){
           showImage("Progress", progress);
         });
         showImage("input", input);
      }
   }
#endif
   cv::destroyAllWindows();
   return 0;
}
