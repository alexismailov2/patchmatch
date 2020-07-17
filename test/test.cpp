#include <patchmatch/patchmatch.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define DEBUG
#include "../src/debug.hpp"
#include "../src/new/nnf.h"
#include "../src/new/inpaint.h"

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
            cv::circle(mask, cv::Point{x, y}, 20, cv::Scalar{255, 255, 255}, -1);
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

#define SIMPLE_MODE 0
int main(int argc, char* argv[])
{
  cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::cvtColor(input, input, cv::COLOR_BGR2BGRA);
#if SIMPLE_MODE
  cv::Mat mask = cv::imread("data/IMG_9464.png", cv::IMREAD_COLOR);
  cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
  cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
  showImage("Mask", mask);
  auto config = pm::PatchMatch::Config();
  config.Original(input)
        .Mask(mask)
        .ImageCompletionSteps(10)
        .PatchMatchingSteps(20)
        .ProgressCb([](cv::Mat const& completedImage, pm::PatchMatch::Progress const& progress) -> bool {
           showImage("Progress", completedImage);
           std::cout << "Percent progress: " << progress.getPercent() << "%" << std::endl;
           return true;
        });
  cv::Mat completedImage = pm::PatchMatch(config).ImageComplete();
  cv::imwrite("patch-5_out.png", completedImage);
#else
   cv::Mat red = cv::Mat::zeros(input.rows, input.cols, CV_8UC4);
   cv::Mat mask = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
   cv::Mat render = cv::Mat::zeros(input.rows, input.cols, CV_8UC4);

   showImage("input", input);
   cv::setMouseCallback("input", draw_circle, &mask);
   while(true)
   {
      red.setTo(cv::Scalar(0, 0, 255), mask);
      cv::addWeighted(input, 1.0, red, 0.5, 0.0, render);
      showImage("input", render);
      cv::waitKey(1);
      if (run)
      {
         run = false;
         cv::imwrite("patch-5_mask.png", mask);
#if 1
         auto config = pm::PatchMatch::Config();
         config.Original(input)
               .Mask(mask)
               .ImageCompletionSteps(10)
               .PatchMatchingSteps(20)
               .ProgressCb([&](cv::Mat const& completedImage, pm::PatchMatch::Progress const& progress) -> bool {
                 cv::Mat origSize;
                 cv::resize(completedImage, origSize, cv::Size{input.cols, input.rows});
                 cv::rectangle(origSize, cv::Rect(0, 0, origSize.cols * progress.getPercent() / 100.0f, 50), cv::Scalar{0, 255, 0}, -1);
                 showImage("Progress", origSize);
                 std::cout << "Percent progress: " << progress.getPercent() << "%" << std::endl;
                 return true;
               });
         cv::Mat completedImage = pm::PatchMatch(config).ImageComplete();
         showImage("Progress", completedImage);
#else
//        auto metric = PatchSSDDistanceMetric(3);
//        auto result = Inpainting(input, mask, &metric).run(true, true);
//        showImage("Progress", result);
#endif
      }
   }
#endif
   cv::destroyAllWindows();
   return 0;
}

