#include <patchmatch/patchmatch.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define DEBUG
#include "../src/debug.hpp"
#include "../src/new/nnf.h"
#include "../src/new/inpaint.h"

#include <experimental/filesystem>

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
            cv::circle(mask, cv::Point{x, y}, 100, cv::Scalar{255, 255, 255}, -1);
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

template<typename T>
void LabelConcreteImage(T file, std::tuple<std::string, std::string, std::string, std::string> const& datasetFolderPath, std::function<bool(cv::Mat const&)>&& skipPredicat = [](cv::Mat const&) -> bool { return false; })
{
   cv::Mat input = cv::imread(file.path().string());
   if (skipPredicat(input))
   {
      return;
   }
   cv::cvtColor(input, input, cv::COLOR_BGR2BGRA);
   cv::Mat red = cv::Mat::zeros(input.rows, input.cols, CV_8UC4);
   cv::Mat mask = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
   cv::Mat render = cv::Mat::zeros(input.rows, input.cols, CV_8UC4);
   showImage("input", input);
   cv::setMouseCallback("input", draw_circle, &mask);
   while(true)
   {
      red.setTo(cv::Scalar(0, 0, 255), mask);
      cv::addWeighted(input, 1.0, red, 0.8, 0.0, render);
      showImage("input", render);
      cv::waitKey(1);
      if (run)
      {
         run = false;
         auto filename = file.path().filename().string();
         filename = filename.substr(0, filename.size() - 4);
         cv::imwrite(std::get<1>(datasetFolderPath) + "/" + filename + ".png", mask);
         break;
      }
   }
}

void IterateThroughAllImagesInFolder(std::vector<std::tuple<std::string, std::string, std::string, std::string>> const& datasetFolderPathes,
                                     std::function<void(std::tuple<std::string, std::string, std::string, std::string> const&, std::experimental::filesystem::directory_entry)>&& callback)
{
   for (auto datasetFolderPath : datasetFolderPathes)
   {
      if (!std::get<1>(datasetFolderPath).empty())
      {
         std::experimental::filesystem::create_directories(std::get<1>(datasetFolderPath));
      }
      if (!std::get<2>(datasetFolderPath).empty())
      {
         std::experimental::filesystem::create_directories(std::get<2>(datasetFolderPath));
      }
      for (auto file : std::experimental::filesystem::directory_iterator(std::get<0>(datasetFolderPath)))
      {
         callback(datasetFolderPath, file);
      }
   }
}

void LabelingAllImagesInFolder(std::vector<std::tuple<std::string, std::string, std::string, std::string>> const& datasetFolderPathes,
                               std::function<bool(cv::Mat const&)>&& skipPredicat = [](cv::Mat const&) -> bool { return false; })
{
#if 0
   for (auto datasetFolderPath : datasetFolderPathes)
   {
      std::experimental::filesystem::create_directories(datasetFolderPath.second);
      for (auto file : std::experimental::filesystem::directory_iterator(datasetFolderPath.first))
      {
         LabelConcreteImage(file, datasetFolderPath, std::move(skipPredicat));
      }
   }
#else
   IterateThroughAllImagesInFolder(datasetFolderPathes,
      [skipPredicat = std::move(skipPredicat)](std::tuple<std::string, std::string, std::string, std::string> const& datasetFolderPath,
                                               std::experimental::filesystem::directory_entry file) {
     LabelConcreteImage(file, datasetFolderPath/*, std::move(skipPredicat)*/);
   });
#endif
}

void InpaintingAllImagesInFolder(std::vector<std::tuple<std::string, std::string, std::string, std::string>> const& datasetFolderPathes)
{
   IterateThroughAllImagesInFolder(datasetFolderPathes,
      [](std::tuple<std::string, std::string, std::string, std::string> const& datasetFolderPath,
         std::experimental::filesystem::directory_entry file) {
     cv::Mat input = cv::imread(file.path().string());
//     if (skipPredicat(input))
//     {
//        return;
//     }
     cv::cvtColor(input, input, cv::COLOR_BGR2BGRA);
     auto filename = file.path().filename().string();
     cv::Mat mask = cv::imread(std::get<1>(datasetFolderPath) + "/" + filename, cv::IMREAD_COLOR);
     cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
     cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
     showImage("Mask", mask);

     cv::Mat globalMask = cv::imread(std::get<3>(datasetFolderPath) + "/" + filename, cv::IMREAD_COLOR);
     cv::cvtColor(globalMask, globalMask, cv::COLOR_BGR2GRAY);
     cv::threshold(globalMask, globalMask, 1, 255, cv::THRESH_BINARY);
     showImage("Global Mask", globalMask);

     auto config = pm::PatchMatch::Config();
     config.Original(input)
           .Mask(mask)
           .GlobalMask(globalMask)
           .ImageCompletionSteps(10)
           .PatchMatchingSteps(20)
           .ProgressCb([&](cv::Mat const& completedImage, pm::PatchMatch::Progress const& progress) -> bool {
               cv::Mat origSize;
               cv::resize(completedImage, origSize, cv::Size{input.cols, input.rows});
               cv::rectangle(origSize, cv::Rect(0, 0, origSize.cols * progress.getPercent() / 100.0f, 50), cv::Scalar{0, 255, 0}, -1);
               showImage("Progress", origSize);
               cv::waitKey(1);
               std::cout << "Percent progress: " << progress.getPercent() << "%" << std::endl;
               return true;
           });
     cv::Mat completedImage = pm::PatchMatch(config).ImageComplete();
     cv::cvtColor(input, input, cv::COLOR_BGRA2BGR);
     input.copyTo(completedImage, globalMask);
     cv::imwrite(std::get<2>(datasetFolderPath) + "/" + filename, completedImage);
   });
}

int main(int argc, char* argv[])
{
#if 0
   LabelingAllImagesInFolder({std::make_tuple("/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/train_unet/dataset/test_/imgs",
                                              "/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/train_unet/dataset/test_/deteteObjects",
                                              "", "")});
#else
   InpaintingAllImagesInFolder({std::make_tuple("/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/train_unet/dataset/test_/imgs",
                                                "/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/train_unet/dataset/test_/deteteObjects",
                                                "/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/train_unet/dataset/test_/inpaintedImgs",
                                                "/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/train_unet/dataset/test_/masks_cable")/*,
                                std::make_tuple("/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/UNetTorchTrain/dataset/200_folo/2/imgs",
                                                "/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/UNetTorchTrain/dataset/200_folo/2/deteteObjects",
                                                "/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/UNetTorchTrain/dataset/200_folo/2/inpaintedImgs",
                                                "/home/oleksandr_ismailov/WORK/Upwork/deffects_viewer/UNetTorchTrain/dataset/200_folo/2/masks")*/});
#endif
   return 0;
}

#if 0
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
#endif
