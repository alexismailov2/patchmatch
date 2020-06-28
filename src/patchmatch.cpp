#include <patchmatch/patchmatch.hpp>

#include "TimeStamp.hpp"
#include "dist.hpp"
#include "debug.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <cassert>

namespace {
constexpr int patch_w  = 8;
constexpr int pm_iters = 5;
constexpr int rs_max   = INT_MAX;
constexpr int sigma    = 1 * patch_w * patch_w;

constexpr uint32_t shift = 16U;
constexpr uint32_t XY_TO_INT(uint32_t x, uint32_t y)
{
  return (y << shift) | x;
}

constexpr uint32_t INT_TO_X(uint32_t v)
{
  return v & ((1U << shift) - 1);
}

constexpr uint32_t INT_TO_Y(uint32_t v)
{
  return v >> shift;
}

/**
 * Find bounding rect for mask.
 * @param mask mask with shape of selected region.
 * @return bounding box as cv::Rect.
 */
auto getBox(cv::Mat const& mask) -> cv::Rect
{
  auto const rect = cv::boundingRect(mask);
  auto xmin = rect.x;
  auto ymin = rect.y;
  auto xmax = rect.x + rect.width;
  auto ymax = rect.y + rect.height;
  xmin = xmin - patch_w + 1;
  ymin = ymin - patch_w + 1;
  xmin = (xmin < 0) ? 0 : xmin;
  ymin = (ymin < 0) ? 0 : ymin;

  xmax = (xmax > mask.cols - patch_w + 1) ? mask.cols - patch_w +1 : xmax;
  ymax = (ymax > mask.rows - patch_w + 1) ? mask.rows - patch_w +1 : ymax;

  return cv::Rect{xmin, ymin, xmax - xmin, ymax - ymin};
}

/**
 * Calc dist and check if it better than exist.
 * @param a image.
 * @param b another image.
 * @param ax coordinate of a image.
 * @param ay coordinate of a image.
 * @param xbest non const reference to best match coordinate.
 * @param ybest non const reference to best match coordimate.
 * @param dbest non const reference to best match difference.
 * @param bx coordinate of b image.
 * @param by coordinate of b image.
 */
void improve_guess(cv::Mat const& a, cv::Mat const& b, int ax, int ay, int &xbest, int &ybest, int &dbest, int bx, int by)
{
  auto const d = dist(a, b, ax, ay, bx, by, dbest);
  if ((d < dbest) && (ax != bx || ay != by))
  {
    dbest = d;
    xbest = bx;
    ybest = by;
  }
}

/**
 * Match image a to image b.
 * @param a image
 * @param b another image
 * @param ann_ a to b coords
 * @param annd_ a to b abs difference
 * @param dilated_mask dilated mask
 */
void patchmatch(cv::Mat const& a, cv::Mat const& b, cv::Mat& ann_, cv::Mat& annd_, cv::Mat const& dilated_mask)
{
  TAKEN_TIME_US();

  // Effective width and height (possible upper left corners of patches).
  int aew = a.cols - patch_w + 1;
  int aeh = a.rows - patch_w + 1;
  int bew = b.cols - patch_w + 1;
  int beh = b.rows - patch_w + 1;

  // Initialization
  for (int ay = 0; ay < aeh; ay++)
  {
    for (int ax = 0; ax < aew; ax++)
    {
      int bx, by;
      bool valid = false;
      while (!valid)
      {
        bx = rand() % bew;
        by = rand() % beh;
        int mask_pixel = (int) dilated_mask.at<uchar>(by, bx);
        // should find patches outside the hole
        if (mask_pixel == 255)
        {
          valid = false;
        }
        else
        {
          valid = true;
        }
      }
      ann_.at<int32_t>(ay, ax) = XY_TO_INT(bx, by);
      annd_.at<int32_t>(ay, ax) = dist(a, b, ax, ay, bx, by);
    }
  }

  for (int iter = 0; iter < pm_iters; iter++)
  {
    // In each iteration, improve the NNF, by looping in scanline or reverse-scanline order.
    int ystart = 0;
    int yend = aeh;
    int ychange = 1;
    int xstart = 0;
    int xend = aew;
    int xchange = 1;
    if (iter % 2 == 1)
    {
      xstart = xend-1;
      xend = -1;
      xchange = -1;
      ystart = yend-1;
      yend = -1;
      ychange = -1;
    }

    {
      TAKEN_TIME_US();
      std::cout << "count of ops: " << aeh * aew << std::endl;
      for (int ay = ystart; ay != yend; ay += ychange)
      {
        for (int ax = xstart; ax != xend; ax += xchange)
        {
          /* Current (best) guess. */
          int v = ann_.at<int32_t>(ay, ax);//(*ann)[ay][ax];
          int xbest = INT_TO_X(v);
          int ybest = INT_TO_Y(v);
          int dbest = annd_.at<int32_t>(ay, ax);//(*annd)[ay][ax];

          {
            // Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations).
            if ((unsigned)(ax - xchange) < (unsigned)aew)
            {
              int vp = ann_.at<int32_t>(ay, ax - xchange);
              int xp = INT_TO_X(vp) + xchange;
              int yp = INT_TO_Y(vp);

              if (((unsigned)xp < (unsigned)aew))
              {
                const int mask_pixel = (int)dilated_mask.at<uchar>(yp, xp);
                if (mask_pixel != 255)
                {
                  improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                }
              }
            }

            if ((unsigned)(ay - ychange) < (unsigned)aeh)
            {
              int vp = ann_.at<int32_t>(ay - ychange, ax);
              int xp = INT_TO_X(vp);
              int yp = INT_TO_Y(vp) + ychange;

              if (((unsigned)yp < (unsigned)aeh))
              {
                const int mask_pixel = (int)dilated_mask.at<uchar>(yp, xp);
                if (mask_pixel != 255)
                {
                  improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                }
              }
            }
          }
          {
            // Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess.
            int rs_start = rs_max;
            if (rs_start > MAX(b.cols, b.rows))
            {
              rs_start = MAX(b.cols, b.rows);
            }
            for (int mag = rs_start; mag >= 1; mag /= 2)
            {
              // Sampling window
              int xmin = MAX(xbest - mag, 0);
              int xmax = MIN(xbest + mag + 1, bew);
              int ymin = MAX(ybest - mag, 0);
              int ymax = MIN(ybest + mag + 1, beh);
              bool do_improve = false;
              do
              {
                int xp = xmin + rand() % (xmax - xmin);
                int yp = ymin + rand() % (ymax - ymin);
                int mask_pixel = (int)dilated_mask.at<uchar>(yp, xp);
                if (mask_pixel != 255)
                {
                  improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                  do_improve = true;
                }
              } while (!do_improve);
            }

            ann_.at<int32_t>(ay, ax) = XY_TO_INT(xbest, ybest);
            annd_.at<int32_t>(ay, ax) = dbest;
          }
        }
      }
    }
  }
}

/**
 * Random initialization.
 * @param img image.
 * @param mask mask.
 */
void randomnInit(cv::Mat& img, cv::Mat const& mask)
{
   TAKEN_TIME_US();
   // TODO: Should be used this one instead
   //cv::Mat random;
   //cv::randu(random, cv::Scalar(0, 0, 0), Scalar(255, 255, 255));

   for (auto y = 0; y < img.rows; ++y)
   {
      for (auto x = 0; x < img.cols; ++x)
      {
         if (mask.at<uchar>(y, x) != 0)
         {
            img.at<cv::Vec3b>(y, x)[0] = rand() % 256;
            img.at<cv::Vec3b>(y, x)[1] = rand() % 256;
            img.at<cv::Vec3b>(y, x)[2] = rand() % 256;
            img.at<cv::Vec3b>(y, x)[3] = 255;
         }
      }
   }
}
} /// end namespace anonymous

auto imageComplete(cv::Mat original,
                   cv::Mat mask,
                   imageCompleteProgressCb progressCb) -> cv::Mat
{
  TAKEN_TIME_US();
  // some parameters for scaling
  int rows = original.rows;
  int cols = original.cols;
  int startscale = (int)-1*ceil(log2(std::min(rows, cols))) + 5;
  //int startscale = -3;
  double scale = pow(2, startscale);

  // Resize image to starting scale
  cv::Mat resize_img;
  cv::Mat resize_mask;
  cv::resize(original, resize_img, cv::Size(), scale, scale, cv::INTER_AREA);
  cv::resize(mask, resize_mask, cv::Size(), scale, scale, cv::INTER_AREA);
  cv::threshold(resize_mask, resize_mask, 127, 255, 0);

  randomnInit(resize_img, resize_mask);

  // just for DEBUG
  //int index = 0;

  // go through all scale
  for (int logscale = startscale; logscale <= -1; logscale++)
  {
    //index++;
    scale = pow(2, logscale);
    std::cout << "Scaling is " << scale << std::endl;
    auto const mask_box = getBox(resize_mask);

    // dilate the mask
    //
    // if patch_w = 3
    // kernel width = 5 , 0 1 2 is 1
    // pixel is    result should be
    // 0 0 0 0     1 1 1 0
    // 0 0 0 0     1 1 1 0
    // 0 0 1 0     1 1 1 0
    // 0 0 0 0     0 0 0 0
    cv::Mat element = cv::Mat::zeros(2*patch_w - 1, 2*patch_w - 1, CV_8UC1);
    element(cv::Rect(patch_w - 1, patch_w - 1, patch_w, patch_w)) = 255;
    cv::Mat dilated_mask;
    dilate(resize_mask, dilated_mask, element);

    // iterations of image completion
    int im_iterations = 5;
    for (int im_iter = 0; im_iter < im_iterations; ++im_iter)
    {
      cv::Mat B = resize_img.clone();
      bitwise_and(resize_img, 0, B, resize_mask);

      // use patchmatch to find NN
      cv::Mat ann_ = cv::Mat::zeros(resize_img.rows, resize_img.cols, CV_32S);
      cv::Mat annd_ = cv::Mat::zeros(resize_img.rows, resize_img.cols, CV_32S);

      patchmatch(resize_img, B, ann_, annd_, dilated_mask);

      double t3 =  (double)cv::getTickCount();
      // create new image by letting each patch vote
      cv::Mat R = cv::Mat::zeros(resize_img.rows, resize_img.cols, CV_32FC4);
      cv::Mat Rcount = cv::Mat::zeros(resize_img.rows, resize_img.cols, CV_32FC4);

      for (int y = mask_box.y; y < mask_box.y + mask_box.height; ++y)
      {
        for (int x = mask_box.x; x < mask_box.x + mask_box.width; ++x)
        {
            int v = ann_.at<int32_t>(y, x);
            int xbest  = INT_TO_X(v);
            int ybest = INT_TO_Y(v);
            cv::Rect srcRect(cv::Point(x, y), cv::Size(patch_w, patch_w));
            cv::Rect dstRect(cv::Point(xbest, ybest), cv::Size(patch_w, patch_w));
            auto d = (float const) annd_.at<int32_t>(y, x);
            float sim = exp(-d / (2*pow(sigma, 2) ));
            cv::Mat toAssign;
            addWeighted(R(srcRect), 1.0, resize_img(dstRect), sim, 0, toAssign, CV_32FC4);
            toAssign.copyTo(R(srcRect));
            add(Rcount(srcRect), sim, toAssign, cv::noArray(), CV_32FC4);
            toAssign.copyTo(Rcount(srcRect));
        }
      }
      double p3 = ((double)cv::getTickCount() - t3) / cv::getTickFrequency();
      std::cout << "time for voting = " << p3 << std::endl;

      R /= Rcount;
      R.convertTo(R, CV_8UC3);

      // keep pixel outside mask
      cv::Mat old_img = resize_img.clone();
      R.copyTo(resize_img, resize_mask);

      // measure how much image has changed, if not much then stop  TODO
      if (im_iter > 0)
      {
        double diff = 0;
        int mask_count_white = 0;
        int mask_count_black = 0;
        int mask_count_other = 0;
        for (int h = 0; h < resize_img.rows; h++)
        {
          for (int w = 0; w < resize_img.cols; w++)
          {
            int mask_pixel = (int) resize_mask.at<uchar>(h, w);
            // white pixel in mask is hole
            if (mask_pixel == 255)
            {
              cv::Vec3b const& new_pixel = resize_img.at<cv::Vec3b>(h, w);
              cv::Vec3b const& old_pixel = old_img.at<cv::Vec3b>(h, w);
              diff += pow(new_pixel[0] - old_pixel[0], 2);
              diff += pow(new_pixel[1] - old_pixel[1], 2);
              diff += pow(new_pixel[2] - old_pixel[2], 2);
              mask_count_white += 1;
            } else if (mask_pixel == 0) {
              mask_count_black += 1;
            } else {
              mask_count_other += 1;
            }
          }
        }
        assert(mask_count_other == 0);
        std::cout << "diff is " << diff << std::endl;
        std::cout << "mask count is " << mask_count_white << std::endl;
        std::cout << "norm diff is " << diff/mask_count_white << std::endl;
        if (diff/mask_count_white < 0.02)
        {
          break;
        }
      }
      progressCb(R);
    }


    if (logscale < 0)
    {
      cv::Mat upscale_img;
      cv::resize(original, upscale_img, cv::Size(), 2*scale, 2*scale, cv::INTER_AREA);

      cv::resize(resize_img, resize_img, cv::Size(upscale_img.cols, upscale_img.rows), 0, 0, cv::INTER_CUBIC);
      cv::resize(mask, resize_mask, cv::Size(upscale_img.cols, upscale_img.rows), 0, 0, cv::INTER_AREA);

      cv::threshold(resize_mask, resize_mask, 127, 255, 0);

      cv::Mat inverted_mask;
      cv::bitwise_not(resize_mask, inverted_mask);
      upscale_img.copyTo(resize_img, inverted_mask);
    }
  }

  progressCb(resize_img);
  return resize_img.clone();
}
