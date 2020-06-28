#pragma once

#include <functional>

#include <opencv2/core/mat.hpp>

/**
 * Callback for progress of image reconstruction handling.
 */
using imageCompleteProgressCb = std::function<void(cv::Mat const& progress)>;

/**
 * Image completion alghorithm.
 * @param original original image.
 * @param mask missed region mask.
 * @param progressCb slice of a result on each step.
 * @return completed image.
 */
auto imageComplete(cv::Mat original,
                   cv::Mat mask,
                   imageCompleteProgressCb progressCb = [](cv::Mat const&){}) -> cv::Mat;
