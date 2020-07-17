#pragma once

#include <vector>
#include <patchmatch/patchmatch.hpp>

#include "masked_image.h"
#include "nnf.h"

class Inpainting {
public:
    Inpainting(cv::Mat image, cv::Mat mask, const PatchDistanceMetric *metric);
    Inpainting(cv::Mat image, cv::Mat mask, cv::Mat global_mask, const PatchDistanceMetric *metric);
    cv::Mat run(bool verbose = false, pm::PatchMatch::Progress::Cb verbose_visualize = nullptr, unsigned int random_seed = 1212);

private:
    void _initialize_pyramid(void);
    MaskedImage _expectation_maximization(MaskedImage source, MaskedImage target, int level, bool verbose);
    void _expectation_step(const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source, bool upscaled);
    void _maximization_step(MaskedImage &target, const cv::Mat &vote);

private:
    pm::PatchMatch::Config _config;

    MaskedImage m_initial;
    std::vector<MaskedImage> m_pyramid;

    NearestNeighborField m_source2target;
    NearestNeighborField m_target2source;
    const PatchDistanceMetric *m_distance_metric;
};

