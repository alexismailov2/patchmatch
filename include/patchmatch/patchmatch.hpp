#pragma once

#include <opencv2/core/mat.hpp>

#include <functional>
#include <cstdint>

namespace pm {
/**
 * Class which provide image completion over patch match alghorithm.
 */
    class PatchMatch {
    public:
        /**
         * Proxy struct for getting process metrics.
         */
        struct Progress {
            /**
             * Callback for progress of image reconstruction handling.
             */
            using Cb = std::function<bool(cv::Mat const &, pm::PatchMatch::Progress const &)>;

            /**
             * Calculates average calculated time left in seconds.
             * @return time left in seconds.
             */
            auto getTimeLeftInSeconds() const -> uint32_t;

            /**
             * Calculates percent of planned iterations for each scale.
             * @return percent.
             */
            auto getPercent() const -> float;

            uint8_t countSteps{};   ///< tolal count of steps in each scale.
            uint8_t countScales{};  ///< total count of scales.
            uint8_t currentStep{};  ///< current step in current .
            uint8_t currentScale{}; ///< 0 - origin size, 2 - half size, 3 - quarter size, etc...
        };

        /**
         * Config class for setting input parameters.
         */
        struct Config {
            Config() = default;

            /**
             * Set original image.
             * @param original original image.
             * @return self.
             */
            auto Original(cv::Mat original, uint8_t scales = 0) -> Config &;

            /**
             * Set mask for image(should be grayscale CV_8UC1).
             * @param mask mask image.
             * @return self.
             */
            auto Mask(cv::Mat mask) -> Config &;

            /**
             * Set steps which will be used for each scale.
             * @param steps steps for each scale.
             * @return self.
             */
            auto Steps(uint8_t steps) -> Config &;

            /**
             * Set scales count(power of two divisions from original size).
             * @param scales power of two divisions from original size.
             * @return self.
             */
            auto Scales(uint8_t scales) -> Config &;

            /**
             * Set proggess callback.
             * @param progressCb progress callback.
             * @return self.
             */
            auto ProgressCb(Progress::Cb &&progressCb) -> Config &;

        private:
            cv::Mat _original{};
            cv::Mat _mask{};
            uint8_t _steps{5};
            uint8_t _scales{};
            uint8_t _possibleScales{};
            Progress::Cb _progressCb{[](cv::Mat const &, pm::PatchMatch::Progress const &) -> bool { return true; }};
            bool _hasProgressCb{};
            friend PatchMatch;
        };

    public:
        /**
         * Constructor which provide to set config.
         * @param config config which let to use named parameter idiom for setting needed params.
         */
        PatchMatch(Config const &config);

        /**
         * Image completion algorithm start.
         * @return completed image.
         */
        auto ImageComplete() -> cv::Mat;

    public:
        /**
         * Image completion alghorithm.
         * @param original original image.
         * @param mask missed region mask.
         * @param progressCb slice of a result on each step.
         * @return completed image.
         */
        static auto imageComplete(Config const &config) -> cv::Mat;

    private:
        Config _config;
    };

} /// end namespace pm
