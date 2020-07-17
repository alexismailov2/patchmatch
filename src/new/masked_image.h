#pragma once

#include <opencv2/core.hpp>

class MaskedImage {
public:
    MaskedImage(cv::Mat image = {},
                cv::Mat mask = {},
                cv::Mat global_mask = {},
                cv::Mat grady = {},
                cv::Mat gradx = {},
                bool grad_computed = false)
        : m_image{image}
        , m_mask{mask}
        , m_global_mask{global_mask}
        , m_image_grady{grady}
        , m_image_gradx{gradx}
        , m_image_grad_computed{grad_computed}
    {
    }

    MaskedImage(int width, int height)
        : m_image{cv::Mat::zeros(height, width, CV_8UC3)}
        , m_mask{cv::Mat::zeros(height, width, CV_8U)}
    {
    }

    inline MaskedImage clone()
    {
        return MaskedImage(m_image.clone(),
                           m_mask.clone(),
                           m_global_mask.clone(),
                           m_image_grady.clone(),
                           m_image_gradx.clone(),
                           m_image_grad_computed);
    }

    cv::Size size() const
    {
        return m_image.size();
    }

    const cv::Mat &image() const
    {
        return m_image;
    }

    const cv::Mat &mask() const
    {
        return m_mask;
    }

    const cv::Mat &global_mask() const
    {
        return m_global_mask;
    }

    const cv::Mat &grady() const
    {
        assert(m_image_grad_computed);
        return m_image_grady;
    }

    const cv::Mat &gradx() const
    {
        assert(m_image_grad_computed);
        return m_image_gradx;
    }

    void init_global_mask_mat()
    {
        m_global_mask = cv::Mat(m_mask.size(), CV_8U);
        m_global_mask.setTo(cv::Scalar(0));
    }

    void set_global_mask_mat(const cv::Mat &other)
    {
        m_global_mask = other;
    }

    bool is_masked(int y, int x) const
    {
        return static_cast<bool>(m_mask.at<unsigned char>(y, x));
    }

    bool is_globally_masked(int y, int x) const
    {
        return !m_global_mask.empty() && static_cast<bool>(m_global_mask.at<unsigned char>(y, x));
    }

    void set_mask(int y, int x, bool value)
    {
        m_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
    }

    void set_global_mask(int y, int x, bool value)
    {
        m_global_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
    }

    void clear_mask()
    {
        m_mask.setTo(cv::Scalar(0));
    }

    inline const unsigned char *get_image(int y, int x) const {
        return m_image.ptr<unsigned char>(y, x);
    }
    inline unsigned char *get_mutable_image(int y, int x) {
        return m_image.ptr<unsigned char>(y, x);
    }

    inline unsigned char get_image(int y, int x, int c) const {
        return m_image.ptr<unsigned char>(y, x)[c];
    }
    inline int get_image_int(int y, int x, int c) const {
        return static_cast<int>(m_image.ptr<unsigned char>(y, x)[c]);
    }

    bool contains_mask(int y, int x, int patch_size) const;
    MaskedImage downsample() const;
    MaskedImage upsample(int new_w, int new_h) const;
    MaskedImage upsample(int new_w, int new_h, const cv::Mat &new_global_mask) const;
    void compute_image_gradients();
    void compute_image_gradients() const;

    static const cv::Size kDownsampleKernelSize;
    static const int kDownsampleKernel[6];

private:
	cv::Mat m_image;
	cv::Mat m_mask;
  cv::Mat m_global_mask;
  cv::Mat m_image_grady;
  cv::Mat m_image_gradx;
  bool m_image_grad_computed{};
};

