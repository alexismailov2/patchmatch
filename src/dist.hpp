#pragma once

#include <opencv2/core/mat.hpp>

#define SSE  0
#define NEON 1

#define RGB24 0

#if SSE
#include <emmintrin.h>
#include <immintrin.h>

inline void dist_sse(unsigned char const* a, unsigned char const* b, __m128i& ab16)
{
#if RGB24
   auto a16 = _mm_loadu_si128((__m128i*)a);
   auto b16 = _mm_loadu_si128((__m128i*)b);
   auto ab = _mm_sad_epu8(a16, b16);
   ab16 = _mm_add_epi16(ab16, ab);
   auto a16_ = _mm_loadl_epi64((__m128i*)(a + 16));
   auto b16_ = _mm_loadl_epi64((__m128i*)(b + 16));
   auto ab_ = _mm_sad_epu8(a16_, b16_);
   ab16 = _mm_add_epi16(ab16, ab_);
#else
   auto a16 = _mm_loadu_si128((__m128i*)a);
   auto b16 = _mm_loadu_si128((__m128i*)b);
   auto ab = _mm_sad_epu8(*(__m128i*)a, *(__m128i*)b);
   ab16 = _mm_add_epi16(ab16, ab);
   auto a16_ = _mm_loadu_si128((__m128i*)(a + 16));
   auto b16_ = _mm_loadu_si128((__m128i*)(b + 16));
   auto ab_ = _mm_sad_epu8(*(__m128i*)(a + 16), *(__m128i*)(b + 16));
   ab16 = _mm_add_epi16(ab16, ab_);
#endif
}
#elif NEON
# include <arm_neon.h>

inline void dist_neon(unsigned char const* a, unsigned char const* b, uint32_t& sum)
{
#if RGB24
  auto a16 = vld1q_u8(a);
  auto b16 = vld1q_u8(b);
  auto ab = vabdq_u8(a16, b16);
  sum += ab[0] + ab[1] + ab[2] + ab[3] + ab[4] + ab[5] + ab[6] + ab[7] + ab[8] + ab[9] + ab[10] + ab[11] + ab[12] + ab[13] + ab[14] + ab[15];
  auto a16_ = vld1_u8(a + 16);
  auto b16_ = vld1_u8(b + 16);
  auto ab_ = vabd_u8(a16_, b16_);
  uint8_t ab8[8];
  vst1_u8(ab8, ab_);
  sum += ab8[0] + ab8[1] + ab8[2] + ab8[3] + ab8[4] + ab8[5] + ab8[6] + ab8[7];
#else
  auto a16 = vld1q_u8(a);
  auto b16 = vld1q_u8(b);
  auto ab = vabdq_u8(a16, b16);
  auto a16_ = vld1q_u8(a + 16);
  auto b16_ = vld1q_u8(b + 16);
  auto ab_ = vabdq_u8(a16_, b16_);
  sum += ab[0] + ab[1] + ab[2] + ab[3] + ab[4] + ab[5] + ab[6] + ab[7] + ab[8] + ab[9] + ab[10] + ab[11] + ab[12] + ab[13] + ab[14] + ab[15];
  sum += ab_[0] + ab_[1] + ab_[2] + ab_[3] + ab_[4] + ab_[5] + ab_[6] + ab_[7] + ab_[8] + ab_[9] + ab_[10] + ab_[11] + ab_[12] + ab_[13] + ab_[14] + ab_[15];
#endif
}
#endif

inline int dist(cv::Mat const& a, cv::Mat const& b, int ax, int ay, int bx, int by, int cutoff=INT_MAX)
{
#if SSE
   __m128i ab16{};
  {
    dist_sse(a.ptr(ay++, ax), b.ptr(by++, bx), ab16);
    dist_sse(a.ptr(ay++, ax), b.ptr(by++, bx), ab16);
    dist_sse(a.ptr(ay++, ax), b.ptr(by++, bx), ab16);
    dist_sse(a.ptr(ay++, ax), b.ptr(by++, bx), ab16);
    dist_sse(a.ptr(ay++, ax), b.ptr(by++, bx), ab16);
    dist_sse(a.ptr(ay++, ax), b.ptr(by++, bx), ab16);
    dist_sse(a.ptr(ay++, ax), b.ptr(by++, bx), ab16);
    dist_sse(a.ptr(ay, ax), b.ptr(by, bx), ab16);
  }
  uint16_t ab16_[8];
  _mm_storel_epi64((__m128i*)ab16_, ab16);
  return ab16[0] + ab16[1];
#elif NEON
  uint32_t sum{};
  dist_neon(a.ptr(ay++, ax), b.ptr(by++, bx), sum);
  dist_neon(a.ptr(ay++, ax), b.ptr(by++, bx), sum);
  dist_neon(a.ptr(ay++, ax), b.ptr(by++, bx), sum);
  dist_neon(a.ptr(ay++, ax), b.ptr(by++, bx), sum);
  dist_neon(a.ptr(ay++, ax), b.ptr(by++, bx), sum);
  dist_neon(a.ptr(ay++, ax), b.ptr(by++, bx), sum);
  dist_neon(a.ptr(ay++, ax), b.ptr(by++, bx), sum);
  dist_neon(a.ptr(ay, ax),   b.ptr(by, bx),   sum);
  return sum;
#else
   auto ans{0};
   for (auto dy = 0; dy < 8; dy++)
   {
      for (auto dx = 0; dx < 8; dx++)
      {
#if RGB24
         auto const& ac = a.at<cv::Vec3b>(ay + dy, ax + dx);
         auto const& bc = b.at<cv::Vec3b>(by + dy, bx + dx);
#else
         auto const& ac = a.at<cv::Vec4b>(ay + dy, ax + dx);
         auto const& bc = b.at<cv::Vec4b>(by + dy, bx + dx);
#endif
         auto const db = std::abs(ac[0] - bc[0]);
         auto const dg = std::abs(ac[1] - bc[1]);
         auto const dr = std::abs(ac[2] - bc[2]);
         auto const da = std::abs(ac[3] - bc[3]);
         ans += dr + dg + db + da;
      }
   }
   return (ans >= cutoff) ? cutoff : ans;
#endif
}
