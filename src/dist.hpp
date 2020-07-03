#pragma once

#include <opencv2/core/mat.hpp>

#ifndef SSE
#define SSE  1
#endif

#ifndef NEON
#define NEON 0
#endif

#if SSE
#include <emmintrin.h>
#include <immintrin.h>

inline uint64_t dist_sse(unsigned char const* a, unsigned char const* b, size_t offset)
{
  __m128i ab16{};
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)a),  _mm_loadu_si128((__m128i*)b)));
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)(a + 16)),_mm_loadu_si128((__m128i*)(b + 16))));
  a += offset; b += offset;
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)a),  _mm_loadu_si128((__m128i*)b)));
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)(a + 16)),_mm_loadu_si128((__m128i*)(b + 16))));
  a += offset; b += offset;
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)a),  _mm_loadu_si128((__m128i*)b)));
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)(a + 16)),_mm_loadu_si128((__m128i*)(b + 16))));
  a += offset; b += offset;
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)a),  _mm_loadu_si128((__m128i*)b)));
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)(a + 16)),_mm_loadu_si128((__m128i*)(b + 16))));
  a += offset; b += offset;
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)a),  _mm_loadu_si128((__m128i*)b)));
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)(a + 16)),_mm_loadu_si128((__m128i*)(b + 16))));
  a += offset; b += offset;
  ab16 = _mm_add_epi16(ab16,  _mm_sad_epu8(_mm_loadu_si128((__m128i*)a),  _mm_loadu_si128((__m128i*)b)));
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)(a + 16)),_mm_loadu_si128((__m128i*)(b + 16))));
  a += offset; b += offset;
  ab16 = _mm_add_epi16(ab16,  _mm_sad_epu8(_mm_loadu_si128((__m128i*)a),  _mm_loadu_si128((__m128i*)b)));
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)(a + 16)),_mm_loadu_si128((__m128i*)(b + 16))));
  a += offset; b += offset;
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)a),  _mm_loadu_si128((__m128i*)b)));
  ab16 = _mm_add_epi16(ab16, _mm_sad_epu8(_mm_loadu_si128((__m128i*)(a + 16)),_mm_loadu_si128((__m128i*)(b + 16))));
  return ab16[0] + ab16[1];
}
#elif NEON
# include <arm_neon.h>

inline uint64_t dist_neon(unsigned char const* a, unsigned char const* b, size_t offset)
{
  uint16x8_t abd16 = vpaddlq_u8(vabaq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), vld1q_u8(a + 16), vld1q_u8(b + 16)));
  a += offset; b += offset;
  abd16 = vpadalq_u8(abd16, vabaq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), vld1q_u8(a + 16), vld1q_u8(b + 16)));
  a += offset; b += offset;
  abd16 = vpadalq_u8(abd16, vabaq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), vld1q_u8(a + 16), vld1q_u8(b + 16)));
  a += offset; b += offset;
  abd16 = vpadalq_u8(abd16, vabaq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), vld1q_u8(a + 16), vld1q_u8(b + 16)));
  a += offset; b += offset;
  abd16 = vpadalq_u8(abd16, vabaq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), vld1q_u8(a + 16), vld1q_u8(b + 16)));
  a += offset; b += offset;
  abd16 = vpadalq_u8(abd16, vabaq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), vld1q_u8(a + 16), vld1q_u8(b + 16)));
  a += offset; b += offset;
  abd16 = vpadalq_u8(abd16, vabaq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), vld1q_u8(a + 16), vld1q_u8(b + 16)));
  a += offset; b += offset;
  abd16 = vpadalq_u8(abd16, vabaq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), vld1q_u8(a + 16), vld1q_u8(b + 16)));

  auto res32 = vpaddlq_u16(abd16);
  auto res64 = vpaddlq_u32(res32);
  return vgetq_lane_u64(res64, 0) + vgetq_lane_u64(res64, 1);
}
#endif

inline int dist(cv::Mat const& a, cv::Mat const& b, int ax, int ay, int bx, int by, int cutoff=INT_MAX)
{
#if SSE
  return dist_sse(a.ptr(ay, ax), b.ptr(by, bx), a.ptr(ay + 1, ax) - a.ptr(ay, ax));
#elif NEON
  return dist_neon(a.ptr(ay, ax), b.ptr(by, bx), a.cols * 4);
#else
   auto ans{0};
   for (auto dy = 0; dy < 8; dy++)
   {
      for (auto dx = 0; dx < 8; dx++)
      {
         auto const& ac = a.at<cv::Vec4b>(ay + dy, ax + dx);
         auto const& bc = b.at<cv::Vec4b>(by + dy, bx + dx);
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
