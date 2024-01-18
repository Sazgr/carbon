#pragma once

#include <cstdint>
#include <array>
#include <algorithm>
#include <immintrin.h>
#include "types.h"

template<typename T = float>
static inline const T ReLU(const T x){
    return std::max<T>(0, x);
}

template<typename T = float>
static inline const T CReLU(const T x){
    return std::clamp<T>(x, 0, 1);
}

template<typename T = float>
static inline const T SCReLU(const T x){
    T clipped = std::clamp<T>(x, 0, 1);
    return clipped * clipped;
}

template<std::size_t N>
static inline void vecReLU(float* v) {
  constexpr size_t width = sizeof(__m256) / sizeof(float);
  constexpr size_t chunks = N / width;

  const __m256 zero = _mm256_setzero_ps();

  __m256* vector = (__m256*)v;

  for (size_t j = 0; j < chunks; j++) vector[j] = _mm256_max_ps(zero, vector[j]);
}

template<std::size_t N>
float vecDotProduct(float* v1, float* v2) {
  constexpr size_t width = sizeof(__m256) / sizeof(float);
  constexpr size_t chunks = N / width;

  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();

  __m256* vector1 = (__m256*)v1;
  __m256* vector2 = (__m256*)v2;

  for (size_t j = 0; j < chunks; j += 2) {
    s0 = _mm256_add_ps(_mm256_mul_ps(vector1[j], vector2[j]), s0);
    s1 = _mm256_add_ps(_mm256_mul_ps(vector1[j + 1], vector2[j + 1]), s1);
  }

  const __m256 r8 = _mm256_add_ps(s0, s1);
  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  const __m128 r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 0x1));
  return _mm_cvtss_f32(r1);
}

template<typename T = float>
static inline const T ReLUPrime(const T x){
    return x > 0 ? 1 : 0;
} 

template<typename T = float>
static inline const T CReLUPrime(const T x){
    return x > 0 && x < 1 ? 1 : 0;
}

template<typename T = float>
static inline const T SCReLUPrime(const T x){
    return x > 0 && x < 1 ? 2 * x : 0;
} 

template<typename T = float>
static inline const T sigmoid(const T x){
    return 1 / (1 + expf(-x));
}

template<typename T = float>
static inline const T sigmoidPrime(const T x){
    return sigmoid(x) * (1 - sigmoid(x));
}

inline float errorFunction(float output, float eval, float wdl) {
    float expected = EVAL_CP_RATIO * sigmoid(eval) + (1 - EVAL_CP_RATIO) * wdl;
    return pow(sigmoid(output) - expected, 2);
}

struct Features
{
    uint8_t n = 0;
    std::array<std::array<int16_t, 2>, 32> features;

    uint8_t stm;

    inline void add(int16_t featureWhite, int16_t featureBlack)
    {
        features[n][0] = featureWhite;
        features[n][1] = featureBlack;
        n++;
    }
    inline void clear()
    {
        stm = 0;
        n = 0;
    }
};

struct NN {
    using Accumulator = std::array<float, HIDDEN_SIZE * 2>;
    using Color = uint8_t;

    std::array<float, INPUT_SIZE * HIDDEN_SIZE> inputFeatures;
    std::array<float, HIDDEN_SIZE> inputBias;
    std::array<float, HIDDEN_SIZE * 2> hiddenFeatures;
    std::array<float, OUTPUT_SIZE> hiddenBias;

    NN(){
        std::random_device rd;
        std::mt19937                    gen(rd());
        std::normal_distribution<float> input_distribution(0.0, std::sqrt(1.0 / static_cast<float>(32)));
        std::normal_distribution<float> hidden_distribution(0.0, std::sqrt(1.0 / static_cast<float>(HIDDEN_SIZE)));

        for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
            inputFeatures[i] = input_distribution(gen);
        }

        for (int i = 0; i < HIDDEN_SIZE * 2; i++) {
            hiddenFeatures[i] = hidden_distribution(gen);
        }

        std::memset(inputBias.data(), 0, sizeof(float) * HIDDEN_SIZE);
        std::memset(hiddenBias.data(), 0, sizeof(float) * OUTPUT_SIZE);
    }

    float forward(Accumulator& accumulator, Accumulator& activated, const Features& features, Color stm) const;
    void testFen(const std::string& fen) const;
    void load(const std::string& path);
    void save(const std::string& path);
    void quantize(const std::string& path, bool print = false);


    friend std::ostream& operator<<(std::ostream& os, const NN& nn) {
        os << "Neural Network Summary:" << std::endl;

        os << "Input Features:" << std::endl;
        for (int i = 0; i < 16; i++) {
            os << std::setw(5) << nn.inputFeatures[i] << " ";
        }
        os << std::endl;

        os << "Input Bias:" << std::endl;
        for (int i = 0; i < 16; i++) {
            os << std::setw(5) << nn.inputBias[i] << " ";
        }
        os << std::endl;

        os << "Hidden Features:" << std::endl;
        for (int i = 0; i < 16; i++) {
            os << std::setw(5) << nn.hiddenFeatures[i] << " ";
        }
        os << std::endl;

        os << "Hidden Bias:";
        os << std::setw(5) << nn.hiddenBias[0];
        os << std::endl;

        return os;
    }
};

struct NNTrace {
    NN::Accumulator accumulator;
    NN::Color stm = 0;
    Features features;

    float wdl = 0;
    float eval = 0;
    float output = 0;
};