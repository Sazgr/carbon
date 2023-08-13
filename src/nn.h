#pragma once

#include <cstdint>
#include <array>
#include <algorithm>
#include "types.h"

float errorFunction(float output, float eval, float wdl);
float errorGradient(float output, float eval, float wdl);

template<typename T = float>
static inline const T ReLU(const T x){
    return std::max<T>(0, x);
} 

template<typename T = float>
static inline const T ReLUPrime(const T x){
    return x > 0 ? 1 : 0;
} 

template<typename T = float>
static inline const T sigmoid(const T x){
    return 1 / (1 + expf(-x));
}

template<typename T = float>
static inline const T sigmoidPrime(const T x){
    return sigmoid(x) * (1 - sigmoid(x));
}

struct Features
{
    uint8_t n = 0;
    std::array<std::array<int16_t, 2>, 32> features;

    void add(int16_t featureWhite, int16_t featureBlack)
    {
        features[n][0] = featureWhite;
        features[n][1] = featureBlack;
        n++;
    }
    void clear()
    {
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

    float forward(Accumulator& accumulator, Features& features, Color stm) const;
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