#pragma once

#include "types.h"
#include <array>
#include <cstring>

struct Gradient {
    float M = 0;
    float V = 0;

    void clear() {
        M = 0;
        V = 0;
    }
};

struct NNGradients {
    std::array<Gradient, INPUT_SIZE * HIDDEN_SIZE> inputFeatures;
    std::array<Gradient, HIDDEN_SIZE>              inputBias;
    std::array<Gradient, HIDDEN_SIZE * OUTPUT_SIZE * 2>          hiddenFeatures;
    std::array<Gradient, OUTPUT_SIZE>              hiddenBias;

    NNGradients() {
        clear();
    }

    void clear() {
        std::memset(inputFeatures.data(), 0, sizeof(Gradient) * INPUT_SIZE * HIDDEN_SIZE);
        std::memset(inputBias.data(), 0, sizeof(Gradient) * HIDDEN_SIZE);
        std::memset(hiddenFeatures.data(), 0, sizeof(Gradient) * HIDDEN_SIZE * OUTPUT_SIZE * 2);
        std::memset(hiddenBias.data(), 0, sizeof(Gradient) * OUTPUT_SIZE);
    }
};

struct BatchGradients {
    std::array<float, INPUT_SIZE * HIDDEN_SIZE> inputFeatures;
    std::array<float, HIDDEN_SIZE>              inputBias;
    std::array<float, HIDDEN_SIZE * OUTPUT_SIZE * 2>          hiddenFeatures;
    std::array<float, OUTPUT_SIZE>              hiddenBias;

    BatchGradients() {
        clear();
    }

    void clear(){
        std::memset(inputFeatures.data(), 0, sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
        std::memset(inputBias.data(), 0, sizeof(float) * HIDDEN_SIZE);
        std::memset(hiddenFeatures.data(), 0, sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE * 2);
        std::memset(hiddenBias.data(), 0, sizeof(float) * OUTPUT_SIZE);
    }
};