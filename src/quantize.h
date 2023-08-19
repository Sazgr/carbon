#pragma once

#include "types.h"
#include "nn.h"
#include <iostream>
#include <fstream>

constexpr int Q1 = 64;
constexpr int Q2 = 128;

class QuantizedNN {
    public:
    using input_type = int16_t;
    using hidden_type = int16_t;
    using Accumulator = std::array<int16_t, HIDDEN_SIZE * 2>;
    using Color = uint8_t;

    std::array<input_type, INPUT_SIZE * HIDDEN_SIZE> inputFeatures;
    std::array<input_type, HIDDEN_SIZE> inputBias;
    std::array<hidden_type, HIDDEN_SIZE * 2> hiddenFeatures;
    std::array<int32_t, OUTPUT_SIZE> hiddenBias;

    QuantizedNN(const NN& nn, bool print = false){
        float inputMax = 0.0f;
        float inputBiasMax = 0.0f;
        float hiddenMax = 0.0f;
        float hiddenBiasMax = 0.0f;

        for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
            float w = nn.inputFeatures[i];
            inputMax = std::max(inputMax, w);
            int16_t a = static_cast<int16_t>(std::round(w * Q1));
            inputFeatures[i] = a;
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float b = nn.inputBias[i];
            inputBiasMax = std::max(inputBiasMax, b);
            int16_t a = static_cast<int16_t>(std::round(b * Q1));
            inputBias[i] = a;
        }

        for (int i = 0; i < HIDDEN_SIZE * 2; i++) {
            float w = nn.hiddenFeatures[i];
            hiddenMax = std::max(hiddenMax, w);
            int16_t a = static_cast<int16_t>(std::round(w * Q2));
            hiddenFeatures[i] = a;
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float b = nn.hiddenBias[i];
            hiddenBiasMax = std::max(hiddenBiasMax, b);
            int32_t a = static_cast<int32_t>(std::round(b * Q1 * Q2));
            hiddenBias[i] = a;
        }

        if (print){
            std::cout << "inputMax: " << inputMax << "\n";
            std::cout << "inputBiasMax: " << inputBiasMax << "\n";
            std::cout << "hiddenMax: " << hiddenMax << "\n";
            std::cout << "hiddenBiasMax: " << hiddenBiasMax << "\n";
        }
    }

    void save(const std::string& path){
        std::ofstream file(path, std::ios::binary);

        if (file){
            file.write(reinterpret_cast<char*>(inputFeatures.data()), sizeof(inputFeatures));
            file.write(reinterpret_cast<char*>(inputBias.data()), sizeof(inputBias));
            file.write(reinterpret_cast<char*>(hiddenFeatures.data()), sizeof(hiddenFeatures));
            file.write(reinterpret_cast<char*>(hiddenBias.data()), sizeof(hiddenBias));
        }else{
            std::cout << "Couldn't write quantized file " << path << std::endl;
        }
    }

    const int32_t forward(Accumulator& accumulator, Features& features, Color stm) const;

    friend std::ostream& operator<<(std::ostream& os, const QuantizedNN& nn) {
        os << "Neural Network Summary:" << std::endl;

        os << "Input Features:" << std::endl;
        for (int i = 0; i < 16; i++) {
            os << std::setw(5) << int(nn.inputFeatures[i]) << " ";
        }
        os << std::endl;

        os << "Input Bias:" << std::endl;
        for (int i = 0; i < 16; i++) {
            os << std::setw(5) << int(nn.inputBias[i]) << " ";
        }
        os << std::endl;

        os << "Hidden Features:" << std::endl;
        for (int i = 0; i < 16; i++) {
            os << std::setw(5) << int(nn.hiddenFeatures[i]) << " ";
        }
        os << std::endl;

        os << "Hidden Bias:";
        os << std::setw(5) << nn.hiddenBias[0];
        os << std::endl;

        return os;
    }
};
