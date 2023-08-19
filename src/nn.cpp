#include "nn.h"
#include "types.h"
#include "quantize.h"
#include "dataloader.h"
#include <memory>
#include <fstream>
#include <iostream>

float errorGradient(float output, float eval, float wdl) {
    float expected = EVAL_CP_RATIO * sigmoid(eval) + (1 - EVAL_CP_RATIO) * wdl;
    return 2 * (sigmoid(output) - expected);
}

// The forward pass of the network
float NN::forward(Accumulator& accumulator, Features& features, Color stm) const {
    float output = hiddenBias[0]; // Initialize with the bias

    float* stmAccumulator = accumulator.data();
    float* nstmAccumulator = accumulator.data() + HIDDEN_SIZE;

    std::memcpy(stmAccumulator, inputBias.data(), sizeof(float) * HIDDEN_SIZE);
    std::memcpy(nstmAccumulator, inputBias.data(), sizeof(float) * HIDDEN_SIZE);

    for (int i = 0; i < features.n; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            stmAccumulator[j] += inputFeatures[features.features[i][stm] * HIDDEN_SIZE + j];
            nstmAccumulator[j] += inputFeatures[features.features[i][!stm] * HIDDEN_SIZE + j];
        }
    }
    
    vecReLU<HIDDEN_SIZE * 2>(accumulator.data());

    //simd later reduction(+:output)
    output += vecDotProduct<HIDDEN_SIZE>(&hiddenFeatures[0], stmAccumulator);
    output += vecDotProduct<HIDDEN_SIZE>(&hiddenFeatures[HIDDEN_SIZE], nstmAccumulator);
    /*for (int i = 0; i < HIDDEN_SIZE; ++i){
        output += hiddenFeatures[i] * stmAccumulator[i];
    }

    //simd later reduction(+:output)
    for (int i = 0; i < HIDDEN_SIZE; ++i){
        output += hiddenFeatures[HIDDEN_SIZE + i] * nstmAccumulator[i];
    }*/
    
    return output;
}

void NN::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (file) {
        bool sizeMismatch = false;

        // Read inputFeatures
        if (!file.read(reinterpret_cast<char*>(inputFeatures.data()), sizeof(inputFeatures)))
            sizeMismatch = true;

        // Read inputBias
        if (!file.read(reinterpret_cast<char*>(inputBias.data()), sizeof(inputBias)))
            sizeMismatch = true;

        // Read hiddenFeatures
        if (!file.read(reinterpret_cast<char*>(hiddenFeatures.data()), sizeof(hiddenFeatures)))
            sizeMismatch = true;

        // Read hiddenBias
        if (!file.read(reinterpret_cast<char*>(hiddenBias.data()), sizeof(hiddenBias)))
            sizeMismatch = true;

        if (sizeMismatch) {
            std::cout << "Error: Checkpoint data size mismatch in " << path << std::endl;
            exit(0); // Exit
        }

        std::cout << "Loaded checkpoint file " << path << std::endl;
    } else {
        std::cout << "Couldn't read checkpoint file " << path << std::endl;
    }
}

void NN::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);

    if (file) {
        file.write(reinterpret_cast<char*>(inputFeatures.data()), sizeof(inputFeatures));
        file.write(reinterpret_cast<char*>(inputBias.data()), sizeof(inputBias));
        file.write(reinterpret_cast<char*>(hiddenFeatures.data()), sizeof(hiddenFeatures));
        file.write(reinterpret_cast<char*>(hiddenBias.data()), sizeof(hiddenBias));
    } else {
        std::cout << "Couldn't write checkpoint file " << path << std::endl;
    }
}

void NN::quantize(const std::string& path, bool print){
    std::unique_ptr<QuantizedNN> qnn = std::make_unique<QuantizedNN>(*this, print);

    qnn->save(path);

    if (print){
        std::cout << "Quantized network saved to " << path << std::endl;
    }
}