#include "quantize.h"
#include "dataloader.h"
#include "nn.h"

const int32_t QuantizedNN::forward(Accumulator& accumulator, Features& features, Color stm) const{
    int32_t output = hiddenBias[0]; // Initialize with the bias

    auto* stmAccumulator = accumulator.data();
    auto* nstmAccumulator = accumulator.data() + HIDDEN_SIZE;

    std::memcpy(stmAccumulator, inputBias.data(), sizeof(float) * HIDDEN_SIZE);
    std::memcpy(nstmAccumulator, inputBias.data(), sizeof(float) * HIDDEN_SIZE);

    for (int i = 0; i < features.n; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            stmAccumulator[j] += inputFeatures[features.features[i][stm] * HIDDEN_SIZE + j];
            nstmAccumulator[j] += inputFeatures[features.features[i][!stm] * HIDDEN_SIZE + j];
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE * 2; ++i){
        accumulator[i] = ReLU(accumulator[i]);
    }

    #pragma omp simd reduction(+:output)
    for (int i = 0; i < HIDDEN_SIZE; ++i){
        output += hiddenFeatures[i] * stmAccumulator[i];
    }

    #pragma omp simd reduction(+:output)
    for (int i = 0; i < HIDDEN_SIZE; ++i){
        output += hiddenFeatures[HIDDEN_SIZE + i] * nstmAccumulator[i];
    }
    
    return output / (Q1 * Q2);
}