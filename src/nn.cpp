#include "nn.h"
#include "types.h"
#include "quantize.h"
#include "dataloader.h"
#include <memory>
#include <fstream>
#include <iostream>
#include <omp.h>

float errorGradient(float output, float eval, float wdl) {
    float expected = EVAL_CP_RATIO * sigmoid(eval) + (1 - EVAL_CP_RATIO) * wdl;
    return 2 * (sigmoid(output) - expected);
}

void NN::testFen(const std::string& fen) const {
    chess::Position pos{chess::Position::fromFen(fen)};

    Features features;

    chess::Bitboard        pieces = pos.piecesBB();

    const chess::Square ksq_White = pos.kingSquare(chess::Color::White);
    const chess::Square ksq_Black = pos.kingSquare(chess::Color::Black);

    for (chess::Square sq : pieces) {
        const chess::Piece piece      = pos.pieceAt(sq);
        const std::uint8_t pieceType  = static_cast<uint8_t>(piece.type());
        const std::uint8_t pieceColor = static_cast<uint8_t>(piece.color());

        const int featureW = inputIndex(pieceType, pieceColor, static_cast<int>(sq), static_cast<uint8_t>(chess::Color::White), static_cast<int>(ksq_White));
        const int featureB = inputIndex(pieceType, pieceColor, static_cast<int>(sq), static_cast<uint8_t>(chess::Color::Black), static_cast<int>(ksq_Black));

        features.add(featureW, featureB);
    }

    Accumulator accumulator;
    Accumulator activated;
    Color       stm = Color(pos.sideToMove());
    
    std::cout << "Score: " << forward(accumulator, activated, features, stm) << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << std::setw(5) << accumulator[i] << " ";
    }
    std::cout << std::endl;
}

// The forward pass of the network
float NN::forward(Accumulator& accumulator, Accumulator& activated, const Features& features, Color stm) const {
    float output = hiddenBias[features.output_bucket]; // Initialize with the bias

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
    
    #pragma omp simd reduction(+:output)
    for (int i = 0; i < 2 * HIDDEN_SIZE; ++i){
        activated[i] = SCReLU(accumulator[i]);
    }

    #pragma omp simd reduction(+:output)
    for (int i = 0; i < HIDDEN_SIZE * 2; ++i){
        output += hiddenFeatures[features.output_bucket * HIDDEN_SIZE * 2 + i] * activated[i];
    }
    
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