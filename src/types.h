#pragma once

#include <cstdint>
#include <array>
#include <random>
#include <cstring>
#include <sstream>
#include <chrono>
#include <iomanip>

constexpr int INPUT_BUCKETS = 8;
constexpr int OUTPUT_BUCKETS = 6;
constexpr int INPUT_SIZE = 64 * 6 * 2 * INPUT_BUCKETS;
constexpr int HIDDEN_SIZE = 512;
constexpr int OUTPUT_SIZE = 1 * OUTPUT_BUCKETS;

constexpr float EVAL_SCALE = 400.0f;
constexpr float EVAL_CP_RATIO = 0.7f;

constexpr int THREADS = 8;

constexpr std::size_t EPOCH_SIZE = 1e8;
constexpr std::size_t VAL_EPOCH_SIZE = 1e7;

constexpr int KING_BUCKET[64] {
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 3, 3, 3, 3, 2, 2,
    4, 4, 5, 5, 5, 5, 4, 4,
    4, 4, 5, 5, 5, 5, 4, 4,
    6, 6, 7, 7, 7, 7, 6, 6,
    6, 6, 7, 7, 7, 7, 6, 6,
    6, 6, 7, 7, 7, 7, 6, 6,
    6, 6, 7, 7, 7, 7, 6, 6,
};

static inline int kingSquareIndex(int kingSquare, uint8_t kingColor) {
    if constexpr(INPUT_BUCKETS > 1){
        kingSquare = (56 * kingColor) ^ kingSquare;
        return KING_BUCKET[kingSquare];
    }else{
        return 0;
    }
}

static inline int inputIndex(uint8_t pieceType, uint8_t pieceColor, int square, uint8_t view, int kingSquare) {
    const int ksIndex = kingSquareIndex(kingSquare, view);
    square = square ^ (56 * view);
    square = square ^ (7 * !!(kingSquare & 0x4));

    // clang-format off
    return square
           + pieceType * 64
           + !(pieceColor ^ view) * 64 * 6 + ksIndex * 64 * 6 * 2;
    // clang-format on
}