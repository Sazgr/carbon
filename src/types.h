#pragma once

#include <cstdint>
#include <array>
#include <random>
#include <cstring>
#include <sstream>
#include <chrono>
#include <iomanip>

constexpr int BUCKETS = 4;
constexpr int INPUT_SIZE = 64 * 6 * 2 * BUCKETS;
constexpr int HIDDEN_SIZE = 512;
constexpr int OUTPUT_SIZE = 1;

constexpr float EVAL_SCALE = 400.0f;
constexpr float EVAL_CP_RATIO = 0.7f;

constexpr int THREADS = 8;

constexpr std::size_t EPOCH_SIZE = 1e8;
constexpr std::size_t VAL_EPOCH_SIZE = 1e7;

constexpr int KING_BUCKET[64] {
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 3, 3, 3, 3, 2, 2,
    2, 2, 3, 3, 3, 3, 2, 2,
    2, 2, 3, 3, 3, 3, 2, 2,
    2, 2, 3, 3, 3, 3, 2, 2,
};

static inline int kingSquareIndex(int kingSquare, uint8_t kingColor) {
    if constexpr(BUCKETS > 1){
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