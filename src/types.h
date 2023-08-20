#pragma once

#include <cassert>
#include <cstdint>
#include <array>
#include <random>
#include <cstring>
#include <sstream>
#include <chrono>
#include <iomanip>

constexpr int BUCKETS = 1;
constexpr int INPUT_SIZE = 64 * 6 * 2 * BUCKETS;
constexpr int HIDDEN_SIZE = 256;
constexpr int OUTPUT_SIZE = 1;

constexpr float EVAL_SCALE = 400.0f;
constexpr float EVAL_CP_RATIO = 0.7f;

constexpr int THREADS = 6;

constexpr std::size_t EPOCH_SIZE = 1e9;
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

static inline int inputIndex(uint8_t piece, int square, uint8_t view, int kingSquare) {
    assert(square >= 0 && square < 64);
    assert(piece < 12 && piece >= 0);
    const int ksIndex = kingSquareIndex(kingSquare, view);
    square = square ^ 56 ^ (56 * view);
    square = square ^ (7 * !!(kingSquare & 0x4));

    // clang-format off
    return square + (piece ^ view) * 64 + ksIndex * 64 * 6 * 2;
    // clang-format on
}