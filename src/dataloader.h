#pragma once

#include "types.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <array>
#include <thread>

constexpr std::size_t CHUNK_SIZE = (1 << 20);

struct Features;

namespace DataLoader {

    struct DataSetEntry {
        std::array<std::uint64_t, 13> entry_bb{};
        std::array<int, 64> entry_pos{12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12};
        std::uint64_t stm;
        float entry_score;
        float entry_result;

        const float score() const {
            return entry_score / EVAL_SCALE;
        }

        const float wdl() const {
            return entry_result == -1 ? 1.0 : entry_result == 0 ? 0.5 : 0.0;
        }

        const float target() const {
            float p_target = 1 / (1 + expf(-score()));
            float w_target = wdl();

            return p_target * EVAL_CP_RATIO + w_target * (1 - EVAL_CP_RATIO);
        }

        const auto sideToMove() const {
            return stm;
        }

        void loadFeatures(Features& features) const;
        Features loadFeatures() const;
    };

    struct DataSetLoader {
        std::array<DataSetEntry, CHUNK_SIZE> currentData;
        std::array<DataSetEntry, CHUNK_SIZE> nextData;
        std::array<int, CHUNK_SIZE>            permuteShuffle;

        std::ifstream reader;
        std::string                                path;
        std::size_t batchSize     = 16384;
        std::size_t positionIndex = 0;

        std::thread readingThread;

        bool backgroundLoading = true;

        DataSetLoader(const std::string& _path) : reader{_path}, path{_path} {
            init();
        }

        DataSetLoader(const std::string& _path, const std::size_t _batchSize) : reader{_path}, path{_path}, batchSize{_batchSize} {
            init();
        }

        DataSetLoader(const std::string& _path, const std::size_t _batchSize, const bool _backgroundLoading) : reader{_path}, path{_path}, batchSize{_batchSize}, backgroundLoading{_backgroundLoading} {
            init();
        }

        void          loadNext();
        void          loadNextBatch();
        void          init();
        void          shuffle();
        DataSetEntry& getEntry(const int index) {
            return currentData[positionIndex + index];
        }

        friend std::ostream& operator<<(std::ostream& os, const DataSetLoader& data_set_loader) {
            os << "DataSetLoader(batchSize=" << data_set_loader.batchSize << ", positionIndex=" << data_set_loader.positionIndex << ")";
            return os;
        }
    };

} // namespace DataLoader