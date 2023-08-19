#include "dataloader.h"
#include "nn.h"
#include <cassert>
#include <ctime>
#include <iostream>

namespace DataLoader {
    void DataSetLoader::loadNextBatch() {
        positionIndex += batchSize;

        if (positionIndex == CHUNK_SIZE) {
            // Join thread that's reading nextData
            if (readingThread.joinable()) {
                readingThread.join();
            }

            // Bring next data to current position
            std::swap(currentData, nextData);
            positionIndex = 0;

            // Begin a new thread to read nextData if background loading is enabled
            if (backgroundLoading){
                readingThread = std::thread(&DataSetLoader::loadNext, this);
            }
        }
    }

    void DataSetLoader::loadNext() {

        for (std::size_t counter = 0; counter < CHUNK_SIZE; ++counter) {
            std::string board, stm, wdl, score, fullmove, unused;
            reader >> board;
            // If we finished, go back to the beginning
            if (!reader) {
                reader = std::ifstream(path);
                reader >> board;
            }
            reader >> stm >> unused >> unused >> unused >> fullmove >> wdl >> score >> unused;
            DataSetEntry& positionEntry = nextData[permuteShuffle[counter]];
            for (int i{}; i<13; ++i) positionEntry.entry_bb[i] = 0;
            for (int i{}; i<64; ++i) positionEntry.entry_pos[i] = 12;

            // Get info
            int sq = 0;
            for (auto pos = board.begin(); pos != board.end(); ++pos) {
                switch (*pos) {
                    case 'p': (positionEntry.entry_bb[ 0] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  0; break;
                    case 'n': (positionEntry.entry_bb[ 2] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  2; break;
                    case 'b': (positionEntry.entry_bb[ 4] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  4; break;
                    case 'r': (positionEntry.entry_bb[ 6] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  6; break;
                    case 'q': (positionEntry.entry_bb[ 8] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  8; break;
                    case 'k': (positionEntry.entry_bb[10] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] = 10; break;
                    case 'P': (positionEntry.entry_bb[ 1] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  1; break;
                    case 'N': (positionEntry.entry_bb[ 3] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  3; break;
                    case 'B': (positionEntry.entry_bb[ 5] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  5; break;
                    case 'R': (positionEntry.entry_bb[ 7] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  7; break;
                    case 'Q': (positionEntry.entry_bb[ 9] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] =  9; break;
                    case 'K': (positionEntry.entry_bb[11] |= 1ull << sq); (positionEntry.entry_bb[12] |= 1ull << sq); positionEntry.entry_pos[sq] = 11; break;
                    case '/': --sq; break;
                    case '1': break;
                    case '2': ++sq; break;
                    case '3': sq += 2; break;
                    case '4': sq += 3; break;
                    case '5': sq += 4; break;
                    case '6': sq += 5; break;
                    case '7': sq += 6; break;
                    case '8': sq += 7; break;
                    default: break;
                }
                ++sq;
            }
            assert(__builtin_popcountll(positionEntry.entry_bb[12]) <= 32);
            positionEntry.stm = stm[0] == 'w' ? 1 : 0;
            if (wdl == "[0.0]") positionEntry.entry_result = 0.0;
            if (wdl == "[0.5]") positionEntry.entry_result = 0.5;
            if (wdl == "[1.0]") positionEntry.entry_result = 1.0;
            positionEntry.entry_score = stoi(score);
            if (positionEntry.entry_score == 21000 || ((stoi(fullmove) - 1) * 2 + (stm[0] == 'b')) <= 16) {
                counter--;
                continue;
            }
        }

        shuffle();
    }

    void DataSetLoader::shuffle() {
        std::random_device rd;
        std::mt19937 mt{rd()};
        std::shuffle(permuteShuffle.begin(), permuteShuffle.end(), mt);
    }

    void DataSetLoader::init() {
        positionIndex = 0;

        std::iota(permuteShuffle.begin(), permuteShuffle.end(), 0);
        shuffle();

        loadNext();
        std::swap(currentData, nextData);
        loadNext();
    }

    void DataSetEntry::loadFeatures(Features& features) const{
        features.n = 0;
        const std::uint8_t ksq_White = __builtin_ctzll(entry_bb[11]);
        const std::uint8_t ksq_Black = __builtin_ctzll(entry_bb[10]);

        for (std::uint64_t pieces = entry_bb[12]; pieces; pieces &= pieces - 1) {
            assert(__builtin_popcountll(entry_bb[12]) == __builtin_popcountll(pieces) + features.n);
            int sq = __builtin_ctzll(pieces);
            const std::uint8_t piece = entry_pos[sq];

            const int featureW = inputIndex(piece, static_cast<int>(sq), 1, static_cast<int>(ksq_White));
            const int featureB = inputIndex(piece, static_cast<int>(sq), 0, static_cast<int>(ksq_Black));

            features.add(featureW, featureB);
        }
        //if (features.n > 32) for (int i{}; i<64; ++i) std::cout << entry_pos[i] << ' ';
    }

    Features DataSetEntry::loadFeatures() const{
        Features features;
        loadFeatures(features);
        return features;
    }
} // namespace DataLoader