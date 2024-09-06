#include "dataloader.h"
#include "nn.h"
#include <ctime>

namespace DataLoader {

    static constexpr int VALUE_NONE = 32002;
    void                 DataSetLoader::loadFromBuffer() {
        
        m_currentDataSize = m_buffer.size();

        // Permute shuffle
        shuffle();

#pragma omp parallel for schedule(static) num_threads(THREADS)
        // Convert entries into features
        for (std::size_t i = 0; i < m_buffer.size(); ++i) {
            const auto& entry = m_buffer[i];
            m_currentData[m_permuteShuffle[i]].loadEntry(entry);
        }
    }

    void DataSetLoader::loadNextBatch() {
        m_positionIndex = std::min(m_positionIndex + m_batchSize, m_currentDataSize);

        if (m_positionIndex == m_currentDataSize) {
            if (m_readingThread.joinable()) {
                m_readingThread.join();
            }

            m_positionIndex = 0;
            loadFromBuffer();

            if (m_backgroundLoading) {
                m_readingThread = std::thread(&DataSetLoader::loadNext, this);
            } else {
                loadNext();
            }
        }
    }

    void DataSetLoader::tryFillBuffer() {
        std::random_device          rd;
        std::mt19937                mt{rd()};
        double                      prob = static_cast<double>(m_random_fen_skipping) / (m_random_fen_skipping + 1);
        std::bernoulli_distribution dist(prob);

        for (std::size_t counter = 0; counter < CHUNK_SIZE; ++counter) {
            // If we finished, go back to the beginning
            if (!m_reader.hasNext()) {
                m_reader = binpack::CompressedTrainingDataEntryReader(m_path);
            }

            // Get info
            binpack::TrainingDataEntry entry = m_reader.next();

            auto                 do_wld_skip = [&]() {
                std::bernoulli_distribution distrib(1.0 - entry.score_result_prob() * entry.score_result_prob());
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return entry.isInCheck() || (entry.isCapturingMove() && (entry.score == 0 || entry.seeGE(0)));
            };

            // Allow for predetermined filtering without the need to remove positions from the dataset.
            if (entry.score == VALUE_NONE)
                continue;

            constexpr int early_fen_skipping = 16;
            if (entry.ply <= early_fen_skipping) {
                continue;
            }

            if (m_random_fen_skipping && do_skip()) {
                continue;
            }

            bool filtered = true;
            if (filtered && do_filter())
                continue;

            bool wld_filtered = true;
            if (wld_filtered && do_wld_skip())
                continue;

            m_buffer.push_back(entry);
        }
    }

    void DataSetLoader::loadNext() {
        m_buffer.clear();

        while(m_buffer.size() < m_batchSize){
            tryFillBuffer();
        }
    }

    void DataSetLoader::init() {
        m_positionIndex = 0;

        loadNext();
        loadFromBuffer();
        loadNext();
    }

    void loadFeatures(const binpack::TrainingDataEntry& entry, Features& features) {
        features.clear();
        features.stm = static_cast<uint8_t>(entry.pos.sideToMove());
        
        const chess::Position& pos    = entry.pos;
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
    }

} // namespace DataLoader