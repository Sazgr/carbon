#include "trainer.h"
#include "nn.h"
#include "optimizer.h"
#include <functional>

#define EPOCH_ERROR epochError / static_cast<double>(dataSetLoader.batchSize * batchIterations)

inline float expectedEval(float eval, float wdl, float lambda) {
    return lambda * sigmoid(eval) + (1 - lambda) * wdl;
}

inline float errorGradient(float output, float eval, float wdl) {
    float expected = EVAL_CP_RATIO * sigmoid(eval) + (1 - EVAL_CP_RATIO) * wdl;
    return 2 * (sigmoid(output) - expected);
}

inline float errorFunction(float output, float expected) {
    return pow(sigmoid(output) - expected, 2);
}

inline float errorGradient(float output, float expected) {
    return 2 * (sigmoid(output) - expected);
}

void Trainer::batch_thread(int thread_id, std::array<uint8_t, INPUT_SIZE>& active, std::array<std::array<uint8_t, INPUT_SIZE>, THREADS> actives) {
    for (int batchIdx = thread_id; batchIdx < dataSetLoader.batchSize; batchIdx += THREADS) {
        const int threadId = thread_id;

        // Load the current batch entry
        DataLoader::DataSetEntry& entry = dataSetLoader.getEntry(batchIdx);

        alignas(32) NN::Accumulator accumulator;
        NN::Color                   stm = NN::Color(entry.sideToMove());
        Features                    featureset;

        entry.loadFeatures(featureset);

        const float eval     = entry.score();
        const float wdl      = entry.wdl();
        const float lambda   = getLambda();
        const float expected = expectedEval(eval, wdl, lambda);

        //--- Forward Pass ---//
        const float output = nn.forward(accumulator, featureset, stm);

        losses[threadId] += errorFunction(output, expected);

        //--- Backward Pass ---//
        BatchGradients& gradients   = batchGradients[threadId];
        const float     outGradient = errorGradient(output, expected) * sigmoidPrime(output);

        // Hidden bias
        gradients.hiddenBias[0] += outGradient;

        // Hidden features
//simd later
        for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
            gradients.hiddenFeatures[i] += outGradient * accumulator[i];
        }

        std::array<float, HIDDEN_SIZE * 2> hiddenLosses;

//simd later
        for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
            hiddenLosses[i] = outGradient * nn.hiddenFeatures[i] * ReLUPrime(accumulator[i]);
        }

        // Input bias
//simd later
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            gradients.inputBias[i] += hiddenLosses[i] + hiddenLosses[i + HIDDEN_SIZE];
        }

        // Input features
        for (int i = 0; i < featureset.n; ++i) {
            int f1 = featureset.features[i][stm];
            int f2 = featureset.features[i][!stm];

            actives[threadId][f1] = 1;
            actives[threadId][f2] = 1;

//simd later
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                gradients.inputFeatures[f1 * HIDDEN_SIZE + j] += hiddenLosses[j];
                gradients.inputFeatures[f2 * HIDDEN_SIZE + j] += hiddenLosses[j + HIDDEN_SIZE];
            }
        }
    }
}

void Trainer::batch(std::array<uint8_t, INPUT_SIZE>& active) {
    std::array<std::array<uint8_t, INPUT_SIZE>, THREADS> actives;
    std::memset(actives.data(), 0, sizeof(actives));

    std::vector<std::thread> thread_pool;
    auto batch_thread_fun = std::bind(&Trainer::batch_thread, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    for (int thread_id = 0; thread_id < THREADS; ++thread_id) {
        thread_pool.emplace_back(batch_thread_fun, thread_id, std::ref(active), std::ref(actives));
    }
    for (auto& thread : thread_pool) {
        thread.join();
    }
//too lazy to parallelize
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < THREADS; ++j) {
            active[i] |= actives[j][i];
        }
    }
}

void        Trainer::applyGradients_thread(int thread_id, std::array<uint8_t, INPUT_SIZE>& actives) {
    for (int i = thread_id; i < INPUT_SIZE; i += THREADS) {
        if (!actives[i])
            continue;

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            int   index       = i * HIDDEN_SIZE + j;
            float gradientSum = 0;

            for (int k = 0; k < THREADS; ++k) {
                gradientSum += batchGradients[k].inputFeatures[index];
            }

            optimizer.update(nn.inputFeatures[index], nnGradients.inputFeatures[index], gradientSum, learningRate);
        }
    }
    for (int i = thread_id; i < HIDDEN_SIZE; i += THREADS) {
        float gradientSum = 0;

        for (int j = 0; j < THREADS; ++j) {
            gradientSum += batchGradients[j].inputBias[i];
        }

        optimizer.update(nn.inputBias[i], nnGradients.inputBias[i], gradientSum, learningRate);
    }
    for (int i = thread_id; i < HIDDEN_SIZE * 2; i += THREADS) {
        float gradientSum = 0;

        for (int j = 0; j < THREADS; ++j) {
            gradientSum += batchGradients[j].hiddenFeatures[i];
        }

        optimizer.update(nn.hiddenFeatures[i], nnGradients.hiddenFeatures[i], gradientSum, learningRate);
    }
}

void        Trainer::applyGradients(std::array<uint8_t, INPUT_SIZE>& actives) {
{
    std::vector<std::thread> thread_pool;
    auto applyGradients_thread_fun = std::bind(&Trainer::applyGradients_thread, this, std::placeholders::_1, std::placeholders::_2);
    for (int thread_id = 0; thread_id < THREADS; ++thread_id) {
        thread_pool.emplace_back(applyGradients_thread_fun, thread_id, std::ref(actives));
    }
    for (auto& thread : thread_pool) {
        thread.join();
    }
}
    //-- Hidden Bias --//
    float gradientSum = 0;
    for (int i = 0; i < THREADS; ++i) {
        gradientSum += batchGradients[i].hiddenBias[0];
    }

    optimizer.update(nn.hiddenBias[0], nnGradients.hiddenBias[0], gradientSum, learningRate);
}

void Trainer::train() {
    std::ofstream lossFile(savePath + "/loss.csv", std::ios::app);
    lossFile << "epoch,train_error,val_error,learning_rate" << std::endl;

    const std::size_t batchSize = dataSetLoader.batchSize;

    for (currentEpoch = 1; currentEpoch <= maxEpochs; ++currentEpoch) {
        std::uint64_t start           = Misc::getTimeMs();
        std::size_t   batchIterations = 0;
        double        epochError      = 0.0;

        for (int b = 0; b < EPOCH_SIZE / batchSize; ++b) {
            batchIterations++;
            double batchError = 0;

            // Clear gradients and losses
            clearGradientsAndLosses();

            std::array<uint8_t, INPUT_SIZE> actives;

            // Perform batch operations
            batch(actives);

            // Calculate batch error
            for (int threadId = 0; threadId < THREADS; ++threadId) {
                batchError += static_cast<double>(losses[threadId]);
            }

            // Accumulate epoch error
            epochError += batchError;

            // Gradient descent
            applyGradients(actives);

            // Load the next batch
            dataSetLoader.loadNextBatch();

            // Print progress
            if (b % 100 == 0 || b == EPOCH_SIZE / batchSize - 1) {
                std::uint64_t end            = Misc::getTimeMs();
                int           positionsCount = (b + 1) * batchSize;
                int           posPerSec      = static_cast<int>(positionsCount / ((end - start) / 1000.0));
                printf("\rep/ba:[%4d/%4d] |batch error:[%1.9f]|epoch error:[%1.9f]|speed:[%9d] pos/s", currentEpoch, b, batchError / static_cast<double>(dataSetLoader.batchSize), EPOCH_ERROR, posPerSec);
                std::cout << std::flush;
            }
        }

        // Save the network
        if (currentEpoch % saveInterval == 0) {
            save(std::to_string(currentEpoch));
        }

        // Decay learning rate
        lrScheduler.step(learningRate);

        double valError = validate();
        std::cout << std::endl;
        printf("epoch: [%5d/%5d] | val error: [%11.9f] | epoch error: [%11.9f]", currentEpoch, maxEpochs, valError, EPOCH_ERROR);
        std::cout << std::endl;

        // Save the loss
        lossFile << currentEpoch << "," << EPOCH_ERROR << "," << valError << "," << learningRate << std::endl;
    }
}

void Trainer::clearGradientsAndLosses() {
    for (auto& grad : batchGradients) {
        grad.clear();
    }
    memset(losses.data(), 0, sizeof(float) * THREADS);
}

void        Trainer::validationBatch_thread(int thread_id, std::vector<float>& validationLosses) {
    for (int batchIdx = thread_id; batchIdx < valDataSetLoader.batchSize; batchIdx += THREADS) {
        const int threadId = thread_id;

        // Load the current batch entry
        DataLoader::DataSetEntry& entry = valDataSetLoader.getEntry(batchIdx);

        alignas(32) NN::Accumulator accumulator;
        NN::Color                   stm = NN::Color(entry.sideToMove());
        Features                    featureset;

        entry.loadFeatures(featureset);

        const auto eval = entry.score();
        const auto wdl  = entry.wdl();

        //--- Forward Pass ---//
        const float output = nn.forward(accumulator, featureset, stm);

        validationLosses[threadId] += errorFunction(output, eval, wdl);
    }
}

void        Trainer::validationBatch(std::vector<float>& validationLosses) {
    std::vector<std::thread> thread_pool;
    auto validationBatch_thread_fun = std::bind(&Trainer::validationBatch_thread, this, std::placeholders::_1, std::placeholders::_2);
    for (int thread_id = 0; thread_id < THREADS; ++thread_id) {
        thread_pool.emplace_back(validationBatch_thread_fun, thread_id, std::ref(validationLosses));
    }
    for (auto& thread : thread_pool) {
        thread.join();
    }
}

double Trainer::validate() {
    std::size_t batchIterations = 0;
    double      epochError      = 0.0;

    for (int b = 0; b < VAL_EPOCH_SIZE / valDataSetLoader.batchSize; ++b) {
        batchIterations++;
        double batchError = 0;

        std::vector<float> validationLosses;
        validationLosses.resize(THREADS);
        std::memset(validationLosses.data(), 0, sizeof(validationLosses));

        validationBatch(validationLosses);

        // Calculate batch error
        for (int threadId = 0; threadId < THREADS; ++threadId) {
            batchError += static_cast<double>(validationLosses[threadId]);
        }

        // Accumulate epoch error
        epochError += batchError;

        // Load the next batch
        valDataSetLoader.loadNextBatch();
    }

    return epochError / static_cast<double>(valDataSetLoader.batchSize * batchIterations);
}