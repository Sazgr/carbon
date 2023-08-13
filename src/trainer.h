#pragma once

#include "dataloader.h"
#include "lrscheduler.h"
#include "misc.h"
#include "nn.h"
#include "types.h"
#include "optimizer.h"
#include <filesystem>
#include <vector>

class Trainer {
private:
    std::size_t epochSize = 1e7;
    std::string path;

    std::string savePath;
    std::string networkId;
    int         maxEpochs    = 0;
    float       learningRate = 0.01;

    int   lrDecayInterval = 15;
    float lrDecay         = 0.5;
    int   saveInterval    = 1;

public:
    DataLoader::DataSetLoader        dataSetLoader;
    NN                               nn;
    NNGradients                      nnGradients;
    std::vector<BatchGradients>      batchGradients;
    std::vector<float>               losses;
    LearningRateScheduler::StepDecay lrScheduler;
    Optimizer::Adam                  optimizer;

    Trainer(const std::string& _path, const std::size_t _batchSize) : dataSetLoader{_path, _batchSize}, path(_path), lrScheduler(learningRate, lrDecay, lrDecayInterval), optimizer() {
        batchGradients.resize(THREADS);
        losses.resize(THREADS);
        nnGradients.clear();
    }

    void loadFeatures(DataLoader::DataSetEntry& entry, Features& features);
    void clearGradientsAndLosses();
    void train();
    void batch();
    void applyGradients();

    std::size_t getBatchSize() const {
        return dataSetLoader.batchSize;
    }

    void setNetworkId(const std::string& _networkId) {
        if (_networkId.empty()) {
            std::string randomHexValue = Misc::generateRandomHexValue(5);
            networkId                  = "net_" + randomHexValue;
        } else {
            networkId = _networkId;
            // Replace $ with a random number in the network ID
            if (networkId.find("$") != std::string::npos) {
                std::string randomHexValue = Misc::generateRandomHexValue(5);
                networkId.replace(networkId.find("$"), 1, randomHexValue);
            }
        }
    }
    void setSavePath(const std::string& _savePath) {
        savePath = _savePath + "/" + networkId;
        std::filesystem::create_directories(savePath);
        std::filesystem::create_directories(savePath + "/checkpoints/");
        std::filesystem::create_directories(savePath + "/quantized/");
    }

    void save(const std::string& epoch = "") {
        if (epoch.empty()) {
            nn.save(savePath + "/checkpoints/" + networkId + ".ckpt");
            nn.quantize(savePath + "/quantized/" + networkId + ".nn", true);
            return;
        }
        nn.save(savePath + "/checkpoints/" + networkId + "_ep" + epoch + ".ckpt");
        nn.quantize(savePath + "/quantized/" + networkId + "_ep" + epoch + ".nn");
    }

    void setMaxEpochs(const int _maxEpochs) {
        maxEpochs = _maxEpochs;
    }
    void setEpochSize(const int _epochSize) {
        epochSize = _epochSize;
    }

    auto getNetworkId() const {
        return networkId;
    }
    auto getSavePath() const {
        return savePath;
    }

    void loadCheckpoint(const std::string& _checkpointPath) {
        nn.load(_checkpointPath);
    }
    void saveCheckpoint(const std::string& _checkpointPath) {
        nn.save(_checkpointPath);
    }

    void setLearningRate(const float _learningRate) {
        learningRate = _learningRate;
        lrScheduler.initial_learning_rate = learningRate;
    }

    auto getLearningRate() const {
        return learningRate;
    }

    auto getSaveInterval() const {
        return saveInterval;
    }

    auto getMaxEpochs() const {
        return maxEpochs;
    }

    void setSaveInterval(const int _saveInterval) {
        saveInterval = _saveInterval;
    }
};