#include "optimizer.h"
#include "types.h"
#include <algorithm>

namespace Optimizer {
    void Adam::update(float& v, Gradient& grad, const float gsum, const float learningRate) {
        grad.M = beta1 * grad.M + (1 - beta1) * gsum;
        grad.V = beta2 * grad.V + (1 - beta2) * gsum * gsum;

        v -= learningRate * grad.M / (sqrt(grad.V) + epsilon);
    }

    void AdamW::update(float& v, Gradient& grad, const float gsum, const float learningRate) {
        const float decay = 1.0 - 0.01 * learningRate;
        v *= decay;
        grad.M = beta1 * grad.M + (1 - beta1) * gsum;
        grad.V = beta2 * grad.V + (1 - beta2) * gsum * gsum;

        v -= learningRate * grad.M / (sqrt(grad.V) + epsilon);
    }

    void Adamax::update(float& v, Gradient& grad, const float gsum, const float learningRate) {
        grad.M = beta1 * grad.M + (1 - beta1) * gsum;
        grad.V = std::max(beta2 * grad.V, std::abs(gsum));

        v -= learningRate * grad.M / (grad.V + EPSILON);
    }
} // namespace Optimizer