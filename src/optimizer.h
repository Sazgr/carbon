#pragma once

#include "gradient.h"

namespace Optimizer {
    constexpr float BETA1   = 0.9f;
    constexpr float BETA2   = 0.999f;
    constexpr float EPSILON = 1e-8f;

    class Optimizer {
    public:
        int   steps        = 0;

        void step(){
            ++steps;
        }
        void reset(){
            steps = 0;
        }
    };

    class Adam : public Optimizer {
    public:
        static constexpr float DEFAULT_BETA1         = BETA1;
        static constexpr float DEFAULT_BETA2         = BETA2;
        static constexpr float DEFAULT_EPSILON       = EPSILON;

        float beta1   = DEFAULT_BETA1;
        float beta2   = DEFAULT_BETA2;
        float epsilon = DEFAULT_EPSILON;

        Adam(float _beta1 = DEFAULT_BETA1, float _beta2 = DEFAULT_BETA2, float _epsilon = DEFAULT_EPSILON) : beta1(_beta1), beta2(_beta2), epsilon(_epsilon) {
            
        }

        void update(float& v, Gradient& grad, const float gsum, const float learningRate);

        friend std::ostream& operator<<(std::ostream& os, const Adam& adam) {
            os << "Adam(" << "beta1=" << adam.beta1 << ", beta2=" << adam.beta2 << ", epsilon=" << adam.epsilon << ")";
            return os;
        }
    };

    class AdamW : public Optimizer {
    public:
        static constexpr float DEFAULT_BETA1         = BETA1;
        static constexpr float DEFAULT_BETA2         = BETA2;
        static constexpr float DEFAULT_EPSILON       = EPSILON;

        float beta1   = DEFAULT_BETA1;
        float beta2   = DEFAULT_BETA2;
        float epsilon = DEFAULT_EPSILON;

        AdamW(float _beta1 = DEFAULT_BETA1, float _beta2 = DEFAULT_BETA2, float _epsilon = DEFAULT_EPSILON) : beta1(_beta1), beta2(_beta2), epsilon(_epsilon) {
            
        }

        void update(float& v, Gradient& grad, const float gsum, const float learningRate);

        friend std::ostream& operator<<(std::ostream& os, const AdamW& adamw) {
            os << "AdamW(" << "beta1=" << adamw.beta1 << ", beta2=" << adamw.beta2 << ", epsilon=" << adamw.epsilon << ")";
            return os;
        }
    };

    class Adamax : public Optimizer {
    public:
        static constexpr float DEFAULT_BETA1         = BETA1;
        static constexpr float DEFAULT_BETA2         = BETA2;
        static constexpr float DEFAULT_EPSILON       = EPSILON;

        float beta1   = DEFAULT_BETA1;
        float beta2   = DEFAULT_BETA2;
        float epsilon = DEFAULT_EPSILON;

        Adamax(float _beta1 = DEFAULT_BETA1, float _beta2 = DEFAULT_BETA2, float _epsilon = DEFAULT_EPSILON) : beta1(_beta1), beta2(_beta2), epsilon(_epsilon) {
            
        }

        void update(float& v, Gradient& grad, const float gsum, const float learningRate);

        friend std::ostream& operator<<(std::ostream& os, const Adamax& adamax) {
            os << "Adamax(" << "beta1=" << adamax.beta1 << ", beta2=" << adamax.beta2 << ", epsilon=" << adamax.epsilon << ")";
            return os;
        }
    };
} // namespace Optimizer