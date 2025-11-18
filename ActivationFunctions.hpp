#ifndef ActivationFunctions_H
#define ActivationFunctions_H

#include <cmath>
#include <algorithm>

enum ActivationType { SIGMOID, RELU, TANH, STEP };
namespace ActivationFunctions {
    inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    inline double relu(double x) { return std::max(0.0, x); }
    inline double tanh_act(double x) { return std::tanh(x); }
    inline double step(double x) { return (x >= 0.0) ? 1.0 : 0.0; }
}

inline double activate(double x, ActivationType actType) {
    switch (actType) {
        case SIGMOID: return ActivationFunctions::sigmoid(x);
        case RELU: return ActivationFunctions::relu(x);
        case TANH: return ActivationFunctions::tanh_act(x);
        case STEP: return ActivationFunctions::step(x);
        default: return x; 
    }
}

#endif