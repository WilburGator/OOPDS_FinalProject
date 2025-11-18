#include "perceptron.hpp"
#include <stdexcept>
#include <sstream>

Perceptron::Perceptron(const std::vector<double>& w,
                       double b,
                       ActivationType act)
    : weights(w), bias(b), activation(act) {}

void Perceptron::setWeights(const std::vector<double>& w) {
    weights = w;
}

void Perceptron::setBias(double b) {
    bias = b;
}

void Perceptron::setActivation(ActivationType act) {
    activation = act;
}

double Perceptron::forward(const std::vector<double>& inputs) const {
    if (inputs.size() != weights.size()) {
        throw std::invalid_argument("Input size does not match weight size.");
    }

    double sum = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i] * weights[i];
    }

    return activate(sum, activation);
}

std::string Perceptron::summary() const {
    std::ostringstream oss;
    oss << "Perceptron(";
    oss << "weights=[";
    for (size_t i = 0; i < weights.size(); ++i) {
        oss << weights[i];
        if (i + 1 < weights.size()) oss << ", ";
    }
    oss << "], bias=" << bias << ", activation=" << activation << ")";
    return oss.str();
}
