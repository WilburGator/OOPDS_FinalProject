#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include "perceptron.hpp"

/**
 * @brief Represents a fully connected layer of perceptrons.
 */
class Layer {
private:
    std::vector<Perceptron> neurons;
    std::string name;

public:

    Layer() = default;

    Layer(const std::vector<Perceptron>& units,
          const std::string& layerName = "")
        : neurons(units), name(layerName)
    {
        if (neurons.empty()) {
            throw std::invalid_argument("Layer must contain at least one perceptron.");
        }
    }

    std::vector<double> forward(const std::vector<double>& input) const {
        std::vector<double> out;
        out.reserve(neurons.size());
        for (const auto& n : neurons) {
            out.push_back(n.forward(input));
        }
        return out;
    }

    size_t size() const { return neurons.size(); }
    std::string getName() const { return name; }

    // Access to neurons (needed for training)
    const std::vector<Perceptron>& getNeurons() const { return neurons; }
    std::vector<Perceptron>& getNeurons() { return neurons; }
};

#endif
