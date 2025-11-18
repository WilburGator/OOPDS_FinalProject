#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include "Layer.hpp"

/**
 * @brief Feed-forward neural network with multiple layers.
 */
class Network {
private:
    std::vector<Layer> layers;
    std::string modelName;

public:

    Network(const std::string& name = "NeuralNetwork")
        : modelName(name) {}

    // Constructor for initializer list of layers
    Network(const std::initializer_list<Layer>& ls,
            const std::string& name = "NeuralNetwork")
        : layers(ls), modelName(name)
    {
        if (layers.empty()) {
            throw std::invalid_argument("Network must contain at least one layer.");
        }
    }

    void addLayer(const Layer& layer) {
        layers.push_back(layer);
    }

    std::vector<double> forward(const std::vector<double>& input) const {
        if (layers.empty()) {
            throw std::runtime_error("Network has no layers.");
        }

        std::vector<double> out = input;
        for (const auto& layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }

    size_t depth() const { return layers.size(); }
    std::string getName() const { return modelName; }

    // Needed for training
    const std::vector<Layer>& getLayers() const { return layers; }
    std::vector<Layer>& getLayers() { return layers; }
};

#endif
