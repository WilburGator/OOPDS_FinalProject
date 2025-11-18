#ifndef NETWORKTRAINER_HPP
#define NETWORKTRAINER_HPP

#include "Network.hpp"
#include <vector>
#include <cmath>

/**
 * Basic trainer for small feed-forward networks (XOR, etc.)
 * Only supports sigmoid activation.
 */
class NetworkTrainer {
private:
    double lr;

public:
    NetworkTrainer(double learningRate = 0.1)
        : lr(learningRate) {}

    double dSigmoid(double y) {
        return y * (1.0 - y);
    }

    void train(Network& net,
               const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs)
    {
        for (int e = 0; e < epochs; e++) {

            for (size_t t = 0; t < inputs.size(); t++) {

                // FORWARD
                std::vector<std::vector<double>> activations;
                activations.push_back(inputs[t]);

                for (auto& layer : net.getLayers()) {
                    activations.push_back(layer.forward(activations.back()));
                }

                size_t L = net.depth();
                std::vector<std::vector<double>> deltas(L + 1);

                // DELTA OUTPUT LAYER
                deltas[L].resize(activations[L].size());
                for (size_t i = 0; i < deltas[L].size(); i++) {
                    double y = activations[L][i];
                    double error = targets[t][i] - y;
                    deltas[L][i] = error * dSigmoid(y);
                }

                // DELTA HIDDEN (ONLY XOR SMALL NETWORKS)
                for (int l = L - 1; l > 0; l--) {
                    deltas[l].resize(activations[l].size());

                    for (size_t i = 0; i < deltas[l].size(); i++) {
                        double sum = 0.0;

                        for (size_t j = 0; j < deltas[l + 1].size(); j++) {
                            double w = net.getLayers()[l].getNeurons()[j].getWeights()[i];
                            sum += deltas[l + 1][j] * w;
                        }

                        double y = activations[l][i];
                        deltas[l][i] = sum * dSigmoid(y);
                    }
                }

                // UPDATE WEIGHTS
                for (size_t l = 1; l <= L; l++) {
                    auto& layer = net.getLayers()[l - 1];
                    auto& neurons = layer.getNeurons();

                    for (size_t n = 0; n < neurons.size(); n++) {
                        auto w = neurons[n].getWeights();

                        for (size_t k = 0; k < w.size(); k++) {
                            w[k] += lr * deltas[l][n] * activations[l - 1][k];
                        }

                        neurons[n].setWeights(w);
                        neurons[n].setBias(neurons[n].getBias() + lr * deltas[l][n]);
                    }
                }
            }
        }
    }
};

#endif
