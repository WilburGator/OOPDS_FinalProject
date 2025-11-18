#ifndef PERCEPTRONTRAINER_HPP
#define PERCEPTRONTRAINER_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include "perceptron.hpp"

/**
 * @brief Simple perceptron training using the classic Delta Rule.
 * Works for linearly separable problems (AND, OR, NAND, etc.)
 */
class PerceptronTrainer {
private:
    double lr; // learning rate

public:
    PerceptronTrainer(double learningRate = 0.1)
        : lr(learningRate) {}

    /**
     * @brief Train perceptron for a number of epochs.
     */
    void train(Perceptron& p,
               const std::vector<std::vector<double>>& inputs,
               const std::vector<double>& targets,
               int epochs)
    {
        if (inputs.size() != targets.size())
            throw std::invalid_argument("Size mismatch in training data.");

        for (int e = 0; e < epochs; e++) {
            for (size_t i = 0; i < inputs.size(); i++) {

                double out = p.forward(inputs[i]);
                double error = targets[i] - out;

                // Update weights + bias
                std::vector<double> w = p.getWeights();
                for (size_t j = 0; j < w.size(); j++) {
                    w[j] += lr * error * inputs[i][j];
                }
                p.setWeights(w);
                p.setBias(p.getBias() + lr * error);
            }
        }
    }
};

#endif
