#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>
#include <string>
#include "ActivationFunctions.hpp"

/**
 * @brief Advanced Perceptron implementation used as the basic computational unit
 *        for feed-forward neural networks. Supports multiple activation functions.
 *
 * A perceptron computes:  y = f( w·x + b )
 */
class Perceptron {
private:
    std::vector<double> weights;
    double bias;
    ActivationType activation;

public:

    /**
     * @brief Construct a perceptron with given weights, bias and activation.
     */
    Perceptron(const std::vector<double>& w,
               double b,
               ActivationType act = SIGMOID);

    /**
     * @brief Compute the output of the perceptron for the given input vector.
     * @throws std::invalid_argument if input size != weight size.
     */
    double forward(const std::vector<double>& inputs) const;

    // --- getters ---
    const std::vector<double>& getWeights() const { return weights; }
    double getBias() const { return bias; }
    ActivationType getActivation() const { return activation; }

    // --- setters ---
    void setWeights(const std::vector<double>& w);
    void setBias(double b);
    void setActivation(ActivationType act);

    /**
     * @brief Pretty print for debugging.
     */
    std::string summary() const;
};

#endif
