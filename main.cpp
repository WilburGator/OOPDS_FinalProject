#include <iostream>
#include <vector>
#include "perceptron.hpp"
#include "PerceptronTrainer.hpp"
#include "Layer.hpp"
#include "Network.hpp"
#include "NetworkTrainer.hpp"

using namespace std;

int main() {

    // ============================
    // PERCEPTRON TRAINING (AND / OR)
    // ============================
    cout << "=== Perceptron Training Tests ===" << endl;

    // Data for AND gate
    vector<vector<double>> and_inputs = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };
    vector<double> and_targets = {0,0,0,1};

    Perceptron and_p({0.0, 0.0}, 0.0, STEP); 
    PerceptronTrainer pt(0.1);

    cout << "\nTraining perceptron for AND gate..." << endl;
    pt.train(and_p, and_inputs, and_targets, 20);

    cout << "Results (AND):" << endl;
    for (auto &inp : and_inputs) {
        double out = and_p.forward(inp);
        cout << inp[0] << " AND " << inp[1] << " = " << out << endl;
    }

    // OR gate
    vector<double> or_targets = {0,1,1,1};
    Perceptron or_p({0.0, 0.0}, 0.0, STEP);

    cout << "\nTraining perceptron for OR gate..." << endl;
    pt.train(or_p, and_inputs, or_targets, 20);

    cout << "Results (OR):" << endl;
    for (auto &inp : and_inputs) {
        double out = or_p.forward(inp);
        cout << inp[0] << " OR " << inp[1] << " = " << out << endl;
    }


    // ============================
    // XOR TRAINING (NETWORK)
    // ============================

    cout << "\n=== Neural Network XOR Test ===" << endl;

    // XOR dataset
    vector<vector<double>> xor_inputs = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };
    vector<vector<double>> xor_targets = {
        {0}, {1}, {1}, {0}
    };

    // Hidden layer (2 neurons)
    Layer hidden({
        Perceptron({0.5, -0.4}, 0.0, SIGMOID),
        Perceptron({-0.3, 0.8}, 0.0, SIGMOID)
    });

    // Output layer (1 neuron)
    Layer out({
        Perceptron({0.7, -0.1}, 0.0, SIGMOID)
    });

    Network net("xor_net");
    net.addLayer(hidden);
    net.addLayer(out);

    // Trainer
    NetworkTrainer nt(0.5);

    cout << "\nTraining XOR network (this may take a moment)..." << endl;
    nt.train(net, xor_inputs, xor_targets, 3000);

    cout << "Results (XOR):" << endl;
    for (auto &inp : xor_inputs) {
        double res = net.forward(inp)[0];
        cout << inp[0] << " XOR " << inp[1] << " = " << res << endl;
    }

    cout << "\nDone." << endl;

    return 0;
}
