#include "dnf.h"

float DNF::filter(const float signal, const float noise) {
    signal_delayLine.push_back(signal);
    const float delayed_signal = signal_delayLine[0];
    
    for (int i = noiseDelayLineLength-1 ; i > 0; i--) {
	noise_delayLine[i] = noise_delayLine[i-1];
    }
    noise_delayLine[0] = noise;

    // NOISE INPUT TO NETWORK
    torch::Tensor data = torch::from_blob(
        noise_delayLine, {noiseDelayLineLength}, torch::kFloat);

    // REMOVER OUTPUT FROM NETWORK
    torch::Tensor output;

    output = model->forward(data);
    
    auto a = output.accessor<float,1>();
    remover = a[0];
    
    f_nn = delayed_signal - remover;
    
    // FEEDBACK TO THE NETWORK
    optimizer->zero_grad();
    torch::Tensor gradient = torch::tensor({f_nn});
    //output.retain_grad();
    output.backward(gradient);
    optimizer->step();

    return f_nn;
}
