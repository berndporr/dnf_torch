#include "dnf.h"

float DNF::filter(const float signal, const float noise) {
    const float delayed_signal = signal_delayLine.process(signal);
    noise_delayLine.process(noise);

    auto noiseTimeSeries = torch::zeros(noiseDelayLineLength,torch::kFloat);
    for(int i = 0; i < noiseDelayLineLength; i++) {
	noiseTimeSeries[i] = noise_delayLine.get(i);
    }

    // REMOVER OUTPUT FROM NETWORK
    torch::Tensor output = model->forward(noiseTimeSeries,actMethod);
    auto a = output.accessor<float,1>();
    remover = a[0];
    
    f_nn = delayed_signal - remover;
    
    // FEEDBACK TO THE NETWORK
    optimizer->zero_grad();
    torch::Tensor gradient = torch::tensor({-f_nn});
    output.retain_grad();
    output.backward(gradient);
    optimizer->step();

    return f_nn;
}
