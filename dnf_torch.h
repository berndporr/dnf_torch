/**
 * License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
 * Copyright (c) 2020 by Bernd Porr
 * Copyright (c) 2020 by Sama Daryanavard
 **/

#ifndef _DNF_H
#define _DNF_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <torch/torch.h>
#include <thread>
#include <iostream>
#include <deque>

#ifdef NDEBUG
constexpr bool debugOutput = false;
#else
constexpr bool debugOutput = true;
#endif

class DelayLine {
public:
    void init(int delaySamples) {
        delaySamples_ = delaySamples;
        buffer_ = std::deque<float>(delaySamples_, 0.0f);
    }
    
    inline float process(float input) {
        float output = buffer_.front();
        buffer_.pop_front();
        buffer_.push_back(input);
        return output;
    }
    
    float get(int i) const {
	return buffer_[i];
    }
    
    float getNewest() const {
	return buffer_.back();
    }
    
private:
    int delaySamples_ = 0;
    std::deque<float> buffer_;
};


/**
 * Deep Neuronal Filter class.
 * It's designed to be as simple as possible with
 * only a few parameters.
 **/
class DNF {
public:

    /**
     * Options for activation functions of all neurons in the network.
     **/
    enum ActMethod {Act_Sigmoid = 1, Act_Tanh = 2, Act_ReLU = 3, Act_NONE = 0};

private:
    struct Net : public torch::nn::Module {
	std::vector<torch::nn::Linear> fc;
	Net(int nLayers, int nInput, bool withBias = false);
	torch::Tensor forward(torch::Tensor x, ActMethod am);
    };

public:
    /**
     * Constructor which sets up the delay lines, network layers
     * and also calculates the number of neurons per layer so
     * that the final layer always just has one neuron.
     * \param nLayers Number of layers
     * \param nTaps Number of taps for the delay line feeding into the 1st layer
     * \param samplingrate Sampling rate of the signals used in Hz.
     * \param am The activation function for the neurons. Default is tanh.
     * \param tryGPU Does the learning on the GPU if available.
     **/
    DNF(const int nLayers,
	const int nTaps,
	const float samplingrate,
	const ActMethod am = Act_Tanh,
	const bool tryGPU = false
	);

    inline void setLearningRate(float mu) {
	for (auto& group : optimizer->param_groups()) {
            static_cast<torch::optim::SGDOptions&>(group.options()).lr(mu);
        }
    }

    /**
     * Realtime sample by sample filtering operation
     * \param signal The signal contaminated with noise. Should be less than one.
     * \param noise The reference noise. Should be less than one.
     * \returns The filtered signal where the noise has been removed by the DNF.
     **/
    float filter(const float signal, const float noise);

    /**
     * Returns the length of the delay line which
     * delays the signal polluted with noise.
     * \returns Number of delay steps in samples.
     **/
    inline int getSignalDelaySteps() const {
	return signalDelayLineLength;
    }
    
    /**
     * Returns the delayed with noise polluted signal by the delay 
     * indicated by getSignalDelaySteps().
     * \returns The delayed noise polluted signal sample.
     **/
    inline float getDelayedSignal() const {
	return signal_delayLine.get(0);
    }
    
    /**
     * Returns the remover signal.
     * \returns The current remover signal sample.
     **/
    inline float getRemover() const {
	return remover;
    }
    
    /**
     * Returns the output of the DNF: the noise
     * free signal.
     * \returns The current output of the DNF which is idential to filter().
     **/
    inline float getOutput() const {
	return f_nn;
    }
    
    /**
     * Frees the memory used by the DNF.
     **/
    ~DNF() {
	delete optimizer;
	delete model;
    }

    /**
     * Gets the weight distances per layer
     * \returns The Eucledian weight distance in relation to the initial weights.
     **/
    std::vector<float> getLayerWeightDistances() const;

    /**
     * Gets the overall weight distsance
     * \returns The sum of all layer weight distances.
     **/
    float getWeightDistance() const;
    
private:

    void saveInitialParameters() {
	for (const auto& p : model->parameters()) { 
	    initialParameters.push_back(p.detach().clone());
	}
    }

    Net* model = nullptr;
    torch::optim::SGD* optimizer = nullptr;
    std::vector<torch::Tensor> initialParameters;
    const int noiseDelayLineLength;
    const int signalDelayLineLength;
    const float fs;
    const ActMethod actMethod;
    DelayLine signal_delayLine;
    DelayLine noise_delayLine;
    float remover = 0;
    float f_nn = 0;
    static constexpr double xavierGain = 0.01;
	torch::Device device = torch::kCPU;
};

#endif
