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
const bool debugOutput = false;
#else
const bool debugOutput = true;
#endif

class DelayLine {
public:
    void init(size_t delaySamples)
    {
        delaySamples_ = delaySamples;
        buffer_ = std::deque<float>(delaySamples_, 0.0f);
    }

    float process(float input) {
        // Output is the oldest value (front of deque)
        float output = buffer_.front();
        buffer_.pop_front();
	
        // Push new input to the back
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
    size_t delaySamples_ = 0;
    std::deque<float> buffer_;
};


/**
 * Main Deep Neuronal Network main class.
 * It's designed to be as simple as possible with
 * only a few parameters as possible.
 **/
class DNF {
public:

    /**
     * Options for activation functions of the neuron
     * 0 for using the logistic function
     * 1 for using the hyperbolic tan function
     * 2 for unity function (no activation)
     */
    enum ActMethod {Act_Sigmoid = 1, Act_Tanh = 2, Act_ReLU = 3, Act_NONE = 0};

    struct Net : public torch::nn::Module {
	std::vector<torch::nn::Linear> fc;
	
	Net(int nLayers, int nInput) {
	    // calc an exp reduction of the numbers always reaching 1
	    const float b = (float)exp(log(nInput)/(nLayers-1));
	    int inputNeurons = nInput;
	    for(int i=1;i<nLayers;i++) {
		int outputNeurons = (int)ceil(nInput / pow(b,i));
		if (i == (nLayers-1)) outputNeurons = 1;
		char tmp[256];
		sprintf(tmp,"fc%d_%d_%d",i,inputNeurons,outputNeurons);
		if (debugOutput)
		    fprintf(stderr,"Creating FC layer: %s\n",tmp);
		torch::nn::Linear ll = register_module(tmp, torch::nn::Linear(inputNeurons, outputNeurons));
		torch::nn::init::xavier_uniform_(ll->weight,xavierGain);
		torch::nn::init::constant_(ll->bias, 0.0);
		fc.push_back(ll);
		if (1 == outputNeurons) break;
		inputNeurons = outputNeurons;
	    }
	}

	torch::Tensor forward(torch::Tensor x, ActMethod am) {
	    for(auto& f:fc) {
		switch (am) {
		default:
		case Act_Tanh:
		    x = torch::atan(f->forward(x));
		    break;
		case Act_Sigmoid:
		    x = torch::sigmoid(f->forward(x));
		    break;
		case Act_ReLU:
		    x = torch::relu(f->forward(x));
		    break;
		case Act_NONE:
		    x = f->forward(x);
		    break;
		}
	    }
	    return x;
	}

    };

    /**
     * Constructor which sets up the delay lines, network layers
     * and also calculates the number of neurons per layer so
     * that the final layer always just has one neuron.
     * \param nLayers Number of layers
     * \param nTaps Number of taps for the delay line feeding into the 1st layer
     * \param samplingrate Sampling rate of the signals used in Hz.
     * \param am The activation function for the neurons. Default is tanh.
     **/
    DNF(const int nLayers,
	const int nTaps,
	const float samplingrate,
	const ActMethod am = Act_Tanh
	) : noiseDelayLineLength(nTaps),
	    signalDelayLineLength(noiseDelayLineLength / 2),
	    fs(samplingrate),
	    actMethod(am)
	{

	signal_delayLine.init(signalDelayLineLength);
	noise_delayLine.init(noiseDelayLineLength);

	torch::manual_seed(1);
    
	torch::DeviceType device_type;
	if (false) /** torch::cuda::is_available()) **/ {
	    std::cout << "CUDA available! Training on GPU." << std::endl;
	    device_type = torch::kCUDA;
	} else {
	    std::cout << "Training on CPU." << std::endl;
	    device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	model = new Net(nLayers,nTaps);
	model->to(device);
	model->train();
	
	optimizer = new torch::optim::SGD(model->parameters(), 0);
    }

    void setLearningRate(float mu) {
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
     * Returns the output of the DNF: the the noise
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
     * Todo: get the weight distance
     **/
    float getLayerWeightDistance(int layerIndex) {
	return 0;
    }
    
private:
    Net* model = nullptr;
    torch::optim::SGD* optimizer = nullptr;
    const int noiseDelayLineLength;
    const int signalDelayLineLength;
    const float fs;
    const ActMethod actMethod;
    DelayLine signal_delayLine;
    DelayLine noise_delayLine;
    float remover = 0;
    float f_nn = 0;
    static constexpr double xavierGain = 0.01;
};

#endif
