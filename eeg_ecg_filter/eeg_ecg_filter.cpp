#include <chrono>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <Iir.h>


#define _USE_MATH_DEFINES
#include <math.h>

#include "dnf.h"


using namespace std; 

// Integer of the total number of hidden layers
// required including the input layer
const int NLAYERS = 4;

// Number of taps of the delay line
const int nTapsDNF = 125;

// Sampling rate
double fs = 250; // Hz

// pre-filtering
const int filterorder = 2;
const double eegHighpassCutOff = 0.5; // Hz
const double ecgHighpassCutOff = 0.5; // Hz

// activation
const DNF::ActMethod ACTIVATION = DNF::Act_Tanh;

// dnf learning rate
const double dnf_learning_rate = 0.5;

// input filename
const char inputFilename[] = "rawoutfile.tsv";

// output filename
const char outputFilename[] = "eeg_filtered.dat";

int main(int argc, char* argv[]){
    fprintf(stderr, "Reading noisy EEG file: %s.\n",inputFilename);

    FILE *finput = fopen(inputFilename,"rt");
    FILE *foutput = fopen(outputFilename,"wt");

    int nSamples = 0;

    DNF dnf(NLAYERS,nTapsDNF,fs,ACTIVATION);

    //setting up all the filters required
    Iir::Butterworth::HighPass<filterorder> eeg_filterHP;
    eeg_filterHP.setup(fs,eegHighpassCutOff);
    Iir::Butterworth::HighPass<filterorder> ecg_filterHP;
    ecg_filterHP.setup(fs,ecgHighpassCutOff);

    auto start = std::chrono::high_resolution_clock::now();

    dnf.setLearningRate(0);
    
    for(int i=0; i < (250*120);i++) 
	{
	    double t;
	    double eeg;
	    double ecg;
	    if (fscanf(finput,"%lf\t%lf\t%lf\n",&t,&eeg,&ecg)<1) break;
	    nSamples++;

	    eeg = eeg_filterHP.filter(eeg);
	    ecg = ecg_filterHP.filter(ecg);

	    ecg = ecg * 1000;
	    eeg = eeg * 1000;

	    if (i == (int)(fs*4)){
		dnf.setLearningRate(dnf_learning_rate);
	    }

	    double f_nn = dnf.filter(eeg, ecg);

	    fprintf(foutput,"%f\t%f\t%f",f_nn, dnf.getDelayedSignal(), dnf.getRemover());
	    auto ds = dnf.getLayerWeightDistances();
	    for(const auto& d : ds) fprintf(foutput,"\t%f",d);
	    fprintf(foutput,"\n");
	}

    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    const auto seconds_taken = (double)(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())/1E6;
    const double maxSamplingRate = nSamples / seconds_taken;

    printf("Time taken = %f s, max sampling rate = %f Hz\n", seconds_taken, maxSamplingRate);

    fprintf(stderr, "Written result to: %s.\n",outputFilename);

    fclose(finput);
    fclose(foutput);
}
