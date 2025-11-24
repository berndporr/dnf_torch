#include "dnf_torch.h"

int main(int argc, char* argv[]){
    DelayLine dl;
    dl.init(5);
    for(int i=0;i<10;i++) {
	float orig = i*2+10;
	float dld = dl.process(i*2+10);
	printf("%d: %f -> %f\n",i,orig,dld);
	if ((i > 4) && (dld != ((i-5)*2+10))) {
	    throw "Delay not working";
	}
	if ((i < 5) && (dld != 0)) {
	    throw "Delay at start not init.";
	}
    }
}
