
#include <vector>								// vector
#include <fstream>								// ofstream
#include <iostream>								// cout, cerr, endl
#include <cmath>

#include "SWC_CPU_solver.hpp"			// struct params
#include "safe.hpp"


using namespace std;



// constructor
Safe::Safe(params &p_in, const int nSaveStatesBuffer_in) : counter(0), write_counter(0) {
	
	p=p_in;												// params
	nSaveStatesBuffer=nSaveStatesBuffer_in;				// number of states to be held in buffer
	size_t n=0;											// total size of dump vector
	
	// initialize variables
	sprintf(spbuff,"/states/state.bin");				// name of output file, may be overwritten later in singleSave mode
	size=sizeof(Real);									// memory size of one array cell

	nx=p.nx;
	ny=p.ny;
	n = p.n*p.ncomponents*nSaveStatesBuffer;
	
	if(!p.saveSingleFlag) buffer.resize(n);
	printf("save buffer size: %.3f MB\n",buffer.size()*sizeof(Real)/(1024.*1024.));
	
	dt=p.dt*p.stepsSaveState;
	
}


void Safe::save(Real *c, const size_t step){
	
	if(p.saveSingleFlag){								// special mode
		// save every output and create rendered image with mma; for immediate scrollring render and low file size pollution
		save_single(c);
	}else{												// default mode
		// always save data to temp vector
		save_buffer(c);
		// save states periodically, internal counter for fast modulo replacement
		// or save at the last possibility
		if((step<p.stepsEnd-1 and counter==nSaveStatesBuffer) or step==((p.stepsEnd-1)/p.stepsSaveState)*p.stepsSaveState) save_file();
	}
}

void Safe::save_single(Real *c){
	
	sprintf(spbuff,"/states/state_%05d.bin",counter);
	dataout.open(p.pthout+spbuff,ios::binary);
	
	// save header (3 ints + 1 float=4*4 byte): nx,ny,nc,dtt
	// space dimensions, number of components, and time discretization
	dataout.write((char*) &nx, sizeof(int));
	dataout.write((char*) &ny, sizeof(int));
	dataout.write((char*) &p.ncomponents, sizeof(int));
	dataout.write((char*) &dt, sizeof(float));				// # time scale: dt*saveSteps
	
	// one complete concentration field after another:  (u1,u2,...uN, v1,v2,...vN)
	for(int i=0; i<p.n*p.ncomponents; i++) dataout.write((char*) &c[i], size);
	
	dataout.close();
	
	counter++;
}


void Safe::save_buffer(Real *c){
	
	// save current concentration field to temp vector for later file write
	for(int i=0; i<p.ncomponents*p.n; i++) buffer[i+counter*p.ncomponents*p.n] = c[i];	
	
	counter++;
}

void Safe::save_file(){
	
	
	// open, append to file
	dataout.open(p.pthout+spbuff, ios::out | ios::binary | ios::app);
	
	// only write header at first file write
	if(write_counter==0){
		// save header (3 ints + 1 float=4*4 byte): nx,ny,nc,dtt
		// space dimensions, number of components, and time discretization
		dataout.write((char*) &nx, sizeof(int));
		dataout.write((char*) &ny, sizeof(int));
		dataout.write((char*) &p.ncomponents, sizeof(int));
		dataout.write((char*) &dt, sizeof(float));				// # time scale: dt*saveSteps
	}
	
	// save data to file
	for(int i=0; i<counter; i++){
	for(int j=0; j<p.ncomponents*p.n; j++){
		dataout.write((char*) &buffer[j+i*p.ncomponents*p.n], size);
	}}
	
	dataout.close();
	
	counter=0;
	write_counter++;
}
