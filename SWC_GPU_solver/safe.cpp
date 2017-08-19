
#include <vector>								// vector
#include <fstream>								// ofstream
#include <iostream>								// cout, cerr, endl

#include "SWC_GPU_solver.hpp"					// struct params
#include "safe.hpp"


using namespace std;



// constructor
Safe::Safe(params &p_in, const int nSaveStates_in) : counter(0), write_counter(0) {
	
	// initialize buffer to hold data and write it all periodically to a single output file
	p=p_in;												// params
	nSaveStates=nSaveStates_in;							// number of states to be held in buffer
	if(!p.saveSingleFlag) buffer.resize(p.n*p.ncomponents*nSaveStates);
	printf("save buffer size: %.3f MB\n",p.n*p.ncomponents*nSaveStates*sizeof(Real)/(1024.*1024.));
	
	// initialize variables
	sprintf(spbuff,"/states/state.bin");				// name of output file, may be overwritten later
	size=sizeof(Real);									// memory size of one array cell
	dn=1;												// data coarsening: 1,1+dn -> smaller filesize, right now: failure
	ret=0;
	
	// round up for coarsening 5/2 -> 3: 1,3,5
	nx=(p.nx-1)/dn+1;
	ny=(p.ny-1)/dn+1;
	dt=p.dt*p.stepsSaveState;
	
}


void Safe::save(Real *c, const size_t step){
	
	//~ cout << "save , " << counter << ", " << step << endl;
	
	if(p.saveSingleFlag){									// special mode
		// save every output
		save_single(c);
	}else{												// default mode
		// always save data to buffer
		save_buffer(c);
		// save states periodically, internal counter for fast modulo replacement
		// or save at the last possibility
		if((step<p.stepsEnd-1 and counter==nSaveStates) or step==((p.stepsEnd-1)/p.stepsSaveState)*p.stepsSaveState) save_file(); 
	}
}

void Safe::save_single(Real *c){
	
	sprintf(spbuff,"/states/state_%05d.bin",counter);
	dataout.open(p.pthout+spbuff,ios::binary);
	
	// save header (3 ints + 1 float=4*4 byte): nx,ny,nz,nc,dtt
	// space dimensions, number of components, and time discretization
	dataout.write((char*) &nx, sizeof(int));
	dataout.write((char*) &ny, sizeof(int));
	dataout.write((char*) &p.ncomponents, sizeof(int));
	dataout.write((char*) &dt, sizeof(float));				// # time scale: dt*saveSteps
	
	// one complete concentration field after another:  (u1,u2,...uN, v1,v2,...vN)
	if(dn>1){										// coarsen output data
		for(int c0=0; c0<p.ncomponents; c0++){
		for(int x=0; x<p.nx; x+=dn){
		for(int y=0; y<p.ny; y+=dn){
			dataout.write((char*) &c[x + y*p.nx + c0*p.n], size);
		}}}
	}else{											// no coarsening of output data
		for(int i=0; i<p.ncomponents*p.n; i++) dataout.write((char*) &c[i], size); 
	}
	
	dataout.close();
	
	counter++;
}


void Safe::save_buffer(Real *c){
	
	// save current concentration field to buffer for later file write
	for(int i=0; i<p.ncomponents*p.n; i++) buffer[i+counter*p.ncomponents*p.n] = c[i];
	
	counter++;
}

void Safe::save_file(){
	
	//~ cout << "save_file" << endl;
	
	dataout.open(p.pthout+spbuff, ios::out | ios::binary | ios::app);
	
	// only write header at first file write
	if(write_counter==0){
		// save header (3 ints + 1 float=4*4 byte): nx,ny,nz,nc,dtt
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

