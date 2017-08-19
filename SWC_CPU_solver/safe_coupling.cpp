
#include <vector>								// vector
#include <fstream>								// ofstream
#include <iostream>								// cout, cerr, endl
#include <cmath>

#include "SWC_CPU_solver.hpp"			// struct params
#include "safe_coupling.hpp"


using namespace std;



// constructor
Safe_coupling::Safe_coupling(params &p_in, const int nSaveStatesBuffer_in, const int useSafeCouplingQ) : counter(0), write_counter(0) {
	
	p=p_in;															// params
	nSaveStatesBuffer=nSaveStatesBuffer_in;							// will at one point become an ini parameter
	
	// initialize variables
	sprintf(spbuff,"/states/coupling_data.bin");				// name of output file, may be overwritten later
	size=sizeof(Real);											// memory size of one array cell
	
	if(useSafeCouplingQ){						// only create buffer if it is actually used
		size_t n = p.n*nSaveStatesBuffer;		// total size of dump vector
		buffer.resize(n);
	}
	
	//~ cout << "buffer size: " << buffer.size() << endl;
	printf("save coupling buffer size: %.3f MB\n",buffer.size()*sizeof(Real)/(1024.*1024.));
	
}


void Safe_coupling::save(Real *c, const size_t step){
	
	// always save data to temp vector
	save_buffer(c);
	// save states periodically, internal counter for fast modulo replacement
	// or save at the last possibility
	if((step<p.stepsEnd-1 and counter==nSaveStatesBuffer) or step==((p.stepsEnd-1)/p.stepsSaveState)*p.stepsSaveState) save_file();
}

void Safe_coupling::save_buffer(Real *c){
	
	// save current concentration field to temp vector for later file write
	for(int i=0; i<p.n; i++) buffer[i+counter*p.n] = c[i]/p.dt;
	
	counter++;
}

void Safe_coupling::save_file(){
	
	//~ cout << "save_file" << endl;
	
	// open, append to file
	dataout.open(p.pthout+spbuff, ios::out | ios::binary | ios::app);
	
	// only write header at first file write
	if(write_counter==0){
		//~ cout << "write header" << endl;
		// save header (1 int = 1*4 byte): nx
		dataout.write((char*) &p.n, sizeof(int));
	}
	// DEBUG
	//~ for(int i=0; i<10; i++) cout << c[i] << " "; cout << endl;
	
	// save data to file
	//~ cout << "write data" << endl;
	//~ cout << "p.n: " << p.n << ", p.ncomponents: " << p.ncomponents << ", *: " << p.ncomponents*p.n << endl;
	//~ cout << "counter: " << counter << ", write_counter: " << write_counter << endl;
	for(int i=0; i<counter; i++){
	for(int j=0; j<p.n; j++){
		dataout.write((char*) &buffer[j+i*p.n], size);
	}}
	
	dataout.close();
	
	counter=0;
	write_counter++;
}
