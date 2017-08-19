#ifndef SAFE_HPP
#define SAFE_HPP

// initialize temp container to hold data and write it all periodically to a single output file
class Safe{
	
	public:
		Safe(params &p_in, const int nSaveStates_in);	// constructor
		void save(Real *c, const size_t step);			// general save function to call appropriate specialized save function
	
	private:
		params p;										// all simulation parameters
		int counter, write_counter;						// counters for saved states and # of datawrites
		int nSaveStatesBuffer;							// number of states saved in temp vector
		std::vector <Real> buffer;						// temp vector holds concentration fields
		char spbuff[200];								// names of output files
		std::ofstream dataout;							// output data stream
		int nx,ny,size;									// array dimensions, memory size of array unit
		float dt;										// time scale parameter
		int ret;										// return error value
		
		void save_buffer(Real *c);						// save data to temp vector
		void save_file();								// dump buffer to file
		void save_single(Real *c);						// save each output state for render and small filesize pollution
};


#endif // SAFE_HPP
