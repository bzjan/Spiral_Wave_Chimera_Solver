#ifndef SAFE_HPP
#define SAFE_HPP

class Safe{
	
	public:
		Safe(params &p_in, const int nSaveStates_in);	// constructor
		void save(Real *c, const size_t step);			// general save function to call appropriate specialized save function
	
	private:
		params p;										// all simulation parameters
		int counter, write_counter;						// counters for saved states and # of datawrites
		int nSaveStates;								// number of states saved in temp vector
		std::vector <Real> buffer;						// temp vector holds concentration fields
		char spbuff[200];								// names of output files
		std::ofstream dataout;							// output data stream
		int dn,nx,ny,size;							// coarse-graining parameter, array dimensions, memory size of array unit
		float dt;										// time scale parameter
		int ret;										// return error value
		
		void save_buffer(Real *c);						// save data to temp vector
		void save_file();								// dump temp data to file
		void save_single(Real *c);						// save each output state for render and small filesize pollution
};


#endif // SAFE_HPP



