#ifndef SAFE_COUPLING_HPP
#define SAFE_COUPLING_HPP

// initialize temp container to hold data and write it all periodically to a single output file
class Safe_coupling{
	
	public:
		Safe_coupling(params &p_in, int nSaveStates_in, int useSafeCouplingQ);			// constructor
		void save(Real *c, const size_t step);											// general save function to call appropriate specialized save function
	
	private:
		params p;										// all simulation parameters
		int counter, write_counter;						// counters for saved states and # of datawrites
		int nSaveStatesBuffer;							// number of states saved in temp vector
		std::vector <Real> buffer;						// temp vector holds concentration fields
		char spbuff[200];								// names of output files
		std::ofstream dataout;							// output data stream
		int size;										// memory size of a single array unit
		
		void save_buffer(Real *c);						// save data to temp vector
		void save_file();								// dump buffer to file
};


#endif // SAFE_COUPLING_HPP
