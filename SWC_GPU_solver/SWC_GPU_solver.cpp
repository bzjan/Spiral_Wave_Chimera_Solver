/* 
 *  Jan Totz <jantotz@itp.tu-berlin.de>
 */


// C++11 & Boost libraries
#include <boost/program_options.hpp>			// command line parsing
#include <boost/property_tree/ptree.hpp>		// support structure for ini file writing
#include <boost/property_tree/ini_parser.hpp>	// read/write ini file
#include <iostream>								// cout, cerr
#include <fstream>								// ofstream, ifstream
#include <chrono>								// chrono::high_resolution_clock::now()
#include <random>								// default_random_engine, normal_distribution<double>, uniform_real_distribution
#include <stdlib.h>								// system, NULL, EXIT_FAILURE
#include <unistd.h>								// readlink


#include "SWC_GPU_solver.hpp"	// struct params
#include "SWC_GPU_solver.hu"


// namespaces
namespace po = boost::program_options;
namespace pt = boost::property_tree;
using namespace std;

inline double posi_fmod(double i, double n){ return fmod((fmod(i,n)+n),n); }

// custom parser for arrays in inifile, works for arrays of any kind: string, int, float, double...
template<typename T>
std::vector<T> to_array(const std::string& s){
	std::vector<T> result;
	std::stringstream ss(s);
	std::string item;
	while(std::getline(ss, item, ',')) result.push_back(boost::lexical_cast<T>(item));
	return result;
}

std::string get_executable_path(){
	
	char buff[1024];
	ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
	string pthexe;
	
	// get pthfnexe
	if(len != -1){				// success
		buff[len] = '\0';
		pthexe = std::string(buff);
	}else{							// error
		printf("Error! Could not find path to executable!"); exit(EXIT_FAILURE);
	}
	
	// remove filename from pathexe
	const size_t last_slash_idx = pthexe.rfind('/');
	if (std::string::npos != last_slash_idx){
		pthexe = pthexe.substr(0, last_slash_idx+1);
	}
	
	return pthexe;
}


std::string RealArrayToString(Real *array, int n){
	
	std::string result="";
	std::ostringstream strs;
	
	for(int i=0; i<n-1; i++) strs << array[i] << ",";
	strs << array[n-1];
	
	return strs.str();
	
}


std::string RealVectorToString(vector<Real> vec){
	
	std::string result="";
	std::ostringstream strs;
	
	for(int i=0; i<(int)vec.size()-1; i++) strs << vec[i] << ",";
	strs << vec[vec.size()-1];
	
	return strs.str();
}

std::string replaceEnvironmentVariables(std::string input){
	
	size_t pos1 = input.find("$", 0);
	if(pos1 != std::string::npos){										// "$" was found in string
		size_t pos2 = input.find("/", pos1);
		string envVarString = input.substr(pos1+1,pos1+pos2-1).c_str();
		string envVarValue;
		if( getenv(envVarString.c_str()) != NULL){
			envVarValue = getenv(envVarString.c_str());
		}
		return input.replace(pos1,pos1+pos2,envVarValue);
	}else{
		return input;
	}
}

// read command line parameters with boost
void main_cmdln_params(int &ac, char *av[], string &pthexe, string &pthini, string &ini, params &p){
	
	int errorflag=0;			// no error: 0, error: 1
	string spatial_settings1_string;
	
	try{
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("pthini", po::value<string>(&pthini)->default_value(pthexe+"../ini"), "path for ini")
			("pthout", po::value<string>(&p.pthout)->default_value("-1"), "path for output")
			("ini", po::value<string>(&ini)->default_value("chimera_spiral_zbke2k.ini"), "name of inifile")
			("spatial_settings1", po::value<string>(&spatial_settings1_string)->default_value("-1"), "excitable array, no whitespace!")
			("coupling_coeffs", po::value<string>(&p.coupling_coeffs_string)->default_value("-1"), "coupling/diffusion coefficents, no whitespace!")
			("uSeed", po::value<int>(&p.uSeed)->default_value(-1), "uSeed")
			("pSeed", po::value<int>(&p.pSeed)->default_value(-1), "seed for parameter heterogeneity")
			("nx", po::value<int>(&p.nx)->default_value(-1), "nx")
			("ny", po::value<int>(&p.ny)->default_value(-1), "ny")
			("diffusionChoice", po::value<int>(&p.diffusionChoice)->default_value(-1), "diffusionChoice")
			("hehoflag", po::value<int>(&p.hehoflag)->default_value(-1), "hehoflag")
			("reactionModel", po::value<int>(&p.reactionModel)->default_value(-1), "reactionModel")
			("ChimeraCutOffRange", po::value<int>(&p.ChimeraCutOffRange)->default_value(-1), "ChimeraCutOffRange")
			("blockWidth", po::value<int>(&p.blockWidth)->default_value(-1), "blockWidth")
			("stepsEnd", po::value<size_t>(&p.stepsEnd)->default_value(0), "stepsEnd")
			("stepsSaveState", po::value<size_t>(&p.stepsSaveState)->default_value(0), "stepsSaveState")
			("stepsSaveStateOffset", po::value<size_t>(&p.stepsSaveStateOffset)->default_value(0), "stepsSaveStateOffset")
			("delayTimeMax", po::value<float>(&p.delayTimeMax)->default_value(-1.0f), "delayTimeMax")
			("ChimeraK", po::value<Real>(&p.ChimeraK)->default_value(-1.0), "ChimeraK")
			("ChimeraKappa", po::value<Real>(&p.ChimeraKappa)->default_value(-1.0), "ChimeraKappa")
			("dt", po::value<Real>(&p.dt)->default_value(-1.0), "dt")
			("dx", po::value<Real>(&p.dx)->default_value(-1.0), "dx")
			("dy", po::value<Real>(&p.dy)->default_value(-1.0), "dy")
			;
		
		po::variables_map vm;
		po::store(po::parse_command_line(ac, av, desc), vm);
		po::notify(vm);
		
		if(vm.count("help")){ cout << desc << "\n"; exit(EXIT_SUCCESS); }
	}
	catch(exception& e){
		cerr << "error: " << e.what() << "\n";
		errorflag=1;
	}
	catch(...){
		cerr << "Exception of unknown type!\n";
		errorflag=1;
	}
	
	if(errorflag){
		cout<<"--- program shutdown due to error at cmdln_params ---"<<endl; 
		exit(EXIT_FAILURE);
	}
	
	// convert string to vector
	p.spatial_settings1 = to_array<Real>(spatial_settings1_string);
	
}

// resize phase inside function
void get_limitCycleData(vector<Real>& phases, int nc, string pthfn_lc){ 
	
	ifstream fs;
	fs.open(pthfn_lc);
	string temp_line;
	int ni=0;
	while(getline(fs,temp_line)){
		if(!temp_line.empty()) ni++;
	}
	fs.clear();
	fs.seekg(fs.beg);
	//~ cout << "ni: " << ni << endl;		// DEBUG
	phases.resize(ni*nc);
	for(int i=0; i<ni; i++){
	for(int c=0; c<nc; c++){
		fs>>phases[c+i*nc];
	}}
	fs.close();
}



// preliminary for all components, needs extra functions
// fields have the form: (u1,v1, u2,v2, uN,vN); better for GPU performance!
void initialCondition(Real *cfield, params &p, modelparams &m){
	
	int nc = p.ncomponents;
	
	if(p.ic=="uniform_noise"){
		printf("ic: %s\n",p.ic.c_str());
		
		// random numbers from uniform distribution
		default_random_engine generator(p.uSeed);
		uniform_real_distribution<Real> udistribution(p.uMin,p.uMax);
		
		for(int c=0; c<p.ncomponents; c++){
		for(int i=0; i<p.n; i++){
			cfield[c+i*nc]=udistribution(generator);
		}}
		
	
	}else if(p.ic=="homo"){										// homogeneous background concentration
		printf("ic: %s\n",p.ic.c_str());
		
		for(int i=0; i<p.n; i++){
		for(int c=0; c<p.ncomponents; c++){
			cfield[c+i*nc]=m.model_phases[c+2*p.ncomponents];	// 2 = index of background
		}}
		
		
		
	}else if(p.ic=="phase_proto_spiral"){
		switch(p.spaceDim){
			case 2:  printf("ic: phase proto spiral wave\n"); break;
			default: printf("Error: spaceDim=%d is insufficient for phase proto spiral ic, use 2 instead!\n",p.spaceDim); exit(EXIT_FAILURE); break;
		}
		
		// spatial_settings1 overloaded for options: spatial_settings1 = {phasex0,phasey0,phi0,chirality}
		auto result1 = std::minmax_element(p.spatial_settings1.begin(), p.spatial_settings1.end());
		float minmaxdiff=*result1.second - *result1.first;
		if(minmaxdiff<1e-5){ // default settings, otherwise: ini settings
			p.spatial_settings1={0.5,0.5,M_PI,1};
		}
		
		// pattern parameters
		float phasex0 = p.spatial_settings1[0]; 			// x-coords of phase singularity = spiral center
		float phasey0 = p.spatial_settings1[1];				// y-coords of phase singularity = spiral center
		float phi0 = p.spatial_settings1[2];				// initial phase offset
		int chi = p.spatial_settings1[3];					// chirality
		
		vector<Real> phases;				// initialize empty vector
		get_limitCycleData(phases,p.ncomponents,p.pthfn_lc);
		int ni=phases.size()/p.ncomponents;
		
		
		for(int y = 0; y < p.ny; ++y){
		for(int x = 0; x < p.nx; ++x){
			int index = posi_fmod( atan2(y-phasey0*p.ny,chi*(x-phasex0*p.nx)) - phi0, 2.0*M_PI )/(2.0*M_PI)*(ni-1);
			for(int c=0; c<p.ncomponents; c++) cfield[c+(x+y*p.nx)*nc]=phases[c+index*p.ncomponents];
		}}
		
		
		
		
	}else{
		cout<<"Error: ic \""<<p.ic<<"\" does not exist!" <<endl; exit(EXIT_FAILURE);
	}

}


void main_parameterDistribution(Real *hetArray, params &p){
	
	// random numbers from gaussian distribution
	default_random_engine generator(p.pSeed);
	normal_distribution<double> p_normal_distribution(p.het1,p.pSigma);
	uniform_real_distribution<double> p_uniform_distribution(p.het1-p.pSigma,p.het1+p.pSigma);

switch(p.hehoflag){
	case 0:				// homogeneous distribution
		cout << "main_parameterDistribution (" << p.hehoflag << "): homogeneous" << endl;
		for(int i=0; i<p.n; i++) hetArray[i]=p.het1;
		break;
	
	case 1:				// heterogeneous, normal distribution
		cout << "main_parameterDistribution (" << p.hehoflag << "): bounded normal distribution" << endl;
		for(int i=0; i<p.n; i++){
			float prnd=0.;
			while(prnd<p.het1-p.pSigma or prnd>p.het1+p.pSigma) prnd = p_normal_distribution(generator);     // limits for parameter p
			hetArray[i]=prnd;
		}
		break;
		
	case 7:				// heterogeneous, uniform distribution
		cout << "main_parameterDistribution (" << p.hehoflag << "): uniform distribution" << endl;
		for(int i=0; i<p.n; i++) hetArray[i]=p_uniform_distribution(generator);
		break;
	
		
	default:
		cout << "chosen value (" << p.hehoflag << ") for hehoflag is not implemented!" << endl; exit(1);
		break;
}

}


// only linux!, OS-independent option: boost
void main_housekeeper(string pthout, string pthexe){
	
	#ifdef _WIN32
	
	#elif __linux__
	// remove potential old files in directories
	string cmd0 = "rm -r "+pthout+"/* 2> /dev/null";
	int systemRet = system(const_cast<char*>(cmd0.c_str()));
	if(systemRet == -1){cout<<"rm system command failed"<<endl;}
	
	// create directory for output
	cmd0 = "mkdir -p "+pthout+" "+pthout+"/states";
	systemRet = system(const_cast<char*>(cmd0.c_str()));
	if(systemRet == -1){cout<<"mkdir system command failed"<<endl;}
	
	// copy current source code into output directory
	cmd0 = "mkdir -p "+pthout+" "+pthout+"/source_code";
	systemRet = system(const_cast<char*>(cmd0.c_str()));
	if(systemRet == -1){cout<<"mkdir system command failed"<<endl;}
	// cp .{hpp,cpp} is not supported in sh
	cmd0 = "cd "+pthexe+" && cp *.cpp *.hpp *.cu *.hu "+pthout+"/source_code/";
	systemRet = system(const_cast<char*>(cmd0.c_str()));
	if(systemRet == -1){cout<<"cp system command failed"<<endl;}
	#endif
	
}



// strip string of unwanted chars
string stripString(string &str){

	// escaped quotation mark
	char badchars[] = "\"";

	for (unsigned int i = 0; i < strlen(badchars); ++i){
		str.erase (std::remove(str.begin(), str.end(), badchars[i]), str.end());
	}

	return str;
}

void setReactionCouplingParams(params &ip, pt::ptree pt, modelparams &m, params &pExtra){
	
	switch(ip.reactionModel){
		case 24:												// zbke2k
		case 2401:												// zbke2k qhet
			m.ncomponents=2;
			m.nparams=9;
			m.model_params = new Real[9] {1.0/0.11,1.7e-5,1.6e-3,0.1,1.7e-5,1.2,2.4e-4,0.7,5.25e-4};		// ooeps1, eps2, eps3, alpha, beta, gamma, mu, q, phi0
			m.model_phases = new Real[6] {0.1,0.01,0.0001,0.3,0.0,0.0};
			m.coupling_coeffs = new Real[2] {1.0/0.11,2.0};
			break;
			
		case 25:												// fhn
			m.ncomponents=2;
			m.nparams=2;
			m.model_params = new Real[2] {1.0/0.05,1.1};							// ooeps, a
			m.model_phases = new Real[6] {1.5,0.0,0.0,1.5,-1.1,-0.656333};			// bg = stable fixed point
			m.coupling_coeffs = new Real[2] {1.0,0.0};
			break;
			
		default:												// default case
			printf("Error: reactionModel not implemented. (readInifile @ line: %d)\n",__LINE__); exit(EXIT_FAILURE);
			break;
	}
	
	
	// a1) read model data from inifile
	std::vector<Real> modelParameters0 = to_array<Real>(pt.get<std::string>("model_parameters.modelParams","0"));
	std::vector<Real> coupling_coeffs0 = to_array<Real>(pt.get<std::string>("model_parameters.coupling_coeffs","0"));
	std::vector<Real> phases0 = to_array<Real>(pt.get<std::string>("model_parameters.phases","0"));
	
	// a2) read model.coupling_coeffs from commandline
	if(pExtra.coupling_coeffs_string!="-1") coupling_coeffs0 = to_array<Real>(pExtra.coupling_coeffs_string);
	
	// b) apply it; if not set, then write default values into resulting ini file
	if(modelParameters0.size()!=1 or modelParameters0[0]!=0){
		m.model_params = (Real *) malloc(modelParameters0.size()*sizeof(Real));
		for(size_t i=0; i<modelParameters0.size(); i++) m.model_params[i]=modelParameters0[i];
	}else{
		std::string s=RealArrayToString(m.model_params,m.nparams);
		pt.put<std::string>("model_parameters.modelParams",s);
	}
	
	if(coupling_coeffs0.size()!=1 or coupling_coeffs0[0]!=0){
		m.coupling_coeffs = (Real *) malloc(coupling_coeffs0.size()*sizeof(Real));
		for(size_t i=0; i<coupling_coeffs0.size(); i++) m.coupling_coeffs[i]=coupling_coeffs0[i];
	}else{
		std::string s=RealArrayToString(m.coupling_coeffs,m.ncomponents);
		pt.put<std::string>("model_parameters.coupling_coeffs",s);
	}
	
	if(phases0.size()!=1 or phases0[0]!=0){
		m.model_phases = (Real *) malloc(phases0.size()*sizeof(Real));
		for(size_t i=0; i<phases0.size(); i++) m.model_phases[i]=phases0[i];
	}else{
		std::string s=RealArrayToString(m.model_phases,m.ncomponents*3);
		pt.put<std::string>("model_parameters.phases",s);
	}
}


// read parameters from text file
void main_readInifile(string pthini, string pthexe, string &ini, params &ip, pt::ptree pt, modelparams &m, params &pExtra){
	
	// initialize variables
	string pthfnini=pthini+"/"+ini;
	
	// read ini
	pt::ini_parser::read_ini(pthfnini, pt);
	cout<<"\n############# reading ini file ###################"<<endl;
	cout<<pthfnini<<endl;
	ip.nx = max(pt.get<int>("numerics_space.nx",100),1);		// max: in case user input is 0 instead of 1
	ip.ny = max(pt.get<int>("numerics_space.ny",100),1);
	ip.dx = pt.get<Real>("numerics_space.dx",1.0);
	ip.dy = pt.get<Real>("numerics_space.dy",1.0);
	ip.bc = pt.get<string>("numerics_space.boundary_condition","periodic");			// periodic or neumann
	ip.diffusionChoice = pt.get<int>("numerics_space.diffusionChoice",0);
	ip.dt = pt.get<Real>("numerics_time.dt",0.01);
	ip.stepsSaveState = pt.get<size_t>("numerics_time.stepsSaveState",10000);
	ip.stepsSaveStateOffset = pt.get<size_t>("numerics_time.stepsSaveStateOffset",0);
	ip.stepsEnd = pt.get<size_t>("numerics_time.stepsEnd",100000);
	ip.couplingStartTime = pt.get<float>("numerics_time.couplingStartTime",0);
	ip.delayTimeMax = pt.get<float>("numerics_time.delayTimeMax",0);
	ip.delayHistoryUpdateStep = pt.get<int>("numerics_time.delayHistoryUpdateStep",1);
	ip.delayStartTime = pt.get<float>("numerics_time.delayStartTime",1);
	ip.blockWidth = pt.get<int>("device.gpu_blockWidth",8);
	ip.use_tiles = pt.get<int>("device.gpu_use_tiles",0);
	ip.ic = pt.get<string>("initial_condition.ic","uniform_noise");
	ip.pthfn_lc = pt.get<string>("initial_condition.pthfn_lc",pthexe+"/../lc_zbke2k_phi0_1.6e-4.dat");
	ip.uSeed = pt.get<unsigned int>("initial_condition.uSeed",1);
	ip.uMin = pt.get<Real>("initial_condition.uMin",0.);
	ip.uMax = pt.get<Real>("initial_condition.uMax",1.);
	ip.reactionModel = pt.get<int>("model_parameters.reactionModel",10);
	ip.hehoflag = pt.get<int>("model_parameters.hehoflag",0);
	ip.pSeed = pt.get<unsigned int>("model_parameters.pSeed",1);
	ip.pSigma = pt.get<float>("model_parameters.pSigma",0.01);
	ip.het1 = pt.get<Real>("model_parameters.het1",0.01);								// heterogeneity parameter 1
	ip.enforceValueLimits = pt.get<int>("model_parameters.enforceValueLimits",0);
	ip.ChimeraK = pt.get<Real>("chimera.ChimeraK",0.0001);
	ip.ChimeraKappa = pt.get<Real>("chimera.ChimeraKappa",3.0);
	ip.ChimeraCutOffRange = pt.get<int>("chimera.ChimeraCutoffRange",3);
	ip.pthout = pt.get<string>("dir_management.pthout",pthexe+"/../../Simulations/test");
	ip.saveSingleFlag = pt.get<int>("dir_management.saveSingleFlag",0);
	ip.spatial_settings1 = to_array<Real>(pt.get<std::string>("initial_condition.spatial_settings1","0")); 
	
	// assign additional cmd line values to params
	if(pExtra.uSeed>=0){ ip.uSeed = pExtra.uSeed; pt.put<int>("initial_condition.uSeed",ip.uSeed); }
	if(pExtra.pSeed>=0){ ip.pSeed = pExtra.pSeed; pt.put<int>("model_parameters.pSeed",ip.pSeed); }
	if(pExtra.diffusionChoice>=0){ ip.diffusionChoice = pExtra.diffusionChoice; pt.put<int>("numerics_space.diffusionChoice",ip.diffusionChoice); }
	if(pExtra.nx>=0){ ip.nx = pExtra.nx; pt.put<int>("numerics_space.nx",ip.nx); }
	if(pExtra.ny>=0){ ip.ny = pExtra.ny; pt.put<int>("numerics_space.ny",ip.ny); }
	if(pExtra.dx>=0){ ip.dx = pExtra.dx; pt.put<Real>("numerics_space.dx",ip.dx); }
	if(pExtra.dy>=0){ ip.dy = pExtra.dy; pt.put<Real>("numerics_space.dy",ip.dy); }
	if(pExtra.dt>=0){ ip.dt = pExtra.dt; pt.put<Real>("numerics_time.dt",ip.dt); }
	if(pExtra.ChimeraCutOffRange>=0){ ip.ChimeraCutOffRange = pExtra.ChimeraCutOffRange; pt.put<int>("chimera.ChimeraCutOffRange",ip.ChimeraCutOffRange); }
	if(pExtra.ChimeraKappa>=0){ ip.ChimeraKappa = pExtra.ChimeraKappa; pt.put<Real>("chimera.ChimeraKappa",ip.ChimeraKappa); }
	if(pExtra.ChimeraK>=0){ ip.ChimeraK = pExtra.ChimeraK; pt.put<Real>("chimera.ChimeraK",ip.ChimeraK); }
	if(pExtra.hehoflag>=0){ ip.hehoflag = pExtra.hehoflag; pt.put<int>("model_parameters.hehoflag",ip.hehoflag); }
	if(pExtra.reactionModel>=0){ ip.reactionModel = pExtra.reactionModel; pt.put<int>("model_parameters.reactionModel",ip.reactionModel); }
	if(pExtra.blockWidth>=0){ ip.blockWidth = pExtra.blockWidth; pt.put<int>("device.gpu_blockWidth",ip.blockWidth); }
	if(pExtra.stepsEnd>0){ ip.stepsEnd = pExtra.stepsEnd; pt.put<size_t>("numerics_time.stepsEnd",ip.stepsEnd); }
	if(pExtra.stepsSaveState>0){ ip.stepsSaveState = pExtra.stepsSaveState; pt.put<size_t>("numerics_time.stepsSaveState",ip.stepsSaveState); }
	if(pExtra.stepsSaveStateOffset>0){ ip.stepsSaveStateOffset = pExtra.stepsSaveStateOffset; pt.put<size_t>("numerics_time.stepsSaveStateOffset",ip.stepsSaveStateOffset); }
	if(pExtra.delayTimeMax>=0){ ip.delayTimeMax = pExtra.delayTimeMax; pt.put<float>("numerics_time.delayTime",ip.delayTimeMax); }
	if(pExtra.spatial_settings1[0]>=0){ ip.spatial_settings1 = pExtra.spatial_settings1; pt.put<std::string>("initial_condition.spatial_settings1",RealVectorToString(ip.spatial_settings1)); }
	if(pExtra.pthout!="-1"){ ip.pthout = pExtra.pthout; pt.put<std::string>("dir_management.pthout",ip.pthout); }
	
	// interpret environment variables in pthout string
	ip.pthout = replaceEnvironmentVariables(ip.pthout);
	ip.pthfn_lc = replaceEnvironmentVariables(ip.pthfn_lc);
	
	// recalculate total simulation time, so that no superfluous, unsaved simulation steps are performed
	ip.stepsEnd = (ip.stepsEnd/ip.stepsSaveState)*ip.stepsSaveState;
	pt.put<size_t>("numerics_time.stepsEnd",ip.stepsEnd);
	
	// check correctness of delay settings, if inappropriate switch to easiest case
	if(ip.diffusionChoice==6){ 
		ip.delayFlag=1;
		if(ip.delayHistoryUpdateStep>1) ip.delayFlag=2;
	}else{ ip.delayFlag=0; }
	
	if(ip.delayFlag==1 and ip.delayTimeMax==0.0){
		printf("\nWarning: delayTimeMax of 0.0 is not supported for diffusionChoice=6! Switching to diffusionChoice=5 (readInifile @ line: %d)\n",__LINE__);
		ip.diffusionChoice=5;
		ip.delayFlag=0;
	}
	pt.put<int>("numerics_space.diffusionChoice",ip.diffusionChoice);
	
	// input checks
	if(ip.delayFlag){
		if(ip.delayStartTime<ip.delayTimeMax and ip.couplingStartTime<ip.delayTimeMax){
			cout << "\nWarning: delayTimeMax must be <= delayStartTime or couplingStartTime! Setting delayStartTime = delayTimeMin! Correcting...\n" << endl;
			ip.delayStartTime = ip.delayTimeMax;
		}
	}
	pt.put<float>("numerics_time.delayStartTime",ip.delayStartTime);
	pt.put<float>("numerics_time.delayTimeMax",ip.delayTimeMax);
	
	
	// find the space dimension automatically
	ip.spaceDim=2;
	if(ip.nx==1) ip.spaceDim--;
	if(ip.ny==1) ip.spaceDim--;
	
	
	#ifdef DOUBLE
		cout << "dataytpe mode: DOUBLE" << endl;
	#else
		cout << "dataytpe mode: FLOAT" << endl;
	#endif
	
	setReactionCouplingParams(ip,pt,m,pExtra);
	ip.ncomponents=m.ncomponents;
	
	ip.delayStepsMax=(ip.delayTimeMax/ip.dt)/ip.delayHistoryUpdateStep*!!ip.delayFlag;
	ip.delayStartSteps=(ip.delayStartTime/ip.dt)/ip.delayHistoryUpdateStep*!!ip.delayFlag;
	ip.stepsCouplingStart=ip.couplingStartTime/ip.dt;
	pt.put<size_t>("numerics_time.delayStepsMax",ip.delayStepsMax);
	pt.put<size_t>("numerics_time.delayStartSteps",ip.delayStartSteps);
	ip.pthout=stripString(ip.pthout)+"_gpu";
	pt.put<std::string>("dir_management.pthout",ip.pthout);
	main_housekeeper(ip.pthout,pthexe);
	ip.bc=stripString(ip.bc);
	ip.ic=stripString(ip.ic);
	ip.n=ip.nx*ip.ny;							// total number of cells
	pt.put<int>("numerics_space.n",ip.n);
	
	string iniOutString=ip.pthout+"/used0.ini";
	pt::ini_parser::write_ini(iniOutString,pt);
	
	// terminal output
	cout<<"spaceDim: "<<ip.spaceDim<<endl;
	printf("nx: %d, ny: %d\n",ip.nx,ip.ny);
	printf("dx: %.2f, dy: %.2f\n",ip.dx,ip.dy);
	cout<<"dt: "<<ip.dt<<endl;
	cout<<"reaction model: "<<ip.reactionModel<<endl;
	cout<<"stepsSaveState: "<<ip.stepsSaveState<<" steps = "<<ip.stepsSaveState*ip.dt<<" time units"<<endl;
	cout<<"stepsEnd: "<<ip.stepsEnd<<" steps = "<<ip.stepsEnd*ip.dt<<" time units"<<endl;
	//~ cout<<"pthout: "<<ip.pthout<<endl;																				// DEBUG
	cout<<"##############################################\n"<<endl;
	
}

// Courant-Friedrichs-Levy (CFL) criterion
void calcCFL(params &p, Real *k, modelparams &m){
	
	if(p.diffusionChoice==0 || p.diffusionChoice==1 || p.diffusionChoice==2){
		Real kmax = *max_element(k,k+p.n);
		Real diffmax = *max_element(m.coupling_coeffs,m.coupling_coeffs+m.ncomponents);
		if(p.hehoflag) diffmax=kmax;
		Real cfl=2.0*p.spaceDim*p.dt*diffmax/(p.dx*p.dx);
		
		cout<<"CFL: "<< cfl << endl;
		if(cfl>0.9) { cout<<"CFL violated! "<<cfl<<">0.9 (conservative)."<<endl; exit(1); }
	}
}


// translate array from (u1,v1,w1) -> (u1 u2 ... uN, ...) etc (=transpose)
// acts only on cAnalysis, not for array that is going to be saved!
// = transpose!
void translateArrayOrder(Real *cfield, Real *cAnalysis, params &p, int &untranslatedFlag){
	
	switch(p.ncomponents){
		
		case 2:
			for(int c=0; c<p.ncomponents; c++){
			for(int i=0; i<p.n; i++){
				cAnalysis[i+c*p.n]=cfield[c+i*p.ncomponents];
			}}
			break;
			
		default:
			cout << "translateArrayOrder: number of components not supported!" << endl;
			break;
	}
	untranslatedFlag=0;
	
}


// save state as binary float
void saveParamDistro(Real *k, params &p){
	
	//~ cout << "saveParamDistro" << endl;
	
	char spbuff[100];
	sprintf(spbuff,"/paramDistro.bin");
	ofstream dataout;
	dataout.open(p.pthout+spbuff,ios::binary);
	
	// save header (3 ints): nx,ny,nz
	dataout.write((char*) &p.nx, sizeof(int));
	dataout.write((char*) &p.ny, sizeof(int));
	
	// save data
	for(int n=0; n<p.n; n++) dataout.write((char*) &k[n], sizeof(Real));
	
	dataout.close();
}



int main(int ac, char* av[]){

// message at program start
cout << "############ Solver is starting ############ " << endl;

// 0. parameter declaration
struct params inifile_params, inifile_params_cmdline;
struct modelparams model_params;
auto t1=chrono::high_resolution_clock::now(), t2=chrono::high_resolution_clock::now();
string pthini, ini;
string pthexe = get_executable_path();

// 1. read pth of inifile from commandline (default path)
main_cmdln_params(ac,av,pthexe,pthini,ini,inifile_params_cmdline);

// 2a. read data from inifile
pt::ptree iniPropTree;
main_readInifile(pthini,pthexe,ini,inifile_params,iniPropTree,model_params,inifile_params_cmdline);

// 3. prepare system
int nc = inifile_params.ncomponents;											// number of components
Real *cfield = new Real[inifile_params.n*nc]{};									// dynamic variables = concentrations, {} = init with zeros
Real *kfield = new Real[inifile_params.n]{};									// parameter k for every cell
initialCondition(cfield,inifile_params,model_params);
main_parameterDistribution(kfield,inifile_params);
saveParamDistro(kfield,inifile_params);

// 3b. CFL check
calcCFL(inifile_params,kfield,model_params);


// 4. simulation
t1 = std::chrono::high_resolution_clock::now();					// timer start
solverGPU(cfield,kfield,inifile_params,model_params);			// run simulation
t2 = std::chrono::high_resolution_clock::now();					// timer end
int runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

// write report
ofstream dataout;
char spbuff[100];
sprintf(spbuff,"execution time: %d ms = %d h, %d min, %d s.",runtime,runtime/1000/3600,runtime/1000/60%60,int(floor(runtime/1000.+0.5))%60);
dataout.open(inifile_params.pthout+"/report.txt", ios::out);
dataout<<spbuff<<endl;
dataout.close();
cout<<"\n"<<spbuff<<endl;

// deallocate memory
delete[] cfield;
delete[] kfield;
delete[] model_params.coupling_coeffs;
delete[] model_params.model_params;
delete[] model_params.model_phases;

return 0;
}

