#ifndef __SWC_CPU_solver_hpp__
#define __SWC_CPU_solver_hpp__

#ifdef DOUBLE
	typedef double Real;
#else
	typedef float Real;
#endif

// default values in ini / cmdline
// static global declaration here: all members are init with 0 (int, float, double) or NULL (char)
struct params{
	int nx, ny, n, ncomponents, spaceDim;
	Real dx, dy;
	Real dt;
	size_t stepsSaveState, stepsSaveStateOffset, stepsEnd, stepsCouplingStart, delayStartSteps, delayStepsMax;
	int delayFlag;
	float delayTimeMax, delayStartTime, couplingStartTime;
	int diffusionChoice;
	std::string pthout, ic, bc, coupling_coeffs_string, pthfn_lc;
	std::vector<Real> spatial_settings1;
	int hehoflag;
	int reactionModel;
	int uSeed, pSeed;
	Real uMin, uMax, uBackground;
	float pSigma;
	Real het1;
	Real ChimeraK, ChimeraKappa;
	int ChimeraCutOffRange;
	int klen, kradius, kdia;
	int saveSingleFlag;
	Real ksum;
	int nCpuThresh;
};

struct modelparams{
	int ncomponents;
	int nparams;
	Real *model_params;
	Real *model_phases;				// excited(nc), refractory(nc), background(nc) for each component
	Real *coupling_coeffs;			// nonlocal coupling or diffusion
};

#endif
