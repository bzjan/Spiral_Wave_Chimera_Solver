#ifndef __SWC_GPU_solver_hpp__
#define __SWC_GPU_solver_hpp__

// CUDA datatypes
#include "vector_types.h"

#ifdef DOUBLE
	typedef double Real;
	typedef double1 Real1;
	typedef double2 Real2;
	typedef double4 Real4;
#else
	typedef float Real;
	typedef float1 Real1;
	typedef float2 Real2;
	typedef float4 Real4;
#endif

// default values in ini / cmdline
// static global declaration here: all members are init with 0 (int, float, double) or NULL (char)
struct params{
	int nx, ny, n, ncomponents, spaceDim;
	Real dx, dy;
	Real dt;
	size_t stepsSaveState, stepsSaveStateOffset, stepsEnd, stepsCouplingStart, delayStartSteps, delayStepsMin, delayStepsMax;
	int enforceValueLimits;
	int blockWidth;
	int saveSingleFlag, delayFlag;
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
	int use_tiles;
	int kradius, kdia;
	Real ksum;
	int delayHistoryUpdateStep;
};

struct modelparams{
	int ncomponents;
	int nparams;
	Real *model_params;
	Real *model_phases;				// excited(nc), refractory(nc), background(nc) for each component
	Real *coupling_coeffs;			// nonlocal coupling or diffusion
};

void translateArrayOrder(Real *c, Real *cAnalysis, params &p, int &untranslatedFlag);

#endif
