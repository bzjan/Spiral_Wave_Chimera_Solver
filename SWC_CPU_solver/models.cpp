
#include <cmath>
#include <vector>								// vector
#include <string>								// string

#include "SWC_CPU_solver.hpp"			// struct params
#include "models.hpp"



using namespace std;


// fitz hugh nagumo, classic nerve conduction + heart + bz
void model_fhn(Real *c, Real *cnew, Real *k, params &p){
	Real ooeps=1.0/0.05;
	Real a=1.1;
	
	int i2;
	#pragma omp parallel for private(i2)
	for(int i=0; i<p.n; i++){
		i2=i+p.n;
		cnew[i]=c[i]+p.dt*( ooeps*(c[i]-1.0/3.0*c[i]*c[i]*c[i]-c[i2]) ); 		// u
		cnew[i2]=c[i2]+p.dt*( c[i] + a );										// v
	}
}


// zbke2k, more complete modified BZ model
// source: Totz, PRE 2015
void model_zbke2k(Real *c, Real *cnew, params &p){
	Real ooeps1=9.090909090909091;	// 1.0f/0.11
	Real gammaEps2=2.04e-5;
	Real eps31=1.0016;
	Real alpha=0.1;
	Real beta=1.7e-5;
	Real mu=2.4e-4;
	Real q=0.7;
	Real phi=5.25e-4;
	
	Real uss=0.0;
	Real temp=0.0;

	int i2;
	#pragma omp parallel for private(i2,uss,temp)
	for(int i=0; i<p.n; i++){
		i2=i+p.n;
		uss=1.0/(4.0*gammaEps2)*(-(1.0-c[i2])+sqrt(1.0-2.0*c[i2] + c[i2]*c[i2] + 16.0*gammaEps2*c[i]));
		temp=alpha*c[i2]/(eps31-c[i2]);
		
		cnew[i]=c[i]+p.dt*( ooeps1*(phi-c[i]*c[i]-c[i]+gammaEps2*uss*uss+uss*(1.0-c[i2])+(mu-c[i])/(mu+c[i])*(q*temp+beta)) ); 			// x
		cnew[i2]=c[i2]+p.dt*( 2.0*phi + uss*(1.0-c[i2]) - temp );																		// z
	}
}

// zbke2k, more complete modified BZ model accounting for different q as placeholders for different bead sizes
// source: Totz, PRE 2015
void model_zbke2k_qhet(Real *c, Real *cnew, Real *k, params &p){
	Real ooeps1=9.090909090909091;	// 1.0f/0.11
	Real gammaEps2=2.04e-5;
	Real eps31=1.0016;
	Real alpha=0.1;
	Real beta=1.7e-5;
	Real mu=2.4e-4;
	Real phi=5.25e-4;
	
	Real uss=0.0;
	Real temp=0.0;

	int i2;
	#pragma omp parallel for private(i2,uss,temp)
	for(int i=0; i<p.n; i++){
		i2=i+p.n;
		uss=1.0/(4.0*gammaEps2)*(-(1.0-c[i2])+sqrt(1.0-2.0*c[i2] + c[i2]*c[i2] + 16.0*gammaEps2*c[i]));
		temp=alpha*c[i2]/(eps31-c[i2]);
		
		cnew[i]=c[i]+p.dt*( ooeps1*(phi-c[i]*c[i]-c[i]+gammaEps2*uss*uss+uss*(1.0-c[i2])+(mu-c[i])/(mu+c[i])*(k[i]*temp+beta)) ); 			// x, het
		cnew[i2]=c[i2]+p.dt*( 2.0*phi + uss*(1.0-c[i2]) - temp );																		// z
	}
}





void reaction(Real *c, Real *cnew, Real *k, params &p){
	switch(p.reactionModel){
		case 24:	model_zbke2k(c,cnew,p); break;
		case 2401:	model_zbke2k_qhet(c,cnew,k,p); break;
		case 25:	model_fhn(c,cnew,k,p); break;
		default: printf("chosen reactionModel (%d) is not implemented! Program Abort!",p.reactionModel); exit(EXIT_FAILURE); break;
	}
}
