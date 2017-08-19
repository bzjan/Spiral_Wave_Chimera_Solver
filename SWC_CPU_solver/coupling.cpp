
#include <cmath>
#include <vector>								// vector
#include <string>								// string

#include "SWC_CPU_solver.hpp"			// struct params
#include "coupling.hpp"



using namespace std;


void nonlocal_homo_zbke2k(Real* c, Real* cnew, Real* couplecoeff, params &p, Real* kernel, Real* feedback){
	
	
	int idx=0, jx=0, jy=0;
	double sum=0.0;
	
	//~ for(int i=0; i<p.n; i++) cout << i << ", c[i]: " << c[i]  << endl	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
				
			case 2:
				#pragma omp parallel for collapse(2) private(idx,sum,jx,jy)
				for(int x=0; x<p.nx; x++){
				for(int y=0; y<p.ny; y++){
					idx = x + y*p.nx;									// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
					for(int ky=0; ky<p.kdia; ky++){
						jx=x+kx-p.kradius;
						jy=y+ky-p.kradius;
						if(jx<0){ jx+=p.nx; } else if(jx>=p.nx){ jx-=p.nx; }
						if(jy<0){ jy+=p.ny; } else if(jy>=p.ny){ jy-=p.ny; }
						sum += kernel[kx+ky*p.kdia]*c[jx+jy*p.nx];
					}}
					sum -= p.ksum*c[idx];
					
					// calculation
					feedback[idx] = sum;
					cnew[idx] += couplecoeff[0]*sum;
					cnew[p.n+idx] += couplecoeff[1]*sum;
				}}
				break;
				
				
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(1); break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
				
			case 2:
				#pragma omp parallel for collapse(2) private(idx,jx,jy,sum)
				for(int x=0; x<p.nx; x++){
				for(int y=0; y<p.ny; y++){
					idx = x + y*p.nx;								// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
					for(int ky=0; ky<p.kdia; ky++){
						jx=x+kx-p.kradius;
						jy=y+ky-p.kradius;
						if(jx >= 0 && jx < p.nx && jy >= 0 && jy < p.ny){
							sum += kernel[kx+ky*p.kdia]*(c[jx+jy*p.nx]-c[idx]);
						}
					}}
					
					// calculation
					feedback[idx] = sum;
					cnew[idx] += couplecoeff[0]*sum;
					cnew[p.n+idx] += couplecoeff[1]*sum;
				}}
				break;
				
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(1); break;
		}
	}
}



// cdelay = z-component only
// c = z-component only
void nonlocal_delay_homo_zbke2k(Real* c, Real* cnew, Real* cdelay, Real* couplecoeff, params &p, Real* kernel, Real* feedback){
	
	
	int idx=0, jx=0, jy=0;
	double sum=0.0;
	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
				
			case 2:
				#pragma omp parallel for collapse(2) private(idx,sum,jx,jy)
				for(int x=0; x<p.nx; x++){
				for(int y=0; y<p.ny; y++){
					idx = x + y*p.nx;									// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
					for(int ky=0; ky<p.kdia; ky++){
						jx=x+kx-p.kradius;
						jy=y+ky-p.kradius;
						if(jx<0){ jx+=p.nx; } else if(jx>=p.nx){ jx-=p.nx; }
						if(jy<0){ jy+=p.ny; } else if(jy>=p.ny){ jy-=p.ny; }
						sum += kernel[kx+ky*p.kdia]*cdelay[jx+jy*p.nx];
					}}
					sum -= p.ksum*c[idx];
					
					// calculation
					feedback[idx] = sum;
					cnew[idx] += couplecoeff[0]*sum;
					cnew[p.n+idx] += couplecoeff[1]*sum;
				}}
				break;
				
				
			default: 
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); 
				exit(1); 
				break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
				
			case 2:
				#pragma omp parallel for collapse(2) private(idx,jx,jy,sum)
				for(int x=0; x<p.nx; x++){
				for(int y=0; y<p.ny; y++){
					idx = x + y*p.nx;									// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
					for(int ky=0; ky<p.kdia; ky++){
						jx=x+kx-p.kradius;
						jy=y+ky-p.kradius;
						if(jx >= 0 && jx < p.nx && jy >= 0 && jy < p.ny){
							sum += kernel[kx+ky*p.kdia]*(cdelay[jx+jy*p.nx]-c[idx]);
						}
					}}
					
					// calculation
					feedback[idx] = sum;
					cnew[idx] += couplecoeff[0]*sum;
					cnew[p.n+idx] += couplecoeff[1]*sum;
				}}
				break;
				
				
			default:
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!");
				exit(1);
				break;
		}
	}
}













// coupling coefficients are rescaled properly with dt and dxx in create_kernel_and_rescale
void coupling(Real *c, Real *cnew, Real *c0, Real *k, params &p, modelparams &m, Real* kernel, size_t step, Real* feedback){
	
	// iterate over all components by changing the offset
	// move offset in array to address different components
	
	
	if(step>=p.stepsCouplingStart){
	switch(p.diffusionChoice){
		case 5:													// nonlocal kernel, convolution
			if(p.reactionModel==24 or p.reactionModel==2401 or p.reactionModel==25){
				nonlocal_homo_zbke2k(c+p.n,cnew,m.coupling_coeffs,p,kernel,feedback); 
			}
			break;
		case 6:													// nonlocal, delay; fill history until delaySteps+1: only local dynamics, no coupling
			if(step<p.delayStartSteps+1){
				if(p.reactionModel==24 or p.reactionModel==2401 or p.reactionModel==25){
					 nonlocal_homo_zbke2k(c+p.n,cnew,m.coupling_coeffs,p,kernel,feedback);
				}
			}else if(step>=p.delayStartSteps+1){
				Real* cdelay=c0+((step+2) % (p.delayStepsMax+1))*p.n*p.ncomponents;
				if(p.reactionModel==24 or p.reactionModel==2401 or p.reactionModel==25){
					nonlocal_delay_homo_zbke2k(c+p.n,cnew,cdelay+p.n,m.coupling_coeffs,p,kernel,feedback); 
				}
			}
			break;
		case 7: break;											// no coupling
		default: printf("Error: diffusionChoice \"%d\" not implemented!\n",p.diffusionChoice); exit(EXIT_FAILURE); break;
	}}
}
