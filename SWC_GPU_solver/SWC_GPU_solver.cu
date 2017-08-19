/* 
 *  Jan Totz <jantotz@itp.tu-berlin.de>
 */

#include <vector>								// vector (for tip finder)
#include <iostream>								// cout, endl
#include <list>									// list (for fila finder)
#include <string>								// string (in struct of hpp)
#include <stdio.h>								// printf
#include <fstream>								// ofsream, ifstream

#include "SWC_GPU_solver.hpp"
#include "safe.hpp"
#include "vector_types_operator_overloads.hu"


// cuda error checking function
// usage: checkCUDAError("test",__LINE__);
void checkCUDAError(const char *msg, int line) {
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess){
		fprintf(stderr, "Cuda error: line %d: %s: %s.\n", line, msg, cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
#endif
}


// structure holds all device pointers
// simplifies pointer handling
struct device_pointers {
	
	// arrays
	Real *c0;
	Real *c;
	Real *cnew;
	Real *cdelay;
	Real *k;
	Real *output;
	Real *mask;
	
	// previously arrays, now a single value for faster access speed
	Real2 coupling_coeffs2;
	
};

struct host_pointers {
	
	Real *c;
	Real *ctemp;
	Real *mask;
	int *defects;
	
};

class streams {
	
	public:
	streams(){
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
	}
	
	~streams(){
		cudaStreamDestroy(stream1);
		cudaStreamDestroy(stream2);
	}
	
	cudaStream_t stream1;
	cudaStream_t stream2;
	
};


// CPU pointer swap function for GPU
template <typename T>
void swapGPU(T &a, T &b){
	T t = a;
	a = b;
	b = t;
}



// fitz hugh nagumo, classic nerve conduction + heart + bz (zykov)
__global__ void model_fhn(Real2 *c, Real2 *cnew, int len, Real dt){
	Real ooeps=1.0/0.05;
	Real a=0.9;			// oscillatory
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len){
		cnew[i].x=c[i].x+dt*( ooeps*(c[i].x-1.0/3.0*c[i].x*c[i].x*c[i].x-c[i].y) );			// u
		cnew[i].y=c[i].y+dt*( c[i].x + a );													// v
	}
}



// zbke2k, more complete BZ model
// source: Taylor, Tinsley Toth
// must be double, not float!
__global__ void model_zbke2k(Real2 *c, Real2 *cnew, int len, Real dt){
	Real ooeps1=9.090909090909091;	// 1.0/0.11
	Real gammaEps2=2.04e-5;
	Real eps31=1.0016;
	Real alpha=0.1;
	Real beta=1.7e-5;
	
	Real mu=2.4e-4;
	Real q=0.7;
	Real phi=1.6e-4;
	
	Real uss=0.0;
	Real temp=0.0;

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len){
		uss=1.0/(4.0*gammaEps2) * (-(1.0-c[i].y) + sqrt(1.0 + fma(c[i].y,c[i].y,-2.0*c[i].y) + 16.0*gammaEps2*c[i].x));
		temp=alpha*c[i].y/(eps31-c[i].y);
		
		cnew[i].x=c[i].x+dt*( ooeps1*(phi-c[i].x*c[i].x-c[i].x+gammaEps2*uss*uss+uss*(1.0-c[i].y)+(mu-c[i].x)/(mu+c[i].x)*(q*temp+beta)) );			// x
		cnew[i].y=c[i].y+dt*(2.0*phi + uss*(1.0-c[i].y) - temp);																						// z
	}
}


// zbke2k, more complete BZ model
// source: Taylor Tinsley Toth paper
// must be double, not float!
__global__ void model_zbke2k_qhet(Real2 *c, Real2 *cnew, int len, Real dt, Real *het){
	Real ooeps1=9.090909090909091;	// 1.0/0.11
	Real gammaEps2=2.04e-5;
	Real eps31=1.0016;
	Real alpha=0.1;
	Real beta=1.7e-5;
	
	Real mu=2.4e-4;
	Real phi=1.6e-4;
	
	Real uss=0.0;
	Real temp=0.0;

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len){
		uss=1.0/(4.0*gammaEps2) * (-(1.0-c[i].y) + sqrt(1.0 + fma(c[i].y,c[i].y,-2.0*c[i].y) + 16.0*gammaEps2*c[i].x));
		temp=alpha*c[i].y/(eps31-c[i].y);
		
		cnew[i].x=c[i].x+dt*( ooeps1*(phi-c[i].x*c[i].x-c[i].x+gammaEps2*uss*uss+uss*(1.0-c[i].y)+(mu-c[i].x)/(mu+c[i].x)*(het[i]*temp+beta)) );			// x
		cnew[i].y=c[i].y+dt*(2.0*phi + uss*(1.0-c[i].y) - temp);																							// z
	}
}




void reaction(device_pointers *d, params &p, streams *s){
	
	
	int warpsize=32;
	dim3 nblocks((p.ncomponents*p.n-1)/warpsize+1);
	dim3 nthreads(warpsize);
	
	switch(p.reactionModel){
		case 24: model_zbke2k<<<nblocks,nthreads,0,s->stream1>>>((Real2 *)d->c,(Real2 *)d->cnew,p.n,p.dt); break;
		case 2401: model_zbke2k_qhet<<<nblocks,nthreads,0,s->stream1>>>((Real2 *)d->c,(Real2 *)d->cnew,p.n,p.dt,d->k); break;
		case 25: model_fhn<<<nblocks,nthreads,0,s->stream1>>>((Real2 *)d->c,(Real2 *)d->cnew,p.n,p.dt); break;
		default:
			printf("chosen reactionModel (%d) is not implemented! Program Abort!",p.reactionModel);
			exit(EXIT_FAILURE);
			break;
	}
	
	checkCUDAError("reaction()",__LINE__);
}




template <int BC>
__global__ void nonlocal_delay_homo_zbke2k_2d(Real2 *input, Real2 *output, Real2 *delay, int kdia, int kradius, Real ksum, int nx, int ny, const Real * __restrict__ M, const Real2 couplecoeff){
	
	// thread indices
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	if(x<nx && y<ny){
		int idx=x+y*nx;
		Real sum {};
		
		// global boundary conditions
		switch(BC){
			case 0:											// periodic
				for(int kx=0; kx<kdia; kx++){
				for(int ky=0; ky<kdia; ky++){
					int jx=x+kx-kradius;
					int jy=y+ky-kradius;
					if(jx<0){ jx+=nx; } else if(jx>=nx){ jx-=nx; }
					if(jy<0){ jy+=ny; } else if(jy>=ny){ jy-=ny; }
					sum += M[kx+ky*kdia]*delay[jx+jy*nx].y;
				}}
				sum -= ksum*input[idx].y;
				
				output[idx] += couplecoeff*sum;
				break;
			
			case 1:											// neumann
				for(int kx=0; kx<kdia; kx++){
				for(int ky=0; ky<kdia; ky++){
					int jx=x+kx-kradius;
					int jy=y+ky-kradius;
					if(jx >= 0 && jx < nx && jy >= 0 && jy < ny){
						sum += M[kx+ky*kdia]*(delay[jx+jy*nx].y - input[idx].y);
					}
				}}
				
				output[idx] += couplecoeff*sum;
				break;
		}
	}
}



template <int BC, int mask_radius>
__global__ void nonlocal_delay_homo_tiled_zbke2k_2d(Real2 *input, Real2 *output, Real2 *input_delay, int width, int height, 
int o_tile_width, const Real * __restrict__ M, const Real2 diffcoeff){
	
	// declare shared memory arrays for tiles, BLOCK_WIDTH = TILE_WIDTH, but != O_TILE_WIDTH
	extern __shared__ Real input_shared[];
	
	// thread indices, no dependence on blockIdx, blockDim to support tiling. 1 tile = 1 block
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// row and column calculation, since in- and output tiles have different sizes
	int col_o = blockIdx.x*o_tile_width+tx;
	int row_o = blockIdx.y*o_tile_width+ty;
	int col_i = col_o-mask_radius;
	int row_i = row_o-mask_radius;
	
	// global boundary conditions
	int idx = col_i+row_i*width;
	
	Real output_temp{};
	switch(BC){
		case 0:									// periodic
			switch(mask_radius){
				case 5:
					if(col_i==-5) idx += width;
					if(col_i==width+4) idx -= width;
					if(row_i==-5) idx += width*height;
					if(row_i==height+4) idx -= width*height;
				case 4:
					if(col_i==-4) idx += width;
					if(col_i==width+3) idx -= width;
					if(row_i==-4) idx += width*height;
					if(row_i==height+3) idx -= width*height;
				case 3:
					if(col_i==-3) idx += width;
					if(col_i==width+2) idx -= width;
					if(row_i==-3) idx += width*height;
					if(row_i==height+2) idx -= width*height;
				case 2:
					if(col_i==-2) idx += width;
					if(col_i==width+1) idx -= width;
					if(row_i==-2) idx += width*height;
					if(row_i==height+1) idx -= width*height;
				case 1:
					if(col_i==-1) idx += width;
					if(col_i==width) idx -= width;
					if(row_i==-1) idx += width*height;
					if(row_i==height) idx -= width*height;
					break;
			}
			input_shared[tx+ty*blockDim.x] = input_delay[idx].y;
			__syncthreads();
			
			// calculation, not all threads are needed. Threads at tile boundaries are excluded.
			if(ty<o_tile_width && tx < o_tile_width){
				int mask_width=2*mask_radius+1;
				Real input0=input[row_o*width+col_o].y;
				for(int i=0; i<mask_width; i++){
				for(int j=0; j<mask_width; j++){
					output_temp += M[j*mask_width+i]*(input_shared[i+tx+blockDim.x*(j+ty)]-input0);
				}}
			}
			__syncthreads();
			
			break;
		
		case 1:																		// neumann
			if((row_i>=0) && (row_i<height) && (col_i>=0) && (col_i<width)){
				input_shared[tx+ty*blockDim.x] = input_delay[idx].y;
			}else{
				input_shared[tx+ty*blockDim.x] = 0.0;
			}
			__syncthreads();
			
			// calculation, not all threads are needed. Threads at tile boundaries are excluded.
			if(ty<o_tile_width && tx < o_tile_width){
				Real ksum{};
				int mask_width=2*mask_radius+1;
				Real input0=input[row_o*width+col_o].y;
				for(int i=0; i<mask_width; i++){
				for(int j=0; j<mask_width; j++){
					output_temp += M[j*mask_width+i]*input_shared[i+tx+blockDim.x*(j+ty)];
					ksum += M[j*mask_width+i]*!!input_shared[i+tx+blockDim.x*(j+ty)];
				}}
				output_temp -= ksum*input0;
			}
			__syncthreads();
			
			
			break;
	
	}
	
	// write output, exclude output from threads, which contributed to loading data into shared memory but did not calc output
	if(row_o<height && col_o<width && tx<o_tile_width && ty<o_tile_width){
		output[row_o*width+col_o] += diffcoeff*output_temp;
	}
}



void nonlocal_delay_homo_tiled_zbke2k(device_pointers *d, params &p, streams *s){
	
	
	int mem_size=0;
	
	int maskWidth=2*p.ChimeraCutOffRange+1;
	int o_TileWidth=p.blockWidth-maskWidth+1;
	
	dim3 nblocks((p.nx-1)/o_TileWidth+1);
	dim3 nthreads(p.blockWidth);
	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 2:
				nblocks.y=(p.ny-1)/o_TileWidth+1;
				nthreads.y=p.blockWidth;
				mem_size=p.blockWidth*p.blockWidth;
				switch(p.ChimeraCutOffRange){
					case 1: nonlocal_delay_homo_tiled_zbke2k_2d<0,1><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 2: nonlocal_delay_homo_tiled_zbke2k_2d<0,2><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 3: nonlocal_delay_homo_tiled_zbke2k_2d<0,3><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 4: nonlocal_delay_homo_tiled_zbke2k_2d<0,4><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 5: nonlocal_delay_homo_tiled_zbke2k_2d<0,5><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					default: printf("ChimeraCutOffRange must be < 6! Program Abort!"); exit(1); break;
				}
				break;
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(1); break;
		}
	}else if(p.bc=="neummann"){
		switch(p.spaceDim){
			case 2:
				nblocks.y=(p.ny-1)/o_TileWidth+1;
				nthreads.y=p.blockWidth;
				mem_size=p.blockWidth*p.blockWidth;
				switch(p.ChimeraCutOffRange){
					case 1: nonlocal_delay_homo_tiled_zbke2k_2d<1,1><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 2: nonlocal_delay_homo_tiled_zbke2k_2d<1,2><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 3: nonlocal_delay_homo_tiled_zbke2k_2d<1,3><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 4: nonlocal_delay_homo_tiled_zbke2k_2d<1,4><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 5: nonlocal_delay_homo_tiled_zbke2k_2d<1,5><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					default: printf("ChimeraCutOffRange must be < 6! Program Abort!"); exit(1); break;
				}
				break;
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(1); break;
		}
	}
	
	checkCUDAError("nonlocal_delay_homo_tiled_zbke2k()",__LINE__);
}


void nonlocal_delay_homo_zbke2k(device_pointers *d, params &p, streams *s){
	
	
	dim3 nblocks((p.nx-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);
	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 2:
				nblocks.y=(p.ny-1)/p.blockWidth+1;
				nthreads.y=p.blockWidth;
				nonlocal_delay_homo_zbke2k_2d<0><<<nblocks,nthreads>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.kdia,p.kradius,p.ksum,p.nx,p.ny,d->mask,d->coupling_coeffs2);
				break;
			default: printf("spaceDim is not chosen correctly for nonlocal_delay_homo_zbke2k! Program Abort!"); exit(1); break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 2:
				nblocks.y=(p.ny-1)/p.blockWidth+1;
				nthreads.y=p.blockWidth;
				nonlocal_delay_homo_zbke2k_2d<1><<<nblocks,nthreads>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.kdia,p.kradius,p.ksum,p.nx,p.ny,d->mask,d->coupling_coeffs2);
				break;
			default: printf("spaceDim is not chosen correctly for nonlocal_delay_homo_zbke2k! Program Abort!"); exit(1); break;
		}
	}
	
	checkCUDAError("nonlocal_delay_homo_zbke2k()",__LINE__);
}





template <int BC>
__global__ void nonlocal_homo_zbke2k_2d(Real2 *input, Real2 *output, int kdia, int kradius, Real ksum, int nx, int ny, const Real * __restrict__ M, const Real2 couplecoeff){
	
	// thread indices
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	
	if(x<nx && y<ny){
		int idx=x+y*nx;
		Real sum {};
		
		// global boundary conditions
		switch(BC){
			case 0:											// periodic
				for(int kx=0; kx<kdia; kx++){
				for(int ky=0; ky<kdia; ky++){
					int jx=x+kx-kradius;
					int jy=y+ky-kradius;
					if(jx<0){ jx+=nx; } else if(jx>=nx){ jx-=nx; }
					if(jy<0){ jy+=ny; } else if(jy>=ny){ jy-=ny; }
					sum += M[kx+ky*kdia]*input[jx+jy*nx].y;
				}}
				sum -= ksum*input[idx].y;
				
				output[idx] += couplecoeff*sum;		// calculation
				break;
			
			case 1:											// neumann
				for(int kx=0; kx<kdia; kx++){
				for(int ky=0; ky<kdia; ky++){
					int jx=x+kx-kradius;
					int jy=y+ky-kradius;
					if(jx >= 0 && jx < nx && jy >= 0 && jy < ny){
						sum += M[kx+ky*kdia]*(input[jx+jy*nx].y - input[idx].y);
					}
				}}
				
				output[idx] += couplecoeff*sum;		// calculation
				break;
		}
	}
}



void nonlocal_homo_zbke2k(device_pointers *d, params &p, streams *s){
	
	
	dim3 nblocks((p.nx-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);
	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 2:
				nblocks.y=(p.ny-1)/p.blockWidth+1;
				nthreads.y=p.blockWidth;
				nonlocal_homo_zbke2k_2d<0><<<nblocks,nthreads>>>((Real2*)d->c,(Real2*)d->cnew,p.kdia,p.kradius,p.ksum,p.nx,p.ny,d->mask,d->coupling_coeffs2);
				break;
			default: printf("spaceDim is not chosen correctly for nonlocal_homo_zbke2k! Program Abort!"); exit(1); break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 2:
				nblocks.y=(p.ny-1)/p.blockWidth+1;
				nthreads.y=p.blockWidth;
				nonlocal_homo_zbke2k_2d<1><<<nblocks,nthreads>>>((Real2*)d->c,(Real2*)d->cnew,p.kdia,p.kradius,p.ksum,p.nx,p.ny,d->mask,d->coupling_coeffs2);
				break;
			default: printf("spaceDim is not chosen correctly for nonlocal_homo_zbke2k! Program Abort!"); exit(1); break;
		}
	}
	
	checkCUDAError("nonlocal_homo_zbke2k()",__LINE__);
}




template <int BC, int mask_radius>
__global__ void nonlocal_homo_tiled_zbke2k_2d(Real2 *input, Real2 *output, int width, int height,
 int o_tile_width, const Real* __restrict__ M, const Real2 diffcoeff){
	
	// declare shared memory arrays for tiles, BLOCK_WIDTH = TILE_WIDTH, but != O_TILE_WIDTH
	extern __shared__ Real input_shared[];
	
	// thread indices, no dependence on blockIdx, blockDim to support tiling. 1 tile = 1 block
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// row and column calculation, since in- and output tiles have different sizes
	int col_o = blockIdx.x*o_tile_width+tx;
	int row_o = blockIdx.y*o_tile_width+ty;
	int col_i = col_o-mask_radius;
	int row_i = row_o-mask_radius;
	
	// global boundary conditions
	int idx = col_i+row_i*width;
	
	Real output_temp{};
	switch(BC){
		case 0:			// periodic
			switch(mask_radius){
				case 5:
					if(col_i==-5) idx += width;
					if(col_i==width+4) idx -= width;
					if(row_i==-5) idx += width*height;
					if(row_i==height+4) idx -= width*height;
				case 4:
					if(col_i==-4) idx += width;
					if(col_i==width+3) idx -= width;
					if(row_i==-4) idx += width*height;
					if(row_i==height+3) idx -= width*height;
				case 3:
					if(col_i==-3) idx += width;
					if(col_i==width+2) idx -= width;
					if(row_i==-3) idx += width*height;
					if(row_i==height+2) idx -= width*height;
				case 2:
					if(col_i==-2) idx += width;
					if(col_i==width+1) idx -= width;
					if(row_i==-2) idx += width*height;
					if(row_i==height+1) idx -= width*height;
				case 1:
					if(col_i==-1) idx += width;
					if(col_i==width) idx -= width;
					if(row_i==-1) idx += width*height;
					if(row_i==height) idx -= width*height;
					break;
			}
			input_shared[tx+ty*blockDim.x] = input[idx].y;
			__syncthreads();
			
			// calculation, not all threads are needed. Threads at tile boundaries are excluded.
			if(ty<o_tile_width && tx < o_tile_width){
				int mask_width=2*mask_radius+1;				// should be input value for speedup?
				Real input0=input_shared[tx+mask_radius + (ty+mask_radius)*blockDim.x];
				for(int i=0; i<mask_width; i++){
				for(int j=0; j<mask_width; j++){
					output_temp += M[j*mask_width+i]*(input_shared[i+tx+blockDim.x*(j+ty)]-input0);
				}}
			}
			__syncthreads();
			
			break;
		
		case 1:			// neumann
			if(col_i>=0 and col_i<width and row_i>=0 and row_i<height){
				input_shared[tx+ty*blockDim.x] = input[idx].y;
			}else{
				input_shared[tx+ty*blockDim.x] = 0.0;
			}
			__syncthreads();
			
			// calculation, not all threads are needed. Threads at tile boundaries are excluded.
			if(ty<o_tile_width && tx < o_tile_width){
				Real ksum{};
				int mask_width=2*mask_radius+1;				// should be input value for speedup?
				Real input0=input_shared[tx+mask_radius + (ty+mask_radius)*blockDim.x];
				for(int i=0; i<mask_width; i++){
				for(int j=0; j<mask_width; j++){
					output_temp += M[j*mask_width+i]*input_shared[i+tx+blockDim.x*(j+ty)];
					ksum += M[j*mask_width+i]*!!input_shared[i+tx+blockDim.x*(j+ty)];
				}}
				output_temp -= ksum*input0;
			}
			__syncthreads();
			
			break;
	}
	
	// write output, exclude output from threads, which contributed to loading data into shared memory but did not calc output
	if(row_o<height && col_o<width && tx<o_tile_width && ty<o_tile_width){
		output[row_o*width+col_o] += diffcoeff*output_temp;
	}
}




void nonlocal_homo_tiled_zbke2k(device_pointers *d, params &p, streams *s){
	
	
	int mem_size=0;
	
	int maskWidth=2*p.ChimeraCutOffRange+1;
	int o_TileWidth=p.blockWidth-maskWidth+1;
	
	
	dim3 nblocks((p.nx-1)/o_TileWidth+1);
	dim3 nthreads(p.blockWidth);
	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 2:
				nblocks.y=(p.ny-1)/o_TileWidth+1;
				nthreads.y=p.blockWidth;
				mem_size=p.blockWidth*p.blockWidth;	// number of elements
				switch(p.ChimeraCutOffRange){
					case 1: nonlocal_homo_tiled_zbke2k_2d<0,1><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 2: nonlocal_homo_tiled_zbke2k_2d<0,2><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 3: nonlocal_homo_tiled_zbke2k_2d<0,3><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 4: nonlocal_homo_tiled_zbke2k_2d<0,4><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 5: nonlocal_homo_tiled_zbke2k_2d<0,5><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					default: printf("ChimeraCutOffRange must be < 6! Program Abort!"); exit(1); break;
				}
				break;
			default:
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!");
				exit(EXIT_FAILURE);
				break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 2:
				nblocks.y=(p.ny-1)/o_TileWidth+1;
				nthreads.y=p.blockWidth;
				mem_size=p.blockWidth*p.blockWidth;	// number of elements
				switch(p.ChimeraCutOffRange){
					case 1: nonlocal_homo_tiled_zbke2k_2d<1,1><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 2: nonlocal_homo_tiled_zbke2k_2d<1,2><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 3: nonlocal_homo_tiled_zbke2k_2d<1,3><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 4: nonlocal_homo_tiled_zbke2k_2d<1,4><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 5: nonlocal_homo_tiled_zbke2k_2d<1,5><<<nblocks,nthreads,mem_size*sizeof(Real),s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,p.ny,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					default: printf("ChimeraCutOffRange must be < 6! Program Abort!"); exit(1); break;
				}
				break;
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(EXIT_FAILURE); break;
		}
	}
	
	checkCUDAError("nonlocal_homo_tiled_zbke2k()",__LINE__);
}











void coupling(device_pointers *d, params &p, streams *s, size_t step){
	
	// iterate over all components by changing the offset
	// move offset in array to address different components
	if(step>=p.stepsCouplingStart){
		switch(p.diffusionChoice){
			case 5: 											// nonlocal
				if(p.reactionModel==24 or p.reactionModel==2401 or p.reactionModel==25){
					if(p.use_tiles){ nonlocal_homo_tiled_zbke2k(d,p,s); }
					else{ nonlocal_homo_zbke2k(d,p,s); }
				}
				break;
			case 6:												// tnonlocal, delay; fill history until delaySteps+1: only local dynamics, no coupling
				if(p.delayHistoryUpdateStep>1) step = step/p.delayHistoryUpdateStep;
				
				if(step<p.delayStartSteps+1){
					if(p.reactionModel==24 or p.reactionModel==2401 or p.reactionModel==25){
						if(p.use_tiles){ nonlocal_homo_tiled_zbke2k(d,p,s); }
						else{ nonlocal_homo_zbke2k(d,p,s); }
					}
				}else if(step>=p.delayStartSteps+1){
					if(p.reactionModel==24 or p.reactionModel==2401 or p.reactionModel==25){
						if(p.use_tiles){ nonlocal_delay_homo_tiled_zbke2k(d,p,s); }
						else{ nonlocal_delay_homo_zbke2k(d,p,s); }
					}
				}
				break;
			case 7: break;										// no coupling
			default: printf("Error: diffusionChoice \"%d\" not implemented!\n",p.diffusionChoice); exit(1); break;
		}
	}
	
	checkCUDAError("diffusion()",__LINE__);
}


template <typename T>
__global__ void copyArrays(T *in, T *out, int len){
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len) out[i]=in[i];
}


Real kernelfunction(int i, int j, int i0, int j0, params &p){
	
	Real value=0.0;
	
	// exponential decay
	switch(p.spaceDim){
		case 2:
			value=p.dx*p.dy*exp(-sqrt((i-i0)*(i-i0)*p.dx*p.dx+(j-j0)*(j-j0)*p.dy*p.dy)/p.ChimeraKappa);
			break;
	}
	
	return value;
}

void create_kernel_and_rescale(int &n, params &p, host_pointers *h, modelparams &m){
	
	
	switch(p.diffusionChoice){
		
		// nonlocal BZ chimera coupling: Exp(-d/kappa), d = euclidean distance
		case 5:
		case 6:
			{
			int len=0, i=0, j=0, i0=0, j0=0;
			p.kdia=2*p.ChimeraCutOffRange+1;
			p.kradius = p.ChimeraCutOffRange;
			p.ksum = 0.0;
			switch(p.spaceDim){
				case 2:
					len=2*p.ChimeraCutOffRange+1;
					n=pow(len,2);
					i0=p.ChimeraCutOffRange; j0=i0;
					h->mask = new Real[n];
					for(i=0; i<len; i++){
					for(j=0; j<len; j++){
						h->mask[i+j*len] = kernelfunction(i,j,i0,j0,p);
					}}
					for(i=0; i<n; i++) h->mask[i] *= p.ChimeraK*p.dt;
					break;
			}
			// save kernel in binary data format for later
			std::ofstream dataout;
			dataout.open(p.pthout+"/coupling_kernel.bin",std::ios::binary);
			for(int i=0; i<n; i++) dataout.write((char*) &(h->mask[i]), sizeof(Real));
			dataout.close();
			for(int i=0; i<n; i++) p.ksum += h->mask[i];
			
			}
			break;
		
		// no rescaling
		case 7:					// no coupling
			break;
	}
}


void getArraySize(params &p, int &array_size){
	
	array_size=p.n*sizeof(Real2);
}






void cleanup_GPU(Real *c, device_pointers *d, params &p){
	
	printf("cleanup_GPU\n");
	cudaError_t err;
	
	err = cudaFree(d->k);
	err = cudaFree(d->output);
	err = cudaFree(d->mask);
	if(p.delayFlag==0){ err = cudaFree(d->c); err = cudaFree(d->cnew); }
	else if(p.delayFlag==1){ err = cudaFree(d->c0); }
	
	// DEBUG
	if(err != cudaSuccess){
		printf("Cuda error: %s\n",cudaGetErrorString(err) );
		exit(EXIT_FAILURE);
	}
}


void copy_GPU_to_CPU(device_pointers *d, Real *c, params &p, streams *s){
	
	cudaMemcpy(c,d->output,p.n*sizeof(Real2),cudaMemcpyDeviceToHost);
	
}

// manage copy operation on GPU
void copy_GPU_to_GPU(device_pointers *d, params &p, streams *s){
	
	
	cudaDeviceSynchronize();
	
	
	int warpsize=32;
	dim3 nblocks2((p.n-1)/warpsize+1,1,1);
	dim3 nthreads2(warpsize,1,1);
	
	copyArrays<<<nblocks2,nthreads2,0,s->stream1>>>((Real2*)d->c,(Real2*)d->output,p.n);
	
	checkCUDAError("copyArrays invocation",__LINE__);
	
	cudaDeviceSynchronize();
}



void init_GPU(streams *s, params &p, Real *c, Real *k, device_pointers *d, modelparams &m){
	
	host_pointers h;
	h.c=c;
	int array_size=0;
	Real needed_mem=0.0;		// memory in bytes
	
	// datatype dependent improvements
	#ifdef DOUBLE
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);			// shared memory in 64 bit mode, better for double datatypes
	#endif
	if(p.use_tiles) cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);				// get more shared memory, better for tiling
	
	// DEBUG: available memory
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem,&totalMem);
	printf("available/total GPU memory: %.2f/%.2f\n" ,freeMem/(1024.*1024.),totalMem/(1024.*1024.));
	
	// allocate GPU memory & move data from host to device
	getArraySize(p,array_size);
	if(p.delayFlag==0){												// no delay
		printf("c & cnew size: %.3f MB = states: 2\n",2*array_size/(1024.*1024.));
		needed_mem+=2*array_size;
		if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
		cudaMalloc(&(d->c),array_size);
		cudaMalloc(&(d->cnew),array_size);
		cudaMemcpy(d->c,h.c,array_size,cudaMemcpyHostToDevice);
	}else if(p.delayFlag==1){										// with delay
		size_t history_array_size=array_size*(p.delayStepsMax+1);
		printf("c0 size: %.3f MB = states: %zu\n",history_array_size/(1024.*1024.), p.delayStepsMax+1);
		needed_mem+=history_array_size;
		if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
		cudaMalloc(&(d->c0),history_array_size);
		// init arrays for first step
		d->c = d->c0;
		cudaMemcpy(d->c,h.c,array_size,cudaMemcpyHostToDevice);
		d->cnew = d->c0+p.n*p.ncomponents;
	}else if(p.delayFlag==2){										// with delay, omitStates
		size_t history_array_size=array_size*(p.delayStepsMax+2);
		printf("c0 size: %.3f MB = states: %zu\n",history_array_size/(1024.*1024.), p.delayStepsMax+2);
		needed_mem+=history_array_size;
		if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
		cudaMalloc(&(d->c0),history_array_size);
		// init arrays for first step
		d->c = d->c0+p.n*p.ncomponents;
		cudaMemcpy(d->c,h.c,array_size,cudaMemcpyHostToDevice);
		if(p.delayHistoryUpdateStep % 2 == 0){		// even
			d->cnew = d->c0;
		}else{										// uneven
			d->cnew = d->c0+(1+1)*p.n*p.ncomponents;
		}
	}
	printf("output size: %.3f MB\n",array_size/(1024.*1024.));
	needed_mem+=array_size;
	if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
	cudaMalloc(&(d->output),array_size);
	
	
	// mask for convolutions and rescaling
	int maskSize=0;
	create_kernel_and_rescale(maskSize,p,&h,m);
	if( p.diffusionChoice==5 || p.diffusionChoice==6 ){
		printf("kernel size: %.3f MB\n",maskSize*sizeof(Real)/(1024.*1024.));
		needed_mem+=maskSize*sizeof(Real);
		if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
		cudaMalloc(&(d->mask),maskSize*sizeof(Real));
		cudaMemcpy(d->mask,h.mask,maskSize*sizeof(Real),cudaMemcpyHostToDevice);
	}
	
	// set coupling coefficients' values
	d->coupling_coeffs2.x=m.coupling_coeffs[0]; 
	d->coupling_coeffs2.y=m.coupling_coeffs[1];
	
	
	
	int hetArraySize=p.n;
	printf("het-array size: %.3f MB\n",hetArraySize*sizeof(Real)/(1024.*1024.));
	needed_mem+=hetArraySize*sizeof(Real);
	if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
	cudaMalloc(&(d->k),hetArraySize*sizeof(Real));
	cudaMemcpy(d->k,k,hetArraySize*sizeof(Real),cudaMemcpyHostToDevice);
	
	// info
	cudaMemGetInfo(&freeMem,&totalMem);
	printf("available/total GPU memory: %.2f/%.2f\n" ,freeMem/(1024.*1024.),totalMem/(1024.*1024.));
	printf("total amount of GPU memory required: %.3f MB\n",needed_mem/(1024.*1024.));
	printf("total number of threads per block: %.0f <= 1024?\n",pow(p.blockWidth,p.spaceDim));
	if(pow(p.blockWidth,p.spaceDim)>1024){ printf("Error: Using too many threads per block!"); exit(EXIT_FAILURE);}
	
	checkCUDAError("init_GPU()",__LINE__);
	
}



void rd_dynamics(device_pointers *d, params &p, streams *s, size_t step){
	
	switch(p.delayFlag){
		case 0:											// no delay
			reaction(d,p,s);
			coupling(d,p,s,step);
			swapGPU(d->c,d->cnew);						// issue device pointer swap from host
			break;
		
		case 1:											// with delay
			reaction(d,p,s);
			coupling(d,p,s,step);
			// update pointer positions
			d->c = d->c0 + ((step+1) % (p.delayStepsMax+1))*p.n*p.ncomponents;
			d->cnew = d->c0 + ((step+2) % (p.delayStepsMax+1))*p.n*p.ncomponents;
			
			if(step>=p.delayStartSteps){
				size_t delaySteps=0;
				delaySteps=p.delayStepsMax;
				d->cdelay=d->c0+((step+3) % (delaySteps+1))*p.n*p.ncomponents;
			}
			
			break;
			
		case 2:											// with delay, omit steps for memory
			reaction(d,p,s);
			coupling(d,p,s,step);
			
			// pointer position iteration as update
			{
				int i=(step+1) % p.delayHistoryUpdateStep;
				size_t stepCoarse = (step+1) / p.delayHistoryUpdateStep;
				
				if(i==0){												// move data to next field (step 0)
					// array source: c
					d->c = d->c0 + p.n*p.ncomponents + (stepCoarse % (p.delayStepsMax+1))*p.n*p.ncomponents;
					// array target: cnew
					if(p.delayHistoryUpdateStep % 2 == 0){			// even
						d->cnew = d->c0;
					}else{											// odd
						d->cnew = d->c0 + p.n*p.ncomponents + ((stepCoarse+1) % (p.delayStepsMax+1))*p.n*p.ncomponents;
					}
					// move delay pointer further along
					if(stepCoarse>=p.delayStartSteps){
						size_t delaySteps=0;
						delaySteps=p.delayStepsMax;
						d->cdelay=d->c0+p.n*p.ncomponents + ((stepCoarse+2) % (delaySteps+1))*p.n*p.ncomponents;
					}
					
				}else if(i==1){											// set up start of swap cycle (step 1)
					d->c = d->cnew;
					// array target: cnew
					if(p.delayHistoryUpdateStep % 2 == 0){			// even
						d->cnew = d->c0 + p.n*p.ncomponents + ((stepCoarse+1) % (p.delayStepsMax+1))*p.n*p.ncomponents;
					}else{											// odd
						d->cnew = d->c0;
					}
				}else if(i>1){											// swap cycle (step 2 and more)
					swapGPU(d->c,d->cnew);
				}
			}
			break;
		
		default: printf("Unknown value for delayFlag (rd_dynamics).\n"); break;
	}
}




void solverGPU_2d(Real *c, Real *k, params &p, modelparams &m){
	
	printf("solverGPU_2d\n");
	
	// init
	device_pointers d;
	streams s;
	int untranslatedFlag=1;
	Real *ctemp = (Real *) calloc(p.n*p.ncomponents,sizeof(Real));
	
	// init class for saving
	int nSaveStates=100;
	Safe safe(p,nSaveStates);
	
	// save initial condition, ic=0
	translateArrayOrder(c,ctemp,p,untranslatedFlag);
	safe.save(ctemp,0);
	
	// prepare GPU
	init_GPU(&s,p,c,k,&d,m);
	
	// time loop
	for(size_t step=0; step<p.stepsEnd; step++){
		rd_dynamics(&d,p,&s,step);
		
		if(step>0 and (!(step%p.stepsSaveState))){
			
			// copy
			copy_GPU_to_GPU(&d,p,&s);
			copy_GPU_to_CPU(&d,c,p,&s);
			if(c[0]!=c[0]){ printf("step: %zu, u[0]=%f. Abort!\n",step,c[0]);  exit(EXIT_FAILURE); }
			
			// translate array from concentration major to space major
			translateArrayOrder(c,ctemp,p,untranslatedFlag);
			
			// save
			if(!(step%p.stepsSaveState)){ 
				if(untranslatedFlag){ safe.save(c,step); }
				else{ safe.save(ctemp,step); }
			}
		}
		
		// DEBUG
		checkCUDAError("Loop iteration",__LINE__);
	}
	
	// clean up data
	cleanup_GPU(c,&d,p);
	free(ctemp);
}



void solverGPU(Real *c, Real *k, params &p, modelparams &m){
	
	// serial version
	switch(p.spaceDim){
		case 2: solverGPU_2d(c,k,p,m); break;	// cartesian 2d
	}

}
