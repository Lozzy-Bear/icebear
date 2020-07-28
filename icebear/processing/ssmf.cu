#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
/*
Conversion of Spread Spectrum matched filter and decimation (ssmf.c)
into CUDA code to increase analysis speed.
Draven Galeschuk - June 19/2018

TO DO:
 - Add error checks, especcially for out of bounds memmory access(maybe not nessisary?)
 - See if decimation and matched filter can be implmented differently
*/

/*
To run this code with the CUDA fft functions, the folowing actions are required:

LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64
export LD_LIBRARY_PATH=/usr/local/cuda0.1/lib64> ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

Note: If the above commands do not work, try the following 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

Note: The first two commands only need to be run once per session.

To compile:
################################################################################################################################################
#                                                                                                                                              #
#           nvcc -Xcompiler -fPIC -shared -o libssmf.so ssmf.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcufft              #
#                                                                                                                                              #
################################################################################################################################################
*/


/* Kernel - This function is run on GPU
ssmf code loops over three variables, based on the block and thead dimensions. Decimation rate 
is assumed to be 200. Result stores real and imaginary combinations of measasurement and code
data in a result matrix. 
@meas - complex type array of measuremnts (np.complex64 in python)
@code - "	"	"	"     (must be np.comlex64, may chagned to float 32 if .cu file is adjusted)
@result = a 2000 x 100 complex value matrix
*/
 

/* NEED TO DO 
       -> Adjust code to perform decimation to any fdec value
       -> Adjust code to accept any defined array sizes (not limited to 20000)
       -> Set up code to dectect GPU capabilities and adjust appropriately 

*/


// Decimation and filtering function. Computes a decimation at a rate of 200
__global__ void ssmf_kernel(cufftComplex *meas, cufftComplex *code, cufftComplex *result, int a_shift)
{
   // Allocates shared memory arrays inside the block
   __shared__ cufftComplex smeas[100];

   // indexing Yay!
   int rg = blockIdx.z;
   int cid = threadIdx.x;
   int ti = blockIdx.y;
   int di = threadIdx.x + blockIdx.y * 200;
   int shift = gridDim.y * blockDim.y;

   // Store a set of values to be desimated in shared memory
   // Perform a reduction on the values heald in smeas
   // Multiply each measurement by the coresponding code value
   // Sum all values in smeas and store in the first element
   // summed by sequential adressing.
   if(cid<100){
      smeas[cid].x = meas[di+rg+100+a_shift*20000].x*code[di+100].x + meas[di+rg+a_shift*20000].x*code[di].x;
      smeas[cid].y = meas[di+rg+100+a_shift*20000].y*code[di+100].x + meas[di+rg+a_shift*20000].y*code[di].x;
   }

   __syncthreads();

   if(cid<50){
      smeas[cid].x += smeas[cid+50].x;
      smeas[cid].y += smeas[cid+50].y;
   }

   __syncthreads();

   if(cid<25){
      smeas[cid].x += smeas[cid+25].x;
      smeas[cid].y += smeas[cid+25].y;
   }

   __syncthreads();

   if(cid<5){
      smeas[cid].x = smeas[cid*5].x + smeas[cid*5+1].x + smeas[cid*5+2].x + smeas[cid*5+3].x + smeas[cid*5+4].x;
      smeas[cid].y = smeas[cid*5].y + smeas[cid*5+1].y + smeas[cid*5+2].y + smeas[cid*5+3].y + smeas[cid*5+4].y;
   }

   __syncthreads();

   if(cid<1){
      smeas[cid].x += smeas[cid*5+1].x + smeas[cid*5+2].x + smeas[cid*5+3].x + smeas[cid*5+4].x;
      smeas[cid].y += smeas[cid*5+1].y + smeas[cid*5+2].y + smeas[cid*5+3].y + smeas[cid*5+4].y;
   }

   // Store reduced value in appropriate element of the result matrix
   if(cid == 0){
      result[rg*shift + ti].x = smeas[0].x;
      result[rg*shift + ti].y = smeas[0].y;
   }

}



// Perfrom conjugate multiplication between two matricies ie: arr1 * conj(arr2)
// This is cross correlation of arr1 and arr2 in fourier space
__global__ void conj_kernel(cufftComplex *arr1, cufftComplex *arr2, cufftComplex *res, cufftComplex *var_temp, int a_shift, int avg){

   //Define thread variable i and intermediate float variables
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   float xr;
   float xi;
   float yri;
   float yir;
   float x;
   float y;

   //Required if statement to eclude threads of index greater than number of elements (result of base 2 thread requirement)
   if(i<200000){
      xr = arr1[i].x * arr2[i].x;	//part of new real computed from real parts of arr
      xi = arr1[i].y * arr2[i].y;	//part of new real computed from imag parts of arr
      yri = arr1[i].x * arr2[i].y;	//part of new imag computed from arr1 real and arr2 imag
      yir = arr1[i].y * arr2[i].x;	//part of new imag computed from arr1 imag and arr2 real

      x = xr + xi;	//New real value for index
      y = yir - yri;	//New imaginary value for index


      //Floating point errors cause non-zero results where a zero is expected, ie: arr1 == arr2
      //The following if statements attempt to correct for these errors by checking ratios
      if(abs(x/xr)<0.000001 && abs(x/xi)<0.000001){
         res[i].x += 0;
	 var_temp[i+a_shift*200000].x = 0;
      }else{
         res[i].x += x/avg;//arr1[i].x * arr2[i].x + arr1[i].y * arr2[i].y;
	 var_temp[i+a_shift*200000].x = x;
      }

      if(abs(y/yir)<0.000001 && abs(y/yri)<0.000001){
         res[i].y += 0;
	 var_temp[i+a_shift*200000].y = 0;
      }else{
         res[i].y += y/avg;//arr1[i].y * arr2[i].x - arr1[i].x * arr2[i].y;
	 var_temp[i+a_shift*200000].y = y;
      }
   }
}


__global__ void var_kernel( cufftComplex *res, cufftComplex *var_temp, cufftComplex *var, int avg){

   //Define thread variable i and intermediate float variables
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   float tempx = 0.0;
   float tempy = 0.0;

   for (int a_shift=0; a_shift<avg; a_shift++){
      tempx += (var_temp[i+a_shift*200000].x - res[i].x) * (var_temp[i+a_shift*200000].x - res[i].x);
      tempy += (var_temp[i+a_shift*200000].y - res[i].y) * (var_temp[i+a_shift*200000].y - res[i].y);
   }

   var[i].x = sqrtf(tempx/avg);
   var[i].y = sqrtf(tempy/avg);

}


/* Main - Prepares Cuda memory and launches Kernel */

extern "C" {
void ssmf(cufftComplex *meas1, cufftComplex *meas2, cufftComplex *code, cufftComplex *result, cufftComplex *var, size_t measlen, size_t codelen, size_t size, int avg, int check)
{
   // Memory management for the kernnels
   cufftHandle plan;

   // define sizes for each array type
   size_t size_m = measlen * sizeof(cufftComplex);
   size_t size_c = codelen * sizeof(cufftComplex);
   size_t size_out = size * sizeof(cufftComplex);
   size_t var_arr = avg*size * sizeof(cufftComplex);
   int n[1];

   n[0] = 100;

   // Build device pointers
   cufftComplex *d_meas1, *d_meas2, *d_temp1, *d_temp2, *d_code, *d_res, *d_var, *d_var_temp, *smeas;
   cudaMalloc((void **) &d_meas1, size_m);
   cudaMalloc((void **) &d_meas2, size_m);
   cudaMalloc((void **) &d_temp1, size_out);
   cudaMalloc((void **) &d_temp2, size_out);
   cudaMalloc((void **) &d_code, size_c);
   cudaMalloc((void **) &d_res, size_out);
   cudaMalloc((void **) &d_var, size_out);
   cudaMalloc((void **) &d_var_temp, var_arr);
   cudaMalloc((void **) &smeas, 100*sizeof(cufftComplex));
   
   // Assign device pointer values
   cudaMemcpy(d_meas1, meas1, size_m, cudaMemcpyHostToDevice);
   cudaMemcpy(d_meas2, meas2, size_m, cudaMemcpyHostToDevice);
   cudaMemcpy(d_code, code, size_c, cudaMemcpyHostToDevice);
   cudaMemcpy(d_res, result, size_out, cudaMemcpyHostToDevice);
   cudaMemcpy(d_var, var, size_out, cudaMemcpyHostToDevice); 

   // Threads must be base 2, 128 is fist factor greater than 100
   // For thread block limits, see CUDA Toolkit Documentation
   dim3 threadsPerBlock(128, 1, 1);
   // Each block calculates an element of the result matrix
   dim3 blocksPerGrid(1, 100, 2000);
   cufftPlanMany(&plan, 1, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 2000);

   // If passed check == 0, performing single ssmf - ie: decimation and filter
   // If passed check == 1, performing cross correlation ssmfx
   if(check==0){
      for(int i=0; i<avg; i++){
         ssmf_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_meas1, d_code, d_res, i);
         cufftExecC2C(plan, d_res, d_res, CUFFT_FORWARD);
      }
   }else{
      for(int i=0; i<avg; i++){
         // Launch Kernel: Each block will perform 1 decimation, so 200 threads are needed per block
         // threadsPerBlock is the first power of 2 greater than loop parameter halved (since fist
         // command does a read and sum)

         ssmf_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_meas1, d_code, d_temp1, i);
         ssmf_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_meas2, d_code, d_temp2, i);

         //Perform fft 
         cufftExecC2C(plan, d_temp1, d_temp1, CUFFT_FORWARD);
         cufftExecC2C(plan, d_temp2, d_temp2, CUFFT_FORWARD);

         //Perform Cross Correlation
         conj_kernel<<<512,391>>>(d_temp1, d_temp2, d_res, d_var_temp, i, avg);
      }

      //Calculate Variance
      var_kernel<<<512,391>>>(d_res, d_var_temp, d_var, avg);
   }

   // Retreive result data
   cudaMemcpy(result, d_res, size_out, cudaMemcpyDeviceToHost);
   cudaMemcpy(var, d_var, size_out, cudaMemcpyDeviceToHost);
   
   // Free up Device memory
   cudaFree(d_meas1);
   cudaFree(d_meas2);
   cudaFree(d_temp1);
   cudaFree(d_temp2);
   cudaFree(d_code);
   cudaFree(d_res);
   cudaFree(d_var);
   cudaFree(d_var_temp);
   cudaFree(smeas);
   cufftDestroy(plan);
}
}

/*
// A test code for dubgging.
int main(void){
   int N = 2000;
   int len = 20000;
   int dec = 200;
   int size = len * N / dec;

   cufftComplex *meas = new cufftComplex[len + N];
   cufftComplex *code = new cufftComplex[len];
   cufftComplex *result = new cufftComplex[size];

   for (int i = 0; i< len+N; i++){
      meas[i].x = 2.0f;
      meas[i].y = 1.0f;
   }
   for (int i= 0; i< len; i++){
      code[i].x = 1.0f;
      code[i].y = 0.0f;
   }
   for (int i = 0; i< size; i++){
      result[i].x = 0.0f;
      result[i].y = 0.0f;
   }
   
   printf("Starting reduction\n\n");
   printf("Meas:\n");
   for(int i=0; i < 20; i++){
      printf("(%f, %f),  ", meas[i].x, meas[i].y);
   }
   printf("\n\n %d, %d, %d", sizeof(cufftComplex), sizeof(float), sizeof(int));
   
   printf("\n\nCode:\n");
   for(int i=0; i < 5; i++){
      printf("(%f, %f),  ", code[i].x, code[i].y);
   }
   ssmf(meas, code, result, len+N, len, size);
   printf("\n\nResult:\n");
   for(int i=0; i < 20; i++){
      printf("@%d=(%f, %f),\t", i, result[i].x, result[i].y);
   }

   printf("\n\nReduction Finished: Exiting...\n");
}*/
