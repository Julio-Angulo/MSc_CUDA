//Calculo de la FFT 1D usando "fftwf_plan_dft_1d".

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cufftw.h>


#define SIGNAL_SIZE 10



int main()
{
	int i;
	cuFloatComplex *h_xn;
	cuFloatComplex *h_Xk;
	fftwf_complex *in,*out;

	//Se reserva memoria para h_xn en el host
	h_xn = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*SIGNAL_SIZE);

	//Se reserva memoria para h_Xk en el host
	h_Xk = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*SIGNAL_SIZE);

	//Se dan valores a x[n]
	for(i=0;i<SIGNAL_SIZE;i++)
	{
		
			//h_xn[i] = make_cuFloatComplex((float)(rand()%11),(float)(rand()%21));
			h_xn[i] = make_cuFloatComplex((float)(i+1),(float)(0.0));
	}

	//Se imprimen los valores de entrada x[n]
	printf("\n---ELEMENTOS DE ENTRADA x[n]---\n\n");
	for(i = 0; i<SIGNAL_SIZE;i++)
	{
		printf(" x[%d] = (%f) + (%f)\n",i,cuCrealf(h_xn[i]),cuCimagf(h_xn[i]));
	}

	//Se reserva memoria para "in" en el device
	cudaMalloc((void**)&in,sizeof(cufftComplex)*SIGNAL_SIZE);

	//Se reserva memoria para "out" en el device
	cudaMalloc((void**)&out,sizeof(cufftComplex)*SIGNAL_SIZE);

	//Se copian los datos de h_xn >>> in 
	cudaMemcpy(in,h_xn,sizeof(cuFloatComplex)*SIGNAL_SIZE,cudaMemcpyHostToDevice);

	//CUFFT plan
	fftw_plan plan;
	//cufftHandle plan;
	plan = fftwf_plan_dft_1d(SIGNAL_SIZE,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
	//cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 1);

	//Ejecucion de la fft
	fftwf_execute(plan);

	//Se copian los datos de out >>> h_Xk
	cudaMemcpy(h_Xk,out,sizeof(cufftComplex)*SIGNAL_SIZE,cudaMemcpyDeviceToHost);

	//Se imprimen los valores de salida X[k]
	printf("\n---ELEMENTOS DE SALIDA X[k]---\n\n");
	for(i = 0; i<SIGNAL_SIZE;i++)
	{
		printf(" X[%d] = (%f) + (%f)\n",i,cuCrealf(h_Xk[i]),cuCimagf(h_Xk[i]));
	}

	//Se destruye el plan
	fftwf_destroy_plan(plan);

	//Se liberan memorias
	free(h_xn);
	free(h_Xk);
	cudaFree(in);
	cudaFree(out);
}
