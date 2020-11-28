//Calculo de la FFT 3D utilizando la funcion "fftwf_plan_dft_3d";

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cufftw.h>

#define RENGLONES 3
#define COLUMNAS 4
#define PROFUNDIDAD 2


int main()
{
	int i,j,k;
	cuFloatComplex *h_xn;
	cuFloatComplex *h_Xk;
	fftwf_complex *in,*out;

	//Se reserva memoria para h_xn en el host
	h_xn = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*COLUMNAS*RENGLONES*PROFUNDIDAD);

	//Se reserva memoria para h_Xk en el host
	h_Xk = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*COLUMNAS*RENGLONES*PROFUNDIDAD);

	//Se dan valores a x[n]
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<RENGLONES;i++)
		{
			for(j=0;j<COLUMNAS;j++)
			{
				//h_xn[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j] = make_cuFloatComplex((float)(rand()%11),(float)(rand()%21));
				h_xn[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j] = make_cuFloatComplex((float)(((k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j) + 1),(float)(0.0));
			}
		}
	}
	//Se imprimen los valores de entrada x[n]
	printf("\n---ELEMENTOS DE ENTRADA x[n]---\n\n");
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<RENGLONES;i++)
		{
			for(j=0;j<COLUMNAS;j++)
			{
				printf(" (%f) + (%f)",cuCrealf(h_xn[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j]),cuCimagf(h_xn[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}
	//Se reserva memoria para "in" en el device
	cudaMalloc((void**)&in,sizeof(cufftComplex)*COLUMNAS*RENGLONES*PROFUNDIDAD);

	//Se reserva memoria para "out" en el device
	cudaMalloc((void**)&out,sizeof(cufftComplex)*COLUMNAS*RENGLONES*PROFUNDIDAD);

	//Se copian los datos de h_xn >>> in 
	cudaMemcpy(in,h_xn,sizeof(cuFloatComplex)*COLUMNAS*RENGLONES*PROFUNDIDAD,cudaMemcpyHostToDevice);

	//CUFFT plan
	fftwf_plan plan;
	plan = fftwf_plan_dft_3d(PROFUNDIDAD,RENGLONES,COLUMNAS,in,out,FFTW_FORWARD,FFTW_ESTIMATE);

	

	//Ejecucion de la fft
	fftwf_execute(plan);

	//Se copian los datos de out >>> h_Xk
	cudaMemcpy(h_Xk,out,sizeof(cufftComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD,cudaMemcpyDeviceToHost);

	//Se imprimen los valores de salida X[k]
	printf("\n---ELEMENTOS DE SALIDA X[k]---\n\n");
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<RENGLONES;i++)
		{
			for(j=0;j<COLUMNAS;j++)
			{
				printf(" (%f) + (%f)",cuCrealf(h_Xk[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j]),cuCimagf(h_Xk[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}
	//Se destruye el plan
	fftwf_destroy_plan(plan);

	//Se liberan memorias
	free(h_xn);
	free(h_Xk);
	cudaFree(in);
	cudaFree(out);
}
