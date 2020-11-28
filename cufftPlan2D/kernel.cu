
//Calculo de la FFT 2D utilizando la funcion cufftPlan2D();

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>


#define RENGLONES 3
#define COLUMNAS 3


int main()
{
	int i,j;
	cuFloatComplex *h_xn;
	cuFloatComplex *h_Xk;
	cufftComplex *in,*out;

	//Se reserva memoria para h_xn en el host
	h_xn = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*COLUMNAS*RENGLONES);

	//Se reserva memoria para h_Xk en el host
	h_Xk = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*COLUMNAS*RENGLONES);

	//Se dan valores a x[n]
	for(i=0;i<RENGLONES;i++)
	{
		for(j=0;j<COLUMNAS;j++)
		{
			//h_xn[(i*COLUMNAS)+j] = make_cuFloatComplex((float)(rand()%11),(float)(rand()%21));
			h_xn[(i*COLUMNAS)+j] = make_cuFloatComplex((float)(((i*COLUMNAS)+j) + 1),(float)(0.0));
		}
	}

	//Se imprimen los valores de entrada x[n]
	printf("\n---ELEMENTOS DE ENTRADA x[n]---\n\n");
	for(i=0;i<RENGLONES;i++)
	{
		for(j=0;j<COLUMNAS;j++)
		{
			printf(" (%f) + (%f)",cuCrealf(h_xn[(i*COLUMNAS)+j]),cuCimagf(h_xn[(i*COLUMNAS)+j]));
		}
		printf("\n");
	}

	//Se reserva memoria para "in" en el device
	cudaMalloc((void**)&in,sizeof(cufftComplex)*COLUMNAS*RENGLONES);

	//Se reserva memoria para "out" en el device
	cudaMalloc((void**)&out,sizeof(cufftComplex)*COLUMNAS*RENGLONES);

	//Se copian los datos de h_xn >>> in 
	cudaMemcpy(in,h_xn,sizeof(cuFloatComplex)*COLUMNAS*RENGLONES,cudaMemcpyHostToDevice);

	//CUFFT plan
	cufftHandle plan;
	cufftPlan2d(&plan,RENGLONES,COLUMNAS, CUFFT_C2C);

	

	//Ejecucion de la fft
	cufftExecC2C(plan,in,out,CUFFT_FORWARD);

	//Se copian los datos de out >>> h_Xk
	cudaMemcpy(h_Xk,out,sizeof(cufftComplex)*RENGLONES*COLUMNAS,cudaMemcpyDeviceToHost);

	//Se imprimen los valores de salida X[k]
	printf("\n---ELEMENTOS DE SALIDA X[k]---\n\n");
	for(i=0;i<RENGLONES;i++)
	{
		for(j=0;j<COLUMNAS;j++)
		{
			printf(" (%f) + (%f)",cuCrealf(h_Xk[(i*COLUMNAS)+j]),cuCimagf(h_Xk[(i*COLUMNAS)+j]));
		}
		printf("\n");
	}

	//Se destruye el plan
	cufftDestroy(plan);

	//Se liberan memorias
	free(h_xn);
	free(h_Xk);
	cudaFree(in);
	cudaFree(out);
}

