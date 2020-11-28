//Calculo de la FFT 3D usando "cufftPlanMany"----->>>>sin transpuesta

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cufftw.h>


#define RENGLONES 2
#define COLUMNAS 3
#define PROFUNDIDAD 2


int main()
{
	int i,j,k;
	int n[1] = {COLUMNAS};
	//int n[3] = {PROFUNDIDAD,COLUMNAS,RENGLONES};
	int inembed[1] = {COLUMNAS};
	//int inembed[3] = {PROFUNDIDAD,COLUMNAS,RENGLONES};
	int onembed[1] = {COLUMNAS};
	//int onembed[3] = {PROFUNDIDAD,COLUMNAS,RENGLONES};
	cuFloatComplex *h_xn;
	//cuFloatComplex *h_xn_trans;
	cuFloatComplex *h_Xk;
	//cuFloatComplex *h_Xk_trans;
	//fftwf_complex *in,*out;
	cufftComplex *in,*out;

	//Se reserva memoria para h_xn en el host
	h_xn = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

	//Se reserva memoria para h_xn_trans en el host
	//h_xn_trans = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

	//Se reserva memoria para h_Xk en el host
	h_Xk = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

	//Se reserva memoria para h_Xk_trans en el host
	//h_Xk_trans = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

	//Se dan valores a x[n]
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<RENGLONES;i++)
		{
			for(j=0;j<COLUMNAS;j++)
			{
				//h_xn[i] = make_cuFloatComplex((float)(rand()%11),(float)(rand()%21));
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
				printf(" (%f) + (%f) ",cuCrealf(h_xn[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j]),cuCimagf(h_xn[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j]));
			}
			printf("\n");	
		}
		printf("\n\n");
	}
	/*
	//Se saca la transpuesta del arreglo tridimensional "h_xn"
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<COLUMNAS;i++)
		{
			for(j=0;j<RENGLONES;j++)
			{
				h_xn_trans[(k*RENGLONES*COLUMNAS)+(i*RENGLONES)+j] = make_cuFloatComplex(cuCrealf(h_xn[(k*RENGLONES*COLUMNAS)+(j*COLUMNAS)+i]),cuCimagf(h_xn[(k*RENGLONES*COLUMNAS)+(j*COLUMNAS)+i]));
			}
			
		}
	}
	*/
	/*
	//Se imprimen los valores de entrada x[n] (matriz transpuesta)
	printf("\n---ELEMENTOS DE ENTRADA x[n] (Matriz transpuesta)---\n\n");
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<COLUMNAS;i++)
		{
			for(j=0;j<RENGLONES;j++)
			{
				printf(" (%f) + (%f) ",cuCrealf(h_xn_trans[(k*RENGLONES*COLUMNAS)+(i*RENGLONES)+j]),cuCimagf(h_xn_trans[(k*RENGLONES*COLUMNAS)+(i*RENGLONES)+j]));
			}
			printf("\n");	
		}
		printf("\n\n");
	}
	*/


	//Se reserva memoria para "in" en el device
	cudaMalloc((void**)&in,sizeof(cufftComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

	//Se reserva memoria para "out" en el device
	cudaMalloc((void**)&out,sizeof(cufftComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

	//Se copian los datos de h_xn >>> in 
	cudaMemcpy(in,h_xn,sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD,cudaMemcpyHostToDevice);

	//CUFFT plan
	//fftwf_plan plan;
	cufftHandle plan;
	cufftPlanMany(&plan,1,n,inembed,1,COLUMNAS,onembed,1,COLUMNAS,CUFFT_C2C,RENGLONES*PROFUNDIDAD);
	
	//plan = fftwf_plan_dft(1,n,in,out,FFTW_FORWARD,FFTW_ESTIMATE);

	//Ejecucion de la fft
	//fftwf_execute(plan);
	cufftExecC2C(plan,in,out,CUFFT_FORWARD);
	

	//Se copian los datos de out >>> h_Xk
	cudaMemcpy(h_Xk,out,sizeof(cufftComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD,cudaMemcpyDeviceToHost);

	/*
	//Se imprimen los valores de salida X[k] (Matriz transpuesta h_Xk_trans)
	printf("\n---ELEMENTOS DE SALIDA X[k]---\n\n");
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<COLUMNAS;i++)
		{
			for(j=0;j<RENGLONES;j++)
			{
				printf(" (%f) + (%f) ",cuCrealf(h_Xk_trans[(k*RENGLONES*COLUMNAS)+(i*RENGLONES)+j]),cuCimagf(h_Xk_trans[(k*RENGLONES*COLUMNAS)+(i*RENGLONES)+j]));
			}
			printf("\n");	
		}
		printf("\n\n");
	}
	*/
	/*
	//Se saca la transpuesta del arreglo tridimensional "h_Xk_trans"
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<COLUMNAS;i++)
		{
			for(j=0;j<RENGLONES;j++)
			{
				h_Xk[(k*RENGLONES*COLUMNAS)+(j*COLUMNAS)+i] = make_cuFloatComplex(cuCrealf(h_Xk_trans[(k*RENGLONES*COLUMNAS)+(i*RENGLONES)+j]),cuCimagf(h_Xk_trans[(k*RENGLONES*COLUMNAS)+(i*RENGLONES)+j]));
			}
			
		}
	}
	*/
	//Se imprimen los valores de salida X[k] 
	printf("\n---ELEMENTOS DE SALIDA X[k]---\n\n");
	for(k=0;k<PROFUNDIDAD;k++)
	{
		for(i=0;i<RENGLONES;i++)
		{
			for(j=0;j<COLUMNAS;j++)
			{
				printf(" (%f) + (%f) ",cuCrealf(h_Xk[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j]),cuCimagf(h_Xk[(k*RENGLONES*COLUMNAS)+(i*COLUMNAS)+j]));
			}
			printf("\n");	
		}
		printf("\n\n");
	}



	//Se destruye el plan
	//fftwf_destroy_plan(plan);
	cufftDestroy(plan);

	//Se liberan memorias
	free(h_xn);
	free(h_Xk);
	//free(h_xn_trans);
	//free(h_Xk_trans);
	cudaFree(in);
	cudaFree(out);
}
