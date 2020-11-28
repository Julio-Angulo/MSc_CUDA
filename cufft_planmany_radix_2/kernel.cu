//Medición de tiempo para ejecutar una FFT 1D, radix-2. Se usará el comando "cufftPlanMany"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>
#include <cufftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <math_constants.h>
#include <iostream>

//#define RENGLONES 8192
#define COLUMNAS  1
#define PROFUNDIDAD  1



int main()
{
	int loop;
	//Número de iteraciones
	loop = 500;
	
	//Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();
	
	int a,b,RENGLONES;
	//Incrementa la longitud 
	for(b=0;b<=13;b++)
	{
		RENGLONES = pow(2,b);
		//printf("\n\n RENGLONES = %d \n\n",RENGLONES);
		for(a=1;a<=loop;a++)
		{
			//Comandos necesarios para medir el tiempo de la aplicacion (app)
			float elapsedTime_app;
			cudaEvent_t start_app, stop_app;
			cudaEventCreate(&start_app);
			cudaEventCreate(&stop_app);
		
			int i,j,k;
			int n[1] = {RENGLONES};
			int inembed[1] = {RENGLONES};
			int onembed[1] = {RENGLONES};
			cuFloatComplex *h_xn;
			cuFloatComplex *h_xn_trans;
			cuFloatComplex *h_Xk;
			cuFloatComplex *h_Xk_trans;
			cufftComplex *in,*out;
			//fftwf_complex *in,*out;
			FILE *time_fftw;
		
			//Se crean los archivos binarios donde se guardaran los tiempos
			time_fftw= fopen("time_cufft_radix2.bin","a+b");
		

			//Se reserva memoria para h_xn en el host
			h_xn = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

			//Se reserva memoria para h_xn_trans en el host
			h_xn_trans = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

			//Se reserva memoria para h_Xk en el host
			h_Xk = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

			//Se reserva memoria para h_Xk_trans en el host
			h_Xk_trans = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD);

		

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

			/*
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
			*/

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

			//--------------------------------------------------------------------------------------------------------------------
			//Se empieza a medir el tiempo de ejecucion de la aplicacion
			cudaEventRecord(start_app,0);
		
			//Se copian los datos de h_xn_trans >>> in 
			cudaMemcpy(in,h_xn_trans,sizeof(cuFloatComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD,cudaMemcpyHostToDevice);

			//CUFFT plan
			//fftwf_plan plan;
			cufftHandle plan;
			cufftPlanMany(&plan,1,n,inembed,1,1,onembed,1,1,CUFFT_C2C,1);
			//plan = fftwf_plan_many_dft(1,n,1,in,inembed,1,1,out,onembed,1,1,FFTW_FORWARD,FFTW_MEASURE);

			//Ejecucion de la fft
			//fftwf_execute(plan);
			cufftExecC2C(plan,in,out,CUFFT_FORWARD);

			//Se copian los datos de out >>> h_Xk
			cudaMemcpy(h_Xk_trans,out,sizeof(cufftComplex)*RENGLONES*COLUMNAS*PROFUNDIDAD,cudaMemcpyDeviceToHost);

			//Comandos necesarios para medir el tiempo de la aplicacion (app)
			cudaEventRecord(stop_app,0);
			cudaEventSynchronize(stop_app);
			cudaEventElapsedTime(&elapsedTime_app,start_app,stop_app);
			fwrite(&elapsedTime_app,sizeof(float),1,time_fftw);
			//fprintf(time_fftw,"%f\n",elapsedTime_app);
			//printf("\n\nTiempo de ejecucion de la aplicacion: %f milisegundos\n\n",elapsedTime_app);
			//-------------------------------------------------------------------------------------------------------------------

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

			/*
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
			*/


			//Se destruye el plan
			//fftwf_destroy_plan(plan);
			cufftDestroy(plan);

			//Se liberan memorias
			free(h_xn);
			free(h_Xk);
			free(h_xn_trans);
			free(h_Xk_trans);
			cudaFree(in);
			cudaFree(out);

			//Se destruyen los eventos que miden el tiempo de la aplicacion
			cudaEventDestroy(start_app);
			cudaEventDestroy(stop_app);

			//Se cierra el archivo binario "time_fftw"
			fclose(time_fftw);
		}
	}
}


