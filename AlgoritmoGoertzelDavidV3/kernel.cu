
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuComplex.h>
#include <math.h>
#include <math_constants.h>
#include <iostream>

__global__ void Goertzel(int N, cuFloatComplex *device_x, cuFloatComplex *device_X_k)
{
	int xIndex, yIndex, zIndex, n;

	xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	zIndex = threadIdx.z + blockIdx.z * blockDim.z;

	cuFloatComplex sk1,sk2,skn,ykn;
	cuFloatComplex aux_1,aux_2,aux_3,aux_4,aux_5;
	sk1 = make_cuFloatComplex(0.0,0.0);
	sk2 = make_cuFloatComplex(0.0,0.0);

	for(n = 0; n < N; n++)
	{
		aux_1 = cuCmulf(make_cuFloatComplex((2 * cos(2*CUDART_PI_F*xIndex/N)),0.0),sk1);
		aux_2 = cuCaddf(device_x[n],aux_1);
		aux_3 = cuCsubf(aux_2,sk2);
		skn = aux_3;

		sk2 = sk1;
		sk1 = skn;
	}
	aux_4 = cuCmulf(make_cuFloatComplex((2 * cos(2*CUDART_PI_F*xIndex/N)),0.0),sk1);
	skn = cuCsubf(aux_4,sk2);
	aux_5 = cuCmulf(make_cuFloatComplex(cos((-2*CUDART_PI_F*xIndex)/N), sin((-2*CUDART_PI_F*xIndex)/N)),sk1);
	ykn = cuCsubf(skn,aux_5);
	device_X_k[xIndex] = ykn;
}

int main()
{
	//Ingrese el tamaño del vector de entrada x[n]
	int N = 1000;

	//Numero de iteraciones
	int j;
	int loop = 500;

	//Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();
	
	for(j = 0;j < loop;j++)
	{
		//Comandos necesarios para medir el tiempo de la aplicacion (app)
		float elapsedTime_app;
		cudaEvent_t start_app, stop_app;
		cudaEventCreate(&start_app);
		cudaEventCreate(&stop_app);

		//---------------------------------------------------------------------------------------------
		//Se empieza a medir el tiempo de ejecucion de la aplicacion
		cudaEventRecord(start_app,0);

		//Comandos necesarios para medir el tiempo del kernel (kernel)
		float elapsedTime_kernel;
		cudaEvent_t start_kernel, stop_kernel;
		cudaEventCreate(&start_kernel);
		cudaEventCreate(&stop_kernel);

		//Declaracion de variables
		int i;
		cuFloatComplex *host_x;
		cuFloatComplex *host_X_k;
		cuFloatComplex *device_x;
		cuFloatComplex *device_X_k;
		FILE *time_app;
		FILE *time_kernel;

		//Se crean los archivos binarios donde se guardaran los tiempos
		time_app = fopen("time_app.bin","a+b");
		time_kernel = fopen("time_kernel.bin","a+b");

		//Se reserva memoria en host y device para x
		host_x = (cuFloatComplex*) malloc(sizeof(cuFloatComplex)*N);
		host_X_k = (cuFloatComplex*) malloc(sizeof(cuFloatComplex)*N);
		cudaMalloc((void**)&device_x,sizeof(cuFloatComplex)*N);
		cudaMalloc((void**)&device_X_k,sizeof(cuFloatComplex)*N);

		//Se dan valores al vector de entrada x[n]
		for(i = 0;i < N;i++ )
		{
			host_x[i] = make_cuFloatComplex(rand()%11,rand()%21);
			//host_x[i] = make_cuFloatComplex(i+1,0.0);
		}

		/*
		//Se imprime vector de entrada x[n]
		printf("\n\n---VECTOR DE ENTRADA x[n]---\n");
		for(i = 0;i < N;i++)
		{
			printf("\nx[%d] = (%f) + (%f)\n ",i,cuCrealf(host_x[i]),cuCimagf(host_x[i]));
		}
		*/

		//--------------------------------------------------------------------------------------------
		//Se empieza a medir el tiempo de ejecucion del kernel
		cudaEventRecord(start_kernel,0);
	
		//Se copia el arreglo x[n] a la memoria global de la GPU
		cudaMemcpy(device_x,host_x,sizeof(cuFloatComplex)*N,cudaMemcpyHostToDevice);

		//Variables necesarias para dimensionar el block y grid
		dim3 dimBlock; 
		dim3 dimGrid;
		dimBlock.x = N; dimBlock.y = 1; dimBlock.z = 1;
		dimGrid.x  = 1; dimGrid.y  = 1; dimGrid.z  = 1;
		
		//Se invoca el kernel
		Goertzel<<<dimGrid,dimBlock >>>(N, device_x, device_X_k);
		cudaThreadSynchronize();

		//Se leen los resultados de la GPU
		cudaMemcpy(host_X_k,device_X_k,sizeof(cuFloatComplex)*N,cudaMemcpyDeviceToHost);
	
		//Se termina de medir el tiempo de ejecucion del kernel
		cudaEventRecord(stop_kernel,0);
		cudaEventSynchronize(stop_kernel);
		cudaEventElapsedTime(&elapsedTime_kernel,start_kernel,stop_kernel);
		//fprintf(time_kernel,"%f\n",elapsedTime_kernel);
		fwrite(&elapsedTime_kernel,sizeof(float),1,time_kernel);
		//printf("\n\nTiempo de ejecucion del kernel: %f milisegundos\n\n",elapsedTime_kernel);
		//--------------------------------------------------------------------------------------------

		//Se destruyen los eventos que miden el tiempo del kernel
		cudaEventDestroy(start_kernel);
		cudaEventDestroy(stop_kernel);

		//Se cierra el archivo binario "time_kernel"
		fclose(time_kernel);

		/*
		//Se imprime vector de salida X[k]
		printf("\n\n---VECTOR DE SALIDA X[k]---\n");
		for(i = 0;i < N;i++)
		{
			printf("\nX[%d] = (%f) + (%f)\n ",i,cuCrealf(host_X_k[i]),cuCimagf(host_X_k[i]));
		}
		*/

		//Se liberan memorias del Host y Device
		free(host_x);
		free(host_X_k);
		cudaFree(device_x);
		cudaFree(device_X_k);

		//Comandos necesarios para medir el tiempo de la aplicacion (app)
		cudaEventRecord(stop_app,0);
		cudaEventSynchronize(stop_app);
		cudaEventElapsedTime(&elapsedTime_app,start_app,stop_app);
		fwrite(&elapsedTime_app,sizeof(float),1,time_app);
		//fprintf(time_app,"%f\n",elapsedTime_app);
		//printf("\n\nTiempo de ejecucion de la aplicacion: %f milisegundos\n\n",elapsedTime_app);
		//--------------------------------------------------------------------------------------------

		//Se destruyen los eventos que miden el tiempo de la aplicacion
		cudaEventDestroy(start_app);
		cudaEventDestroy(stop_app);

		//Se cierra el archivo binario "time_app"
		fclose(time_app);
	}
}

