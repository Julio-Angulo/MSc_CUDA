///VERSION SECUENCIAL CUDA 1 THREAD
///Este programa encuentra el algoritmo QR de una matriz A
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuComplex.h>
#include <math.h>
#include <math_constants.h>
#include <iostream>

//2D surfaces
surface<void, cudaSurfaceType2D> A_surface;


//Kernel
__global__ void QR_kernel(unsigned long rows,unsigned long columns)
{
	float r,c,s,temp,data_1,data_2,data_3,data_4,aux_1,aux_2;
	int i,j,k;
	for(k = 0;k <= columns-1;k++)
	{
		for(i = rows-1;i >= k+1;i--)
		{
			surf2Dread(&data_1,A_surface,(k*4),i-1);
			surf2Dread(&data_2,A_surface,(k*4),i);
			r = sqrt(pow(data_1,2) + pow(data_2,2));
			c = data_1/r;
			s = data_2/r;

			for(j = k; j <= columns-1; j++)
			{
				surf2Dread(&data_3,A_surface,(j*4),i-1);
				surf2Dread(&data_4,A_surface,(j*4),i);
				temp = data_3;
				aux_1 = c*temp + s*data_4;
				aux_2 = -s*temp + c*data_4;
				surf2Dwrite(aux_1,A_surface,(j*4),i-1);
				surf2Dwrite(aux_2,A_surface,(j*4),i);

			}
			

		}
	}

}

int main()
{
	int i,j,loop;
	unsigned long N,M,k,r;
	float suma;

	//Numero de iteraciones
	loop = 100;

	//Número de muestras
	const int muestras = 6;
	
	float promedio[muestras];

	///Se crean los archivos binarios donde se guardarán los datos
    FILE *da;
    da = fopen("Algoritmo_QR_CUDA_1_thread.bin","a+b"); //Crea o sobre escribe archivo

	//Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();

	N=1;
	for(i=1;i<=muestras;i++)
    {
        ///Número de columnas de la matriz "A"
        N = pow(2,i*2);

		///Número de renglones de la matriz "A"
		M=N;
    
        suma=0.0;
        for(j=0;j<loop;j++)
        {
			//Comandos necesarios para medir el tiempo
			float elapsedTime_app;
			cudaEvent_t start_app, stop_app;
			cudaEventCreate(&start_app);
			cudaEventCreate(&stop_app);

			//Declaración de variables
            float* A_host;
			float* R_host;

            //Se reserva memoria en host y device para la matriz A
            A_host = (float*) malloc(sizeof(float)*N*M);
            R_host = (float*) malloc(sizeof(float)*N*M);
           
			//Se dan valores a la matriz "A"
			for(k=0;k<M;k++)
            {
				for(r=0;r<N;r++)
				{
					A_host[r + (k*N)]=rand()%10;
					//A_host[r + (k*N)]= (k+1) + (r+1);
				}
            }

			/*
			//Se imprime la matriz "A"
			printf("\n\n---MATRIZ A---\n\n");
			for(k=0;k<M;k++)
            {
				for(r=0;r<N;r++)
				{
					printf(" %f ",A_host[r + (k*N)]);
				}
				printf("\n");
            }
			printf("\n");
			*/

			//Allocate CUDA arrays in device memory
			cudaChannelFormatDesc channelDesc;
			channelDesc = cudaCreateChannelDesc<int>();
			cudaArray* A_array;
			
			cudaMallocArray(&A_array,&channelDesc,N,M,cudaArraySurfaceLoadStore);
			

			//Copy to device memory some data located at address A_host in host memory
			cudaMemcpyToArray(A_array,0,0,A_host,N*M*sizeof(float),cudaMemcpyHostToDevice);

			//Bind the arrays to the surface references
			cudaBindSurfaceToArray(A_surface,A_array);
			

			//Dimensionamiento del grid para la función kernel
			//Dimensionamiento del Grid
			dim3 gridDim(1,1,1);
			//Dimensionamiento del block
			dim3 blockDim(1,1,1);

			//---------------------------------------------------------------------------------------------
			//Se empieza a medir el tiempo de ejecucion de la aplicacion
			cudaEventRecord(start_app,0);
			
			//Lanzamiento del kernel
			QR_kernel<<<gridDim,blockDim >>>(M,N);
			cudaThreadSynchronize();

			//---------------------------------------------------------------------------------------------
			//Comandos necesarios para medir el tiempo de la aplicacion (app)
			cudaEventRecord(stop_app,0);
			cudaEventSynchronize(stop_app);
			cudaEventElapsedTime(&elapsedTime_app,start_app,stop_app);

			//Se leen los resultados de la GPU
			cudaMemcpyFromArray(R_host,A_array,0,0,M*N*sizeof(float),cudaMemcpyDeviceToHost);

			/*
			//Se imprime la matriz "R"
			printf("\n\n---MATRIZ R---\n\n");
			for(k=0;k<M;k++)
            {
				for(r=0;r<N;r++)
				{
					printf(" %f ",R_host[r + (k*N)]);
				}
				printf("\n");
            }
			printf("\n");
			*/

			//Se liberan memorias del Host y Device
			free(A_host);
			free(R_host);
			cudaFreeArray(A_array);


			//Suma de todos los tiempos
			suma = suma + elapsedTime_app;

			//Se destruyen los eventos que miden el tiempo de la aplicacion
			cudaEventDestroy(start_app);
			cudaEventDestroy(stop_app);
		}
        promedio[i-1] = suma/(float)loop;
        printf("%d - Tiempo promedio para N = %ld >>> %f mS\n",i,N,promedio[i-1]);

    }
    fwrite(promedio,sizeof(float),muestras,da);
    fclose(da);
}