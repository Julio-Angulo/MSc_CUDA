///VERSION SECUENCIAL CUDA 1 THREAD POLITOPO DESTINO 2D
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
surface<void, cudaSurfaceType2D> c_surface;
surface<void, cudaSurfaceType2D> s_surface;

//Kernel
__global__ void QR_kernel(int M,int N)
{
	float r,temp,data_1,data_2,data_3,data_4,data_5,data_6,aux_1,aux_2,aux_3,aux_4;
	int t,p1,p2,k,i,j;
	int aux_inf,aux_sup,aux_inf_1,aux_sup_1;

	aux_inf = 1-M;
	aux_sup = fminf(M+N-4,floorf((M+3*N-7)/2));

	//printf("\n\n aux_inf = %d  aux_sup = %d\n\n ",aux_inf,aux_sup);
	for(t = aux_inf;t <=aux_sup;t++)
	{
		aux_inf_1 = fmaxf(0,t-N+2);
		aux_sup_1 = fminf(M-2,floorf((t+M-1)/3));

		//printf(" t = %d\n ",t);
		//printf("\n\n aux_inf_1 = %d  aux_sup_1 = %d\n\n ",aux_inf_1,aux_sup_1);
		for(p1 = aux_inf_1;p1 <=aux_sup_1;p1++)
		{

			//printf(" p1 = %d\n ",p1);
			for(p2 = fmaxf(-M+1,t-2*p1-N+1);p2 <= fminf(t-3*p1,-1-p1);p2++)
			{
				//printf(" p2 = %d\n ",p2);

				
				k=p1;
				i=p2;
				j=t-2*p1-p2;
				if(j == k)
				{
					surf2Dread(&data_1,A_surface,(j*4),-i-1);
					surf2Dread(&data_2,A_surface,(j*4),-i);
					r = sqrt(pow(data_1,2) + pow(data_2,2));
					if(j < N-1)
					{
						if(r == 0)
						{
							surf2Dwrite(0,c_surface,(j*4),-i-1);
							surf2Dwrite(0,s_surface,(j*4),-i-1);
						}
						else
						{
							data_3 = data_1/r;
							data_4 = data_2/r;
							surf2Dwrite(data_3,c_surface,(j*4),-i-1);
							surf2Dwrite(data_4,s_surface,(j*4),-i-1);
						}
						surf2Dwrite(r,A_surface,(j*4),-i-1);
						surf2Dwrite(0,A_surface,(j*4),-i);
					}
				}
				
				else
				{
					surf2Dread(&data_5,A_surface,(j*4),-i-1);
					surf2Dread(&data_6,A_surface,(j*4),-i);
					temp = data_5;
					surf2Dread(&aux_1,c_surface,(j*4)-4,-i-1);
					surf2Dread(&aux_2,s_surface,(j*4)-4,-i-1);
					
					aux_3 = aux_1*temp + aux_2*data_6;
					aux_4 = -aux_2*temp + aux_1*data_6;
					
					//printf(" aux_3 = %f\n ",aux_3);
					//printf(" aux_4 = %f\n ",aux_4);
					
					surf2Dwrite(aux_3,A_surface,(j*4),-i-1);
					surf2Dwrite(aux_4,A_surface,(j*4),-i);

					//printf(" c = %f\n ",aux_1);
					//printf(" s = %f\n ",aux_2);
					if(j < N-1)
					{
						surf2Dwrite(aux_1,c_surface,(j*4),-i-1);
						surf2Dwrite(aux_2,s_surface,(j*4),-i-1);
					}
					
				}
				
			}
		}
	}
}

int main()
{
	int i,j,loop;
	int N,M,k,r;
	float suma;
	
	//Numero de iteraciones
	loop = 100;

	//Número de muestras
	const int muestras = 6;
	
	float promedio[muestras];

	///Se crean los archivos binarios donde se guardarán los datos
    FILE *da;
    da = fopen("Algoritmo_QR_CUDA_pol_destino_1_thread_2D.bin","a+b"); //Crea o sobre escribe archivo

	//Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();

	
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
			float* c_s_host;

            //Se reserva memoria en host para la matriz A
            A_host = (float*) malloc(sizeof(float)*N*M);
            R_host = (float*) malloc(sizeof(float)*N*M);
            c_s_host = (float*) malloc(sizeof(float)*(N-1)*(M-1));

			//Se dan valores a la matriz "A"
			for(k=0;k<M;k++)
            {
				for(r=0;r<N;r++)
				{
					//A_host[r + (k*N)]=rand()%10;
					A_host[r + (k*N)]= (k+1) + (r+1);
				}
            }

			//Se dan valores a las matrices "c" y "s"
			for(k=0;k<M-1;k++)
            {
				for(r=0;r<N-1;r++)
				{
					//A_host[r + (k*N)]=rand()%10;
					c_s_host[r + (k*(N-1))]= 0;
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
			cudaArray* c_array;
			cudaArray* s_array;
			
			cudaMallocArray(&A_array,&channelDesc,N,M,cudaArraySurfaceLoadStore);
			cudaMallocArray(&c_array,&channelDesc,N-1,M-1,cudaArraySurfaceLoadStore);
			cudaMallocArray(&s_array,&channelDesc,N-1,M-1,cudaArraySurfaceLoadStore);

			//Copy to device memory some data located at address A_host in host memory
			cudaMemcpyToArray(A_array,0,0,A_host,N*M*sizeof(float),cudaMemcpyHostToDevice);
			cudaMemcpyToArray(c_array,0,0,c_s_host,(N-1)*(M-1)*sizeof(float),cudaMemcpyHostToDevice);
			cudaMemcpyToArray(s_array,0,0,c_s_host,(N-1)*(M-1)*sizeof(float),cudaMemcpyHostToDevice);

			//Bind the arrays to the surface references
			cudaBindSurfaceToArray(A_surface,A_array);
			cudaBindSurfaceToArray(c_surface,c_array);
			cudaBindSurfaceToArray(s_surface,s_array);

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
			free(c_s_host);
			cudaFreeArray(A_array);
			cudaFreeArray(c_array);
			cudaFreeArray(s_array);


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