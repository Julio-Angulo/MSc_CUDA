
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h> 
#include <cuComplex.h>
#include <math.h>
#include <math_constants.h>
#include <iostream>

//2D surfaces
surface<void, cudaSurfaceType2D> A_surface;
surface<void, cudaSurfaceType2D> B_surface;
surface<void, cudaSurfaceType2D> R_surface;



//Kernel
__global__ void Mul_kernel(int I,int J,int K)
{
	
	int x2 = blockDim.x *blockIdx.x + threadIdx.x;
	int x1 = blockDim.y *blockIdx.y + threadIdx.y;
	float data_1,data_2,data_3,aux_1,aux_2;
	int x3;

	//printf(" I = %d, J = %d, K = %d",I,J,K);

	if( (x1 < I) && (x2 < J))
	{
		//printf(" x1 = %d, x2 = %d\n",x1,x2);
		aux_1 = 0.0;
		surf2Dwrite(aux_1,R_surface,(x2*4),x1);
		
		for(x3 = 0; x3 < K; x3++)
		{
			surf2Dread(&data_1,R_surface,(x2*4),x1);
			surf2Dread(&data_2,A_surface,(x3*4),x1);
			surf2Dread(&data_3,B_surface,(x2*4),x3);
			aux_2 = data_1 + (data_2 * data_3);
			surf2Dwrite(aux_2,R_surface,(x2*4),x1);
			//printf(" x1 = %d, x2 = %d, x3 = %d\n",x1,x2,x3);
		}
		
	}	
}

int main()
{
	int i,j,loop,x,y;
	float suma;
	int I,J,K;
	
	//Numero de iteraciones
	loop = 1;

	//Número de muestras
	const int muestras = 1;
	
	float promedio[muestras];

	///Se crean los archivos binarios donde se guardarán los datos
    //FILE *da;
    //da = fopen("Multiplicacion_matricial.bin","a+b"); //Crea o sobre escribe archivo

	//Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();

	
	for(i=1;i<=muestras;i++)
    {
        //Número de filas de la matriz "A"
        I = 3;

		//Número de columnas de la matriz "A" y número de filas de la matriz "B"
		K=3;

		//Número de columnas de la matriz "B"
        J = 4;

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
			float* B_host;
			float* R_host;

            //Se reserva memoria en host para la matriz A
            A_host = (float*) malloc(sizeof(float)*I*K);
            B_host = (float*) malloc(sizeof(float)*K*J);
            R_host = (float*) malloc(sizeof(float)*I*J);

			//Se dan valores a la matriz "A"
			for(x = 0;x <= I-1; x++)
            {
				for(y = 0;y <= K-1; y++)
				{
					//A_host[y + (x*K)]=rand()%10;
					A_host[y + (x*K)]= float(x+1) + float(y+1);
				}
            }

			//Se dan valores a la matriz "B"
			for(x = 0;x <= K-1; x++)
            {
				for(y = 0;y <= J-1; y++)
				{
					//A_host[y + (x*K))]=rand()%10;
					B_host[y + (x*J)]= float(x+1) + float(y+1);
				}
            }

			//Se imprime la matriz "A"
			printf("\n\n---MATRIZ A---\n\n");
			for(x = 0;x <= I-1; x++)
            {
				for(y = 0; y <= K-1; y++)
				{
					printf(" %f ",A_host[y + (x*K)]);
				}
				printf("\n");
            }
			printf("\n");

			//Se imprime la matriz "B"
			printf("\n\n---MATRIZ B---\n\n");
			for(x = 0;x <= K-1; x++)
            {
				for(y = 0; y <= J-1; y++)
				{
					printf(" %f ",B_host[y + (x*J)]);
				}
				printf("\n");
            }
			printf("\n");
			

			//Allocate CUDA arrays in device memory
			cudaChannelFormatDesc channelDesc;
			channelDesc = cudaCreateChannelDesc<int>();
			cudaArray* A_array;
			cudaArray* B_array;
			cudaArray* R_array;
			
			cudaMallocArray(&A_array,&channelDesc,K,I,cudaArraySurfaceLoadStore);
			cudaMallocArray(&B_array,&channelDesc,J,K,cudaArraySurfaceLoadStore);
			cudaMallocArray(&R_array,&channelDesc,J,I,cudaArraySurfaceLoadStore);

			//Copy to device memory some data located at address A_host in host memory
			cudaMemcpyToArray(A_array,0,0,A_host,I*K*sizeof(float),cudaMemcpyHostToDevice);
			cudaMemcpyToArray(B_array,0,0,B_host,K*J*sizeof(float),cudaMemcpyHostToDevice);
			

			//Bind the arrays to the surface references
			cudaBindSurfaceToArray(A_surface,A_array);
			cudaBindSurfaceToArray(B_surface,B_array);
			cudaBindSurfaceToArray(R_surface,R_array);

			//Dimensionamiento del grid para la función kernel
			//Dimensionamiento del Grid
			dim3 gridDim(1,1,1);
			//Dimensionamiento del block
			dim3 blockDim(1,1,1);
			//Dimensionamiento del Block y Grid en "y" 
			if(I < 32 )
			{
				blockDim.y = I;
				gridDim.y = 1;
			}
			else
			{
				blockDim.y = 32;
				gridDim.y = int (ceil(I/blockDim.y));
			}
			//Dimensionamiento del Block y Grid en "x" 
			if(J < 32 )
			{
				blockDim.x = J;
				gridDim.x = 1;
			}
			else
			{
				blockDim.x = 32;
				gridDim.x = int (ceil(J/blockDim.x));
			}

			//---------------------------------------------------------------------------------------------
			//Se empieza a medir el tiempo de ejecucion de la aplicacion
			cudaEventRecord(start_app,0);
			
			//Lanzamiento del kernel
			Mul_kernel<<<gridDim,blockDim >>>(I,J,K);
			cudaThreadSynchronize();
			
			//---------------------------------------------------------------------------------------------
			//Comandos necesarios para medir el tiempo de la aplicacion (app)
			cudaEventRecord(stop_app,0);
			cudaEventSynchronize(stop_app);
			cudaEventElapsedTime(&elapsedTime_app,start_app,stop_app);

			//Se leen los resultados de la GPU
			cudaMemcpyFromArray(R_host,R_array,0,0,I*J*sizeof(float),cudaMemcpyDeviceToHost);

			
			//Se imprime la matriz "R"
			printf("\n\n---MATRIZ R---\n\n");
			for(x = 0;x <= I-1; x++)
            {
				for(y = 0;y <= J-1;y++)
				{
					printf(" %f ",R_host[y + (x*J)]);
				}
				printf("\n");
            }
			printf("\n");
			

			//Se liberan memorias del Host y Device
			free(A_host);
			free(B_host);
			free(R_host);
			cudaFreeArray(A_array);
			cudaFreeArray(B_array);
			cudaFreeArray(R_array);


			//Suma de todos los tiempos
			suma = suma + elapsedTime_app;

			//Se destruyen los eventos que miden el tiempo de la aplicacion
			cudaEventDestroy(start_app);
			cudaEventDestroy(stop_app);
		}
        promedio[i-1] = suma/(float)loop;
        printf("%d - Tiempo promedio para R(%d,%d)= A(%d,%d)*B(%d,%d) >>> %f mS\n",i,I,J,I,K,K,J,promedio[i-1]);

    }
    //fwrite(promedio,sizeof(float),muestras,da);
    //fclose(da);
}