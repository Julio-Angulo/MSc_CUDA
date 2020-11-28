/* Este programa calcula la versi�n paralela de la libreria cuFFT (sin podado) para N= 2^20*/
/// (20/09/2016)
///�sta versi�n sirve para graficar en matlab los tiempos de ejecuci�n (RADIX-2) SIN PODAR N= 2^20
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
#include <time.h>

//////////////////////////////////////////////////////////////////////////
///////////////////////DECLARACI�N DE FUNCIONES///////////////////////////
//////////////////////////////////////////////////////////////////////////
void vector_entrada_xn(int N);
void cuFFT_noprun(void);

//////////////////////////////////////////////////////////////////////////
/////////////////////DECLARACI�N DE VARIABLES GLOBALES////////////////////
//////////////////////////////////////////////////////////////////////////
cuFloatComplex *x_host;
cuFloatComplex *X_host;
cuFloatComplex *x_device;
cuFloatComplex *X_device;
cufftComplex *in,*out;
FILE *db_open,*dc_open;

int N;

//////////////////////////////////////////////////////////////////////////
//////////////////////////DATOS DE ENTRADA////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/// N  >>> N�mero de elementos del vector de entrada
/// Li >>> N�mero de elementos de entrada diferentes de cero
/// Lo >>> N�mero de elementos de salida requeridos
/// loop >>> N�mero de iteraciones
/// muestras >>> N�mero de muestras

//////////////////////////////////////////////////////////////////////////
///////////////////////////DATOS DE SALIDA////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/// X >>> Vector de salida

//////////////////////////////////////////////////////////////////////////
/////////////////// SE INGRESAN LOS DATOS DE ENTRADA /////////////////////
//////////////////////////////////////////////////////////////////////////

///Ingrese el n�mero de iteraciones requeridas
const int loop = 300;

///Ingrese el valor de N_max

const int N_max = 20;



//////////////////////////////////////////////////////////////////////////
//////////////////////////FUNCION PRINCIPAL///////////////////////////////
//////////////////////////////////////////////////////////////////////////

//Funci�n principal 
int main()
{
	int i,j,i_N,j_res,k_res,cont,i_prom;
	float suma;
	float promedio[1];

	FILE *da;
	da = fopen("Tiempos_cuFFT__noprun_N20.bin","a+b"); //Crea o sobre escribe archivo

    //Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();
	
	for(i_N = N_max;i_N <= N_max;i_N++)
    {
        N = (int )pow(2,i_N);
        printf("\n N = %d \n",N);

		///Se abre el archivo binario
		db_open = fopen("Entrada_real_N20_C.bin","rb");
        dc_open = fopen("Entrada_imag_N20_C.bin","rb");

		suma=0.0;
		for(j=0;j<loop;j++)
		{
		
			//Comandos necesarios para medir el tiempo
			float elapsedTime_app;
			cudaEvent_t start_app, stop_app;
			cudaEventCreate(&start_app);
			cudaEventCreate(&stop_app);

			//Se generan en el host los valores del vector de entrada x[n] 
			vector_entrada_xn(N);

			//---------------------------------------------------------------------------------------------
			//Se empieza a medir el tiempo de ejecucion de la aplicacion
			cudaEventRecord(start_app,0);

			//Funci�n auxiliar del host para ejecutar la etapa intermedia
			cuFFT_noprun();

			//---------------------------------------------------------------------------------------------
			//Comandos necesarios para medir el tiempo de la aplicacion (app)
			cudaEventRecord(stop_app,0);
			cudaEventSynchronize(stop_app);
			cudaEventElapsedTime(&elapsedTime_app,start_app,stop_app);

			//Suma de todos los tiempos
			suma = suma + elapsedTime_app;

			//Se destruyen los eventos que miden el tiempo de la aplicacion
			cudaEventDestroy(start_app);
			cudaEventDestroy(stop_app);

			//Se liberan memorias del Host y Device
			free(x_host);
			free(X_host);
			cudaFree(x_device);
			cudaFree(X_device);
			

		}

		
		promedio[0] = suma/(float)loop;
		fclose(db_open);
        fclose(dc_open);
			
			
		
	}
	fwrite(promedio,sizeof(float),1,da);
	printf("\n\n- Tiempo promedio para N = %d >>> %f mS\n",N,promedio[0]);
    fclose(da);				

	return EXIT_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////
/////////////////////////FUNCIONES SECUNDARIAS////////////////////////////
//////////////////////////////////////////////////////////////////////////

//�sta funci�n genera el vector de entrada x[n]
void vector_entrada_xn(int N)
{
	//Declaraci�n de variables locales
	int k;
	float *buffer_real,*buffer_imag;

	//Se reserva memoria para xn_host en el host
	x_host = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*N);

	//Se reserva memoria para "X" en el host
	X_host = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*N);

	buffer_real = (float*)malloc(sizeof(float)*(pow(2,N_max)));
	buffer_imag = (float*)malloc(sizeof(float)*(pow(2,N_max)));

	///Se lee el vector de entrada del archivo binario
	fread(buffer_real,sizeof(float),(int)pow(2,N_max),db_open);
    fread(buffer_imag,sizeof(float),(int)pow(2,N_max),dc_open);


	//Se dan valores a x[n]
	for(k = 0;k < N; k++)
	{
		//x_host[k] = make_cuFloatComplex((float)(rand()%11),(float)(rand()%11));
		//x_host[k] = make_cuFloatComplex((float)(k + 1),(float)(0.0));
		x_host[k] = make_cuFloatComplex(buffer_real[k],buffer_imag[k]);
	}

	/*
	//Se imprimen los valores de entrada x[n]
	printf("\n---ELEMENTOS DE ENTRADA x[n]---\n\n");
	for(k=0;k<N;k++) 
	{
		printf(" %d-> (%f) + (%f)\n",k+1,cuCrealf(x_host[k]),cuCimagf(x_host[k]));
	}
	*/
	
	free(buffer_real);
	free(buffer_imag);
}


//Funci�n auxiliar del host para calcular la etapa intermedia en el device
void cuFFT_noprun(void)
{
	//////////////////////////////////////////////////////////////////////////
	////////////////////////////ETAPA INTERMEDIA//////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//Declaraci�n de variables locales
	int k1,k2,n1,m;
	int n[1] = {N};
	int inembed[1] = {NULL};
	int onembed[1] = {NULL};

	//Asignaci�n de memoria en el device para el arreglo "x_device"
	cudaMalloc((void**)&x_device,N*sizeof(cuFloatComplex));
	
	//Asignaci�n de memoria en el device para "X"
	cudaMalloc((void**)&X_device,N*sizeof(cuFloatComplex));

	//Se pasa el arreglo x_host a x_device
	cudaMemcpy(x_device,x_host,N*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);

	//Asignaci�n de memoria en el device para "in" y "out"
	cudaMalloc((void**)&in,sizeof(cufftComplex)*N);
	cudaMalloc((void**)&out,sizeof(cufftComplex)*N);

	//Se copia el arreglo "x_device" al arreglo "in"
	cudaMemcpy(in,x_device,sizeof(cuFloatComplex)*N,cudaMemcpyDeviceToDevice);

	//Se crea un plan
	cufftHandle plan;
	cufftPlanMany(&plan,1,n,inembed,1,1,onembed,1,1,CUFFT_C2C,1);

	//Ejecuci�n del plan
	cufftExecC2C(plan,in,out,CUFFT_FORWARD);

	//Esperar que el kernel termine de ejecutarse totalmente
	cudaDeviceSynchronize();

	//Se copian los datos del arreglo "out" al arreglo "z_device"
	cudaMemcpy(X_device,out,sizeof(cufftComplex)*N,cudaMemcpyDeviceToDevice);

	//Copia del arreglo "X" del device hacia el host
	cudaMemcpy(X_host,X_device,sizeof(cuFloatComplex)*N,cudaMemcpyDeviceToHost);

	//Se destruye el plan
	cufftDestroy(plan);

	//Se liberan los arreglos "in" y "out"
	cudaFree(in);
	cudaFree(out);

	/*
	//Se imprimen los valores de "X_host"
	///Imprimir X[k]
	printf("\n\n--- ARREGLO X[k] ---\n\n");
	for(m=0;m<=N-1;m++)
	{
		printf("\n X[%d] = %.4f + (%.4f)",m,cuCrealf(X_host[m]),cuCimagf(X_host[m]));
		//fprintf(da,"%.4f %.4f\n",creal(X[i]),cimag(X[i]));
	}
	*/
}

