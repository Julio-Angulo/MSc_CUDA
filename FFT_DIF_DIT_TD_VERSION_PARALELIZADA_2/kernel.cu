///Ésta programa calcula la versión paralelizada del algoritmo FFT_DIF_DIT_TD
///(03/08/2016)
///Ésta versión sirve para graficar en matlab los errores absolutos y relativos (RADIX-2) 2^1 - 2^10

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
///////////////////////DECLARACIÓN DE FUNCIONES///////////////////////////
//////////////////////////////////////////////////////////////////////////
void vector_entrada_xn(int N,int Li);
void arreglo_W(int N);
void asign_rap(int N,int Li,int Lo);
void factor(int N);
void product(int vector_1[50],int vector_2[50],int valor);
void etapa_entrada(void);
__global__ void inputStage_kernel(int N, int Li,int Dip,int Dop,int P,cuFloatComplex *x,cuFloatComplex *W,cuFloatComplex *y,int *flag_inputstage_1_d,int *flag_inputstage_2_d,int *flag_inputstage_3_d);
void etapa_intermedia(void);
void etapa_salida(void);
__global__ void outputStage_kernel(int N,int Lo,int Dip,int Dop,int P,cuFloatComplex *z,cuFloatComplex *W,cuFloatComplex *X,int *flag_outputstage_1_d,int *flag_outputstage_2_d,int *flag_outputstage_3_d);

//////////////////////////////////////////////////////////////////////////
/////////////////////DECLARACIÓN DE VARIABLES GLOBALES////////////////////
//////////////////////////////////////////////////////////////////////////
cuFloatComplex *x_host;
cuFloatComplex *W_host;
//cuFloatComplex *y_host;
//cuFloatComplex *z_host;
cuFloatComplex *X_host;
cuFloatComplex *x_device;
cuFloatComplex *W_device;
cuFloatComplex *y_device;
cuFloatComplex *z_device;
cuFloatComplex *X_device;
cufftComplex *in,*out;

int *flag_inputstage_1,*flag_inputstage_2,*flag_inputstage_3,*flag_outputstage_1,*flag_outputstage_2,*flag_outputstage_3;
int *flag_inputstage_1_d,*flag_inputstage_2_d,*flag_inputstage_3_d,*flag_outputstage_1_d,*flag_outputstage_2_d,*flag_outputstage_3_d;

int Dip,Dop,P,N,Li,Lo;
int vF[50]; //Almacena los factores de N
int svF; //Almacena el numero de factores de N
int Prod[50];
int a;

#define inf 99999

//////////////////////////////////////////////////////////////////////////
//////////////////////////DATOS DE ENTRADA////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/// N  >>> Número de elementos del vector de entrada
/// Li >>> Número de elementos de entrada diferentes de cero
/// Lo >>> Número de elementos de salida requeridos
/// loop >>> Número de iteraciones
/// muestras >>> Número de muestras

//////////////////////////////////////////////////////////////////////////
///////////////////////////DATOS DE SALIDA////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/// X >>> Vector de salida

//////////////////////////////////////////////////////////////////////////
/////////////////// SE INGRESAN LOS DATOS DE ENTRADA /////////////////////
//////////////////////////////////////////////////////////////////////////

///Ingrese el número de iteraciones requeridas
int loop = 1;

///Ingrese el número de muestras requeridas
const int muestras = 1;


//////////////////////////////////////////////////////////////////////////
//////////////////////////FUNCION PRINCIPAL///////////////////////////////
//////////////////////////////////////////////////////////////////////////

//Función principal 
int main()
{
	int i,j,alea_real[1024],alea_imag[1024],i_N,l_res,j_res,k_res,incremento_j;
	//float suma;
	//float promedio[muestras];

	///Se crean los archivos binarios donde se guardarán los datos
    FILE *da;
    FILE *db;
    FILE *dc;
    //FILE *dd;
	FILE *fi_1;
    FILE *fi_2;
    FILE *fi_3;
    FILE *fo_1;
    FILE *fo_2;
    FILE *fo_3;

    da = fopen("Resultados_radix_2_real_CUDA.bin","a+b"); //Crea o sobre escribe archivo
    db = fopen("Resultados_radix_2_imag_CUDA.bin","a+b"); //Crea o sobre escribe archivo
	dc = fopen("Entrada_radix_2_CUDA.txt","w+t"); //Crea o sobre escribe archivo
	//dd = fopen("TIEMPOS_FFT_DIF_DIT_TD_SECUENCIAL_CUDA.bin","a+b"); //Crea o sobre escribe archivo
	fi_1 = fopen("Flag_inputstage_1_radix_2_CUDA.bin","a+b"); //Crea o sobre escribe archivo
	fi_2 = fopen("Flag_inputstage_2_radix_2_CUDA.bin","a+b"); //Crea o sobre escribe archivo
	fi_3 = fopen("Flag_inputstage_3_radix_2_CUDA.bin","a+b"); //Crea o sobre escribe archivo
	fo_1 = fopen("Flag_outputstage_1_radix_2_CUDA.bin","a+b"); //Crea o sobre escribe archivo
    fo_2 = fopen("Flag_outputstage_2_radix_2_CUDA.bin","a+b"); //Crea o sobre escribe archivo
    fo_3 = fopen("Flag_outputstage_3_radix_2_CUDA.bin","a+b"); //Crea o sobre escribe archivo

	///Generación de vector de entrada aleatorio
    srand (time(NULL)); //Utilizo la hr del sistema como semilla
    for(i = 0;i < 1024;i++)
    {
        alea_real[i]=rand()%11;
        //alea_real[i]=i+1;
        alea_imag[i]=rand()%11;
        //alea_imag[i]=0;
        fprintf(dc,"%d %d\n",alea_real[i],alea_imag[i]);
    }
    fclose(dc);
	
	
	//Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();

	//Se reserva espacio para las flags
	
	flag_inputstage_1 = (int *)malloc(1*sizeof(int));
	flag_inputstage_2 = (int *)malloc(1*sizeof(int));
	flag_inputstage_3 = (int *)malloc(1*sizeof(int));
	flag_outputstage_1 = (int *)malloc(1*sizeof(int));
	flag_outputstage_2 = (int *)malloc(1*sizeof(int));
	flag_outputstage_3 = (int *)malloc(1*sizeof(int));
	cudaMalloc((int**)&flag_inputstage_1_d,1*sizeof(int));
	cudaMalloc((int**)&flag_inputstage_2_d,1*sizeof(int));
	cudaMalloc((int**)&flag_inputstage_3_d,1*sizeof(int));
	cudaMalloc((int**)&flag_outputstage_1_d,1*sizeof(int));
	cudaMalloc((int**)&flag_outputstage_2_d,1*sizeof(int));
	cudaMalloc((int**)&flag_outputstage_3_d,1*sizeof(int));
	

	//Inicializaciones
	incremento_j = 1;
	
	flag_inputstage_1[0] = 0;
    flag_inputstage_2[0] = 0;
    flag_inputstage_3[0] = 0;
    flag_outputstage_1[0] = 0;
    flag_outputstage_2[0] = 0;
    flag_outputstage_3[0] = 0;
	

	for(i_N = 1;i_N <= 10;i_N++)
    {
        N = (int )pow(2,i_N);
        printf("\n N = %d \n",N);

        //Se reserva memoria para xn_host en el host
		x_host = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*N);

		//Se reserva memoria para x_device y W_device
		cudaMalloc((void**)&x_device,N*sizeof(cuFloatComplex));
		cudaMalloc((void**)&W_device,N*sizeof(cuFloatComplex));

        ///Generación del vector x
        for(l_res=0;l_res < N;l_res++)
        {
            //x_host[l_res] = make_cuFloatComplex((float)(rand()%11),(float)(rand()%21));
			x_host[l_res] = make_cuFloatComplex((float)alea_real[l_res],(float)alea_imag[l_res]);
            //printf(" %d-> (%f) + (%f)\n",l_res+1,cuCrealf(x_host[l_res]),cuCimagf(x_host[l_res]));
        }

        ///Se genera el arreglo W[N]
        arreglo_W(N);

		//Envío de los arreglos x y W hacia la memoria global del device
		cudaMemcpy(x_device,x_host,N*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
		cudaMemcpy(W_device,W_host,N*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);

		for(j_res=incremento_j;j_res<=N;j_res=j_res+incremento_j)
        {
            Li=j_res;
			for(k_res=incremento_j;k_res<=N;k_res=k_res+incremento_j)
            {
               Lo=k_res;
			   //printf("\n Li = %d  Lo = %d",Li,Lo);
				for(i=1;i<=muestras;i++)
				{
					//suma=0.0;
					for(j=0;j<loop;j++)
					{
						//Comandos necesarios para medir el tiempo
						//float elapsedTime_app;
						//cudaEvent_t start_app, stop_app;
						//cudaEventCreate(&start_app);
						//cudaEventCreate(&stop_app);

						//---------------------------------------------------------------------------------------------
						//Se empieza a medir el tiempo de ejecucion de la aplicacion
						//cudaEventRecord(start_app,0);

						//Se generan en el host los valores del vector de entrada x[n] 
						//vector_entrada_xn(N,Li);

						//Se generan en el host los valores del arreglo W[N]
						//arreglo_W(N);

						//Se generan en el host los factores Dip y Dop
						asign_rap(N,Li,Lo);

						//Cálculo en el host del factor P
						P = N/(Dip*Dop);

						//printf("\n\n FACTOR P:\n\n");
						//printf("\n Dip = %d Dop = %d P = %d ",Dip,Dop,P);

						//Función auxiliar del host para ejecutar la etapa de entrada
						etapa_entrada();

						//Función auxiliar del host para ejecutar la etapa intermedia
						etapa_intermedia();

						//Función auxiliar del host para ejecutar la etapa de salida
						etapa_salida();

						

						///Se imprimen los resultados en los archivos binarios
                        int m;
                        float *parte_real;
                        float *parte_imag;
						parte_real = (float*) malloc(Lo*sizeof(float));
						parte_imag = (float*) malloc(Lo*sizeof(float));

                        for(m=0;m<=Lo-1;m++)
                        {
                            parte_real[m]=cuCrealf(X_host[m]);
                            parte_imag[m]=cuCimagf(X_host[m]);
                            //printf("\n X[%d] = %.4f + (%.4f)",m,creal(X[m]),cimag(X[m]));
                            //fprintf(dc,"%f %f\n",creal(X[m]),cimag(X[m]));
                        }

						

                        fwrite(parte_real,sizeof(float),Lo,da);
                        fwrite(parte_imag,sizeof(float),Lo,db);

						///Se leen los valores de las flags desde el device
						
						cudaMemcpy(flag_inputstage_1,flag_inputstage_1_d,1*sizeof(int),cudaMemcpyDeviceToHost);
						cudaMemcpy(flag_inputstage_2,flag_inputstage_2_d,1*sizeof(int),cudaMemcpyDeviceToHost);
						cudaMemcpy(flag_inputstage_3,flag_inputstage_3_d,1*sizeof(int),cudaMemcpyDeviceToHost);
						cudaMemcpy(flag_outputstage_1,flag_outputstage_1_d,1*sizeof(int),cudaMemcpyDeviceToHost);
						cudaMemcpy(flag_outputstage_2,flag_outputstage_2_d,1*sizeof(int),cudaMemcpyDeviceToHost);
						cudaMemcpy(flag_outputstage_3,flag_outputstage_3_d,1*sizeof(int),cudaMemcpyDeviceToHost);
						

						///Se imprimen el valor de las flags en sus respectivos archivos binarios
						
						fwrite(flag_inputstage_1,1*sizeof(int),1,fi_1);
						fwrite(flag_inputstage_2,1*sizeof(int),1,fi_2);
						fwrite(flag_inputstage_3,1*sizeof(int),1,fi_3);
						fwrite(flag_outputstage_1,1*sizeof(int),1,fo_1);
						fwrite(flag_outputstage_2,1*sizeof(int),1,fo_2);
						fwrite(flag_outputstage_3,1*sizeof(int),1,fo_3);
						

						/*
						printf("\n flag_inputstage_1 = %d \n",flag_inputstage_1[0]);
						printf("\n flag_inputstage_2 = %d \n",flag_inputstage_2[0]);
						printf("\n flag_inputstage_3 = %d \n",flag_inputstage_3[0]);
						printf("\n flag_outputstage_1 = %d \n",flag_outputstage_1[0]);
						printf("\n flag_outputstage_2 = %d \n",flag_outputstage_2[0]);
						printf("\n flag_outputstage_3 = %d \n",flag_outputstage_3[0]);
						*/
			
						//Se liberan memorias del Host y Device
						//free(x_host);
						//free(W_host);
						//free(y_host);
						//free(z_host);
						free(X_host);
						free(parte_real);
						free(parte_imag);
						//cudaFree(x_device);
						//cudaFree(W_device);
						cudaFree(y_device);
						cudaFree(z_device);
						cudaFree(X_device);


						//---------------------------------------------------------------------------------------------
						//Comandos necesarios para medir el tiempo de la aplicacion (app)
						//cudaEventRecord(stop_app,0);
						//cudaEventSynchronize(stop_app);
						//cudaEventElapsedTime(&elapsedTime_app,start_app,stop_app);

						//Suma de todos los tiempos
						//suma = suma + elapsedTime_app;

						//Se destruyen los eventos que miden el tiempo de la aplicacion
						//cudaEventDestroy(start_app);
						//cudaEventDestroy(stop_app);

						///Se resetean las flags
						
						flag_inputstage_1[0] = 0;
						flag_inputstage_2[0] = 0;
						flag_inputstage_3[0] = 0;
						flag_outputstage_1[0] = 0;
						flag_outputstage_2[0] = 0;
						flag_outputstage_3[0] = 0;
						
						
					}
					//promedio[i-1] = suma/(float)loop;
					//printf(" \n\n%d - Tiempo promedio para N = %ld >>> %f mS\n",i,N,promedio[i-1]);
				}
				//fwrite(promedio,sizeof(float),muestras,dd);
				//fclose(dd);
			}
		}
		free(x_host);
		free(W_host);
		cudaFree(x_device);
		cudaFree(W_device);
	}
	fclose(da);
    fclose(db);
	fclose(fi_1);
    fclose(fi_2);
    fclose(fi_3);
    fclose(fo_1);
    fclose(fo_2);
    fclose(fo_3);
	
	free(flag_inputstage_1);
	free(flag_inputstage_2);
	free(flag_inputstage_3);
	free(flag_outputstage_1);
	free(flag_outputstage_2);
	free(flag_outputstage_3);
	cudaFree(flag_inputstage_1_d);
	cudaFree(flag_inputstage_2_d);
	cudaFree(flag_inputstage_3_d);
	cudaFree(flag_outputstage_1_d);
	cudaFree(flag_outputstage_2_d);
	cudaFree(flag_outputstage_3_d);
	

    return EXIT_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////
/////////////////////////FUNCIONES SECUNDARIAS////////////////////////////
//////////////////////////////////////////////////////////////////////////

//Ésta función genera el vector de entrada x[n]
void vector_entrada_xn(int N,int Li)
{
	//Declaración de variables locales
	int k;

	//Se reserva memoria para xn_host en el host
	x_host = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*N);
	
	//Se dan valores a x[n]
	for(k=0;k<N;k++)
	{
		if(k < Li)
		{
			//x[k] = make_cuFloatComplex((float)(rand()%11),(float)(rand()%21));
			x_host[k] = make_cuFloatComplex((float)(k + 1),(float)(0.0));
		}
		else
		{
			x_host[k] = make_cuFloatComplex((float)(0.0),(float)(0.0));
		}
	}

	/*
	//Se imprimen los valores de entrada x[n]
	printf("\n---ELEMENTOS DE ENTRADA x[n]---\n\n");
	for(k=0;k<N;k++) 
	{
		printf(" %d-> (%f) + (%f)\n",k+1,cuCrealf(x_host[k]),cuCimagf(x_host[k]));
	}
	*/

}


//Ésta función genera el arreglo W
void arreglo_W(int N)
{
	//Declaración de variables locales
	int n;

	//Se reserva memoria para W_host en el host
	W_host = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*N);

	//Se genera el arreglo W
	for(n = 1;n <= N;n++)
	{
		W_host[n-1] = make_cuFloatComplex((float)cos((2*CUDART_PI*n)/N),(float)(-1)*sin((2*CUDART_PI*n)/N));
	}
	
	/*
	//Se imprimen los valores del arreglo W[N]
	printf("\n---ARREGLO W[N]---\n\n");
	for(n = 0;n < N; n++)
	{
		printf(" W[%d]-> (%f) + (%f)\n",n+1,cuCrealf(W_host[n]),cuCimagf(W_host[n]));
	}
	*/

}

//Ésta función genera los factores Dip y Dop
void asign_rap(int N,int Li,int Lo)
{
	//Declaración de variables locales
	float NLi,NLo,Diprapt,Doprapt;
	int Nh[50];
	int k[50];
	int G;
	int g,i,t,ta;
	int Dipt[50],Dopt[50];
	float distrapt,distrap;
	int Pos,h,Poss;
	int nk[50];
	int r;

	//Inicializaciones
	G = 0;
	svF = 0;


	//Factores Dip y Dop ideales
	NLi=(float)N/(float)Li;
    NLo=(float)N/(float)Lo;
    Diprapt=NLi;
    Doprapt=NLo;

	//Se encuentran los factores de "N"
	//vF almacena los factores de "N"
	//svF almacena el número de factores de "N"
	factor(N);

	/*
	Almacena en el vector Nh los factores que son diferentes de del vector vF
    En el vector k se almacena la cantidad de veces que se repite cada
    elemento almacenado en el vector Nh.
	*/

    Nh[0] = vF[0];
    k[0]=1;
	for(g=1;g<=svF-1;g=g+1)
    {
        if(vF[g]!=vF[g-1])
        {
           G=G+1;
           Nh[G]=vF[g];
           k[G]=1;
        }
        else
        {
            k[G]=k[G]+1;
        }

    }

	/*
	Almacena en el vector Nh todas las posibles combinaciones que den como
    producto a N. t almacena el numero de elementos del vector Nh.
	*/
	product(Nh,k,G);
	t = a;
	for(i=0;i<t;i=i+1)
    {
        Dipt[i]=Prod[i];
    }

	distrapt=inf;

	for(g=1;g<=t;g=g+1)
    {

        if(Dipt[g-1]<=NLi)
        {

            Pos=g-1;
            for(h=0;h<=G;h=h+1)
            {
                Poss=floor(Pos/(k[h]+1));
                nk[h]=k[h]+Poss*(k[h]+1)-Pos;
                Pos=Poss;
            }

            product(Nh,nk,G);

            ta=a;
            for(i=0;i<ta;i=i+1)
                {
                   Dopt[i]=Prod[i];
                }
          ////////////////////////////////////////////
           //int j;
           //for(j=0;j<ta;j++)
           //{
           //    printf(" %d ",Dopt[j]);
           //}
           //printf("\n\n ta=%d\n\n",ta);
           ///////////////////////////////////////////
            for(r=0;r<ta;r=r+1)
                {
                    distrap=sqrt(pow(Diprapt-(Dipt[g-1]),2)+pow(Doprapt-(Dopt[r]),2));
                    if(distrap<distrapt)
                        {
                            distrapt=distrap;
                            Dip=Dipt[g-1];
                            Dop=Dopt[r];
                        }
                }

        }

    }

	/*
	printf("\n\n FACTOR Dip :\n\n");
	printf(" %d ",Dip);

	printf("\n\n FACTOR Dop:\n\n");
	printf(" %d ",Dop);
	*/

}

//Ésta función encuentra los factores de "N"
void factor(int N)
{
	//Se empieza a verificar los factores desde 2
	int i=2;
	long N_factor;
	N_factor = N;
	 while(i<=N_factor)
	{
      while((N_factor%i)==0)
      {
          vF[svF]=i;
          N_factor=N_factor/i;
         // printf("Factores: %d ",vF[svF]);
          svF++;
      }
	i++;
	}

}

//Ésta función encuentra todas las posibles combinaciones de factores que den como resultado "N"
void product(int vector_1[50],int vector_2[50],int valor)
{
	int d,e,s,pNh,i;
    int cont=0;
    Prod[0]=1;
    a=1;
    for(d=0;d<=valor;d=d+1)
    {

        s=a;
        pNh=1;
        for(e=1;e<=vector_2[d];e=e+1)
        {
            pNh=pNh*vector_1[d];

            for(i=(s*e+1);i<=(s*e+s);i=i+1)
            {
                Prod[i-1]=pNh*Prod[cont];
                cont=cont+1;
            }
            a=a+s;
            cont=0;
        }

    }


}

//Función auxiliar del host para calcular la etapa de entrada en el device
void etapa_entrada(void)
{
	//////////////////////////////////////////////////////////////////////////
	////////////////////////////ETAPA DE ENTRADA//////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//Declaración de variables locales
	int k1,n1,n2;

	//Asignación de memoria en el device
	cudaMalloc((void**)&y_device,P*Dip*Dop*sizeof(cuFloatComplex));

	//Asignación de memoria en el host para "y"
	//y_host = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*P*Dip*Dop);

	//Dimensionamiento del grid para la función kernel "inputStage"
	//Dimensionamiento del Grid
	dim3 gridDim(1,1,1);
	//Dimensionamiento del block
	dim3 blockDim(1,1,1);
	if((P*Dop) < 32 && (Dip) < 32)
	{
		blockDim.x = (P*Dop);
		blockDim.y = (Dip);
		gridDim.x = 1;
		gridDim.y = 1;
	}
	else
	{
		blockDim.x = 32;
		blockDim.y = 32;
		gridDim.x = (unsigned int) (ceilf((float)(P*Dop)/(float)blockDim.x));
		gridDim.y = (unsigned int) (ceilf((float)Dip/(float)blockDim.y));
	}

	//Lanzamiento del kernel "inputStage_kernel"
	inputStage_kernel<<<gridDim,blockDim>>>(N,Li,Dip,Dop,P,x_device,W_device,y_device,flag_inputstage_1_d,flag_inputstage_2_d,flag_inputstage_3_d);

	//Esperar que el kernel termine de ejecutarse totalmente
	cudaDeviceSynchronize();

	/*
	//Copia del arreglo "y" del device hacia el host
	cudaMemcpy(y_host,y_device,sizeof(cuFloatComplex)*P*Dip*Dop,cudaMemcpyDeviceToHost);
	
	
	//Se imprimen los valores de "y"
	printf("\n\n--- ARREGLO y(n1,n2,k1) ---\n\n");
	for(k1 = 0;k1 < Dip;k1++) 
	{
		for(n1 = 0;n1 < Dop;n1++)
		{
			for(n2 = 0;n2 < P;n2++)
			{
				printf(" (%f) + (%f) ",cuCrealf(y_host[(k1*Dop*P)+(n1*P)+n2]),cuCimagf(y_host[(k1*Dop*P)+(n1*P)+n2]));
			}
			printf("\n");
		}
		printf("\n\n");
	}
	printf("\n");
	*/
	
}

//función kernel que ejecuta la etapa de entrada en el device
__global__ void inputStage_kernel(int N, int Li,int Dip,int Dop,int P,cuFloatComplex *x,cuFloatComplex *W,cuFloatComplex *y,int *flag_inputstage_1_d,int *flag_inputstage_2_d,int *flag_inputstage_3_d)
{
	int n1,n2;
	cuFloatComplex t1;

	//Threads
	int n = blockDim.x *blockIdx.x + threadIdx.x;
	int k1 = blockDim.y *blockIdx.y + threadIdx.y;

	//Se resetean las flags
	flag_inputstage_1_d[0] = 0;
	flag_inputstage_2_d[0] = 0;
	flag_inputstage_3_d[0] = 0;

	//printf("\n n = %d k1 = %d",n,k1);

	if( (n < (P*Dop)) && (k1 < Dip))
	{
		n2 = floorf(n/Dop);
		n1 = n - (Dop*n2);
		//Generación de los elementos que dependen de x[0]
		if(n == 0)
		{
			y[(k1*Dop*P)+(0*P)+ 0] = x[0];

			///Flag
			flag_inputstage_1_d[0] = 1;
			
		}
		//Mapeo de x[n] a las entradas del primer conjunto de Dop DFT's
		if((n >= 1) && (n <= (Li-1)))
		{
			t1 = x[n];
			if(k1 == 0)
			{
				y[(0*Dop*P)+(n1*P)+ n2] = t1;
			}
			if(k1 >= 1)
			{
				y[(k1*Dop*P)+(n1*P)+ n2] = cuCmulf(W[((n*k1)%N)-1],t1);
			}

			///Flag
			flag_inputstage_2_d[0] = 1;
		}
		//Rellenado de ceros para los elementos de "y" para Li <= n <= (P*Dop)-1
		if((n >= Li) && (n <= (P*Dop)-1))
		{
			y[(k1*Dop*P)+(n1*P)+ n2] = make_cuFloatComplex(0.0,0.0);

			///Flag
			flag_inputstage_3_d[0] = 1;
		}

		
		//printf("\n (%f) + (%f)\n ",cuCrealf(y[(k1*Dop*P)+(n1*P)+ n2]),cuCimagf(y[(k1*Dop*P)+(n1*P)+ n2]));
	}
}

//Función auxiliar del host para calcular la etapa intermedia en el device
void etapa_intermedia(void)
{
	//////////////////////////////////////////////////////////////////////////
	////////////////////////////ETAPA INTERMEDIA//////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//Declaración de variables locales
	int k1,k2,n1;
	int n[1] = {P};
	int inembed[1] = {P};
	int onembed[1] = {P};

	//Asignación de memoria en el device para "z"
	cudaMalloc((void**)&z_device,P*Dip*Dop*sizeof(cuFloatComplex));

	//Asignación de memoria en el host para "z"
	//z_host = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*P*Dip*Dop);

	//Asignación de memoria en el device para "in" y "out"
	cudaMalloc((void**)&in,sizeof(cufftComplex)*P*Dip*Dop);
	cudaMalloc((void**)&out,sizeof(cufftComplex)*P*Dip*Dop);

	//Se copia el arreglo "y" al arreglo "in"
	cudaMemcpy(in,y_device,sizeof(cuFloatComplex)*P*Dip*Dop,cudaMemcpyDeviceToDevice);

	//Se crea un plan
	cufftHandle plan;
	cufftPlanMany(&plan,1,n,inembed,1,P,onembed,1,P,CUFFT_C2C,Dip*Dop);

	//Ejecución del plan
	cufftExecC2C(plan,in,out,CUFFT_FORWARD);

	//Esperar que el kernel termine de ejecutarse totalmente
	cudaDeviceSynchronize();

	//Se copian los datos del arreglo "out" al arreglo "z_device"
	cudaMemcpy(z_device,out,sizeof(cufftComplex)*P*Dip*Dop,cudaMemcpyDeviceToDevice);

	//Se destruye el plan
	cufftDestroy(plan);

	//Se liberan los arreglos "in" y "out"
	cudaFree(in);
	cudaFree(out);

	/*
	//Se copian los datos del arreglo "z_device" al arreglo "z_host"
	cudaMemcpy(z_host,z_device,sizeof(cuFloatComplex)*P*Dip*Dop,cudaMemcpyDeviceToHost);

	
	///Se imprimen los valores de z(n1,k2,k1)
	printf("\n\n--- ARREGLO z(n1,k2,k1) ---\n\n");
	for(k1 = 0;k1 < Dip;k1++) 
	{
		for(n1 = 0;n1 < Dop;n1++)
		{
			for(k2 = 0;k2 < P;k2++)
			{
				printf(" (%f) + (%f) ",cuCrealf(z_host[(k1*Dop*P)+(n1*P)+k2]),cuCimagf(z_host[(k1*Dop*P)+(n1*P)+k2]));
			}
			printf("\n");
		}
		printf("\n\n");
	}
	printf("\n");
	*/
	
}

//Función auxiliar del host para calcular la etapa de salida en el device
void etapa_salida(void)
{
	//////////////////////////////////////////////////////////////////////////
	////////////////////////////ETAPA DE SALIDA///////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//Declaración de variables locales
	int m;

	//Asignación de memoria en el device para "X"
	cudaMalloc((void**)&X_device,Lo*sizeof(cuFloatComplex));

	//Asignación de memoria en el host para "X"
	X_host = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*Lo);

	//Dimensionamiento del grid para la función kernel "outputStage"
	//Dimensionamiento del Grid
	dim3 gridDim(1,1,1);
	//Dimensionamiento del block
	dim3 blockDim(1,1,1);
	if((Lo) < 1024)
	{
		blockDim.x = Lo;
		gridDim.x = 1;
	}
	else
	{
		blockDim.x = 1024;
		gridDim.x = (unsigned int) (ceilf((float)Lo/(float)blockDim.x));
	}

	//Lanzamiento del kernel "outputStage_kernel"
	outputStage_kernel<<<gridDim,blockDim>>>(N,Lo,Dip,Dop,P,z_device,W_device,X_device,flag_outputstage_1_d,flag_outputstage_2_d,flag_outputstage_3_d);

	//Esperar que el kernel termine de ejecutarse totalmente
	cudaDeviceSynchronize();

	//Copia del arreglo "X" del device hacia el host
	cudaMemcpy(X_host,X_device,sizeof(cuFloatComplex)*Lo,cudaMemcpyDeviceToHost);
			
	/*
	//Se imprimen los valores de "X_host"
    ///Imprimir X[k]
    printf("\n\n--- ARREGLO X[k] ---\n\n");
    for(m=0;m<=Lo-1;m++)
    {
        printf("\n X[%d] = %.4f + (%.4f)",m,cuCrealf(X_host[m]),cuCimagf(X_host[m]));
        //fprintf(da,"%.4f %.4f\n",creal(X[i]),cimag(X[i]));
    }
	*/
	

}

//función kernel que ejecuta la etapa de salida en el device
__global__ void outputStage_kernel(int N,int Lo,int Dip,int Dop,int P,cuFloatComplex *z,cuFloatComplex *W,cuFloatComplex *X,int *flag_outputstage_1_d,int *flag_outputstage_2_d,int *flag_outputstage_3_d)
{
	//Declaración de variables locales
	int n1,k_aux,k1,k2,a,b;
	cuFloatComplex t1,t2,t3,t4,t5;
	

	//Threads
	int k = blockDim.x *blockIdx.x + threadIdx.x;

	//Se resetean las flags
	flag_outputstage_1_d[0] = 0;
	flag_outputstage_2_d[0] = 0;
	flag_outputstage_3_d[0] = 0;

	if(k < Lo)
	{
		for(n1 = 0; n1 <= (Dop-1); n1 = n1+1)
		{
			if(Lo <= Dip)
			{
				//Cálculo de X(k) para 0<=k<=Lo-1.
				//printf("\n--- Caso (Lo <= Dip) ---\n");
				//En la descomposición k = k1 + Dipk2; k2 = 0, y por lo tanto, k = k1
				if(n1 == 0) //Caso para lograr que por lo menos ingrese una vez 
				{
					X[k] = z[(k*Dop*P)+(0*P) + 0];
					
					///Flag
					flag_outputstage_1_d[0] = 1;
				}
				else
				{
					if(n1 == 1)
					{
						X[k] = z[(k*Dop*P)+(0*P) + 0];
					}
					X[k] = cuCaddf(z[(k*Dop*P)+(n1*P) + 0],X[k]);

					///Flag
					flag_outputstage_1_d[0] = 1;

				}

			}
			else
			{
				if((k >= 0) && (k <= (Dip-1)))
				{
					//Cálculo de X(k) para 0<=k<=Dip-1.
					//En la descomposición k = k1 + Dipk2; k2 = 0, y por lo tanto, k = k1
					if(n1 == 0) //Caso para lograr que por lo menos ingrese una vez 
					{
						X[k] = z[(k*Dop*P)+(0*P) + 0];
					}
					else
					{
						if(n1 == 1)
						{
							X[k] = z[(k*Dop*P)+(0*P) + 0];
						}
						X[k] = cuCaddf(z[(k*Dop*P)+(n1*P) + 0],X[k]);

					}
					

				}
				else
				{
					
					if(Dop <= 4)
					{
						//Usando el método directo
						//printf("\n--- Caso (Metodo directo) ---\n");
						if(n1 == 0) //Caso para lograr que por lo menos ingrese una vez 
						{
							k_aux = k-((Dip*P)*floorf(k/(Dip*P)));
							k2 = floorf(k_aux/Dip);
							k1 = k_aux-(Dip*k2);
							X[k] = z[(k1*Dop*P)+(0*P)+ (k2%P)];

							///Flag
							flag_outputstage_2_d[0] = 1;

						}
						else
						{
							if(n1 == 1)
							{
								k_aux = k-((Dip*P)*floorf(k/(Dip*P)));
								k2 = floorf(k_aux/Dip);
								k1 = k_aux-(Dip*k2);
								X[k] = z[(k1*Dop*P)+(0*P)+ (k2%P)];
							}
							a = floorf(k/(Dip*P));
							X[k] = cuCaddf(X[k],cuCmulf(z[(k1*Dop*P)+(n1*P)+ (k2%P)],W[((n1*(k2+P*(a))*Dip)%N)-1]));

							///Flag
							flag_outputstage_2_d[0] = 1;

						}

					}
					
					
					
					else
					{
						//Usando el método filtering 2BF
						//printf("\n--- Caso (Filtro 2BF) ---\n");
						if((Dop-2) >= 1)
						{
							if(n1 == 0) 
							{
								k_aux = k-((Dip*P)*floorf(k/(Dip*P)));
								k2 = floorf(k_aux/Dip);
								k1 = k_aux-(Dip*k2);
								t1 = z[(k1*Dop*P)+((Dop-1)*P)+ (k2%P)];
								b = floorf(k/(Dip*P));
								t4 = cuCmulf(t1,make_cuFloatComplex(2*cuCrealf(W[(((k2+P*(b))*Dip)%N)-1]),0.0));

								///Flag
								flag_outputstage_3_d[0] = 1;
							}

							if((n1 >= 1) && (n1 <= (Dop-2)))
							{
								t2 = t1;
								t1 = cuCaddf(z[(k1*Dop*P)+((-(n1-(Dop-1)))*P)+ (k2%P)],t4); 
								t3 = cuCmulf(t1,make_cuFloatComplex(2*cuCrealf(W[(((k2+P*(b))*Dip)%N)-1]),0.0));
								t4 = cuCsubf(t3,t2);
							}

							if(n1 == (Dop-1))
							{
								t5 = cuCaddf(z[(k1*Dop*P)+(0*P)+ (k2%P)],t4);
								X[k] = cuCsubf(t5,cuCmulf(t1,cuConjf(W[(((k2+P*(b))*Dip)%N)-1])));
							}

						}

						else
						{
							if(Dop == 1)
							{
								k_aux = k-((Dip*P)*floorf(k/(Dip*P)));
								k2 = floorf(k_aux/Dip);
								k1 = k_aux-(Dip*k2);
								t1 = z[(k1*Dop*P)+((Dop-1)*P)+ (k2%P)];
								X[k] = t1;

								///Flag
								flag_outputstage_3_d[0] = 1;
							}
							else
							{
								k_aux = k-((Dip*P)*floorf(k/(Dip*P)));
								k2 = floorf(k_aux/Dip);
								k1 = k_aux-(Dip*k2);
								t1 = z[(k1*Dop*P)+((Dop-1)*P)+ (k2%P)];
								b = floorf(k/(Dip*P));
								t4 = cuCmulf(t1,make_cuFloatComplex(2*cuCrealf(W[(((k2+P*(b))*Dip)%N)-1]),0.0));
								t5 = cuCaddf(z[(k1*Dop*P)+(0*P)+ (k2%P)],t4);
								X[k] = cuCsubf(t5,cuCmulf(t1,cuConjf(W[(((k2+P*(b))*Dip)%N)-1])));

								///Flag
								flag_outputstage_3_d[0] = 1;
							}

						}
						
					}
					
					

				}
				

			}
				
		}
	}
}
