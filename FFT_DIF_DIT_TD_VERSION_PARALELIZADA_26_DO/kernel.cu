///�sta programa calcula la versi�n paralelizada del algoritmo FFT_DIF_DIT_TD
///(19/01/2017)
///Grafica en Matlab los tiempos de ejecuci�n, considerando N-Compuesta. N = (2^5)x(3^4)x(5^4), Li = var�a , Lo= N. (precisi�n doble).

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
void vector_entrada_xn(int Li);
void arreglo_W(int N);
void asign_rap(int N,int Li,int Lo);
void factor(int N);
void product(int vector_1[500],int vector_2[500],int valor);
void etapa_entrada(void);
__global__ void inputStage_kernel(int N, int Li,int Dip,int Dop,int P,cuDoubleComplex *x,cuDoubleComplex *W,cuDoubleComplex *y);
void etapa_intermedia(void);
void etapa_salida(void);
__global__ void outputStage_kernel(int N,int Lo,int Dip,int Dop,int P,cuDoubleComplex *z,cuDoubleComplex *W,cuDoubleComplex *X);

//////////////////////////////////////////////////////////////////////////
/////////////////////DECLARACI�N DE VARIABLES GLOBALES////////////////////
//////////////////////////////////////////////////////////////////////////
cuDoubleComplex *x_host;
cuDoubleComplex *W_host;
//cuDoubleComplex *y_host;
//cuDoubleComplex *z_host;
cuDoubleComplex *X_host;
cuDoubleComplex *x_device;
cuDoubleComplex *W_device;
cuDoubleComplex *y_device;
cuDoubleComplex *z_device;
cuDoubleComplex *X_device;
cufftDoubleComplex *in,*out;
FILE *db_open,*dc_open;

int Dip,Dop,P,N,Li,Lo;
int vF[500]; //Almacena los factores de N
int svF; //Almacena el numero de factores de N
int Prod[500];
int a;

#define inf 99999

//////////////////////////////////////////////////////////////////////////
/////////////////// SE INGRESAN LOS DATOS DE ENTRADA /////////////////////
//////////////////////////////////////////////////////////////////////////

///Ingrese el n�mero de iteraciones requeridas
const int loop = 300;

///Ingrese el valor de n_max, m_max y l_max (N = (2^n_max x 3^m_max x 5^l_max))

const int n_max = 5;
const int m_max = 4;
const int l_max = 4;

///Ingrese el valor de Li_max
const int Lo_max = 1620000;

//////////////////////////////////////////////////////////////////////////
//////////////////////////FUNCION PRINCIPAL///////////////////////////////
//////////////////////////////////////////////////////////////////////////

//Funci�n principal 
int main()
{
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////SELECCI�N DEL DEVICE////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	int device;
	FILE *da;
	cudaSetDevice(0);
	cudaGetDevice(&device);
	if(device == 1)
	{
		printf("\n\n---DEVICE = GeForce GTX 970---\n\n");
		da = fopen("Tiempos_NCompuesta_LiVARIA_LoN_CUDA_GTX970_DO.bin","a+b"); //Crea o sobre escribe archivo
	}
	if(device == 0)
	{
		printf("\n\n---DEVICE = TESLA K20---\n\n");
		da = fopen("Tiempos_NCompuesta_LiVARIA_LoN_CUDA_TESLAK20c_DO.bin","a+b"); //Crea o sobre escribe archivo
	}
	//////////////////////////////////////////////////////////////////////////

	int i,j,i_N,j_res,k_res,cont,i_prom;
	float suma;
	float promedio[13];

	int cont_1,n_1,m_1,l_1,m_ant,l_ant;
	cont_1 = 0;
	m_ant = 0;
	l_ant = 0;

    //Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();
	
	for(i_N = 1;i_N <= 1;i_N++)
    {
		N = (int )((pow(2,n_max))*(pow(3,m_max))*(pow(5,l_max)));

        printf("\n N = %d \n",N);

		for(j_res=Lo_max;j_res <= Lo_max;j_res++)
        {
            Lo=j_res;
			for(n_1 = 1;n_1 <= n_max;n_1++)
			{
				for(m_1 = m_ant; m_1 <= n_1;m_1++)
				{
					m_ant = m_1;
					for(l_1 = l_ant;l_1 <= m_1;l_1++)
					{
						l_ant = l_1;
						if((m_1 <= m_max) && (l_1 <= l_max))
						{
							Li = (int )((pow(2,n_1))*(pow(3,m_1))*(pow(5,l_1)));
							cont_1++;
							printf("\n Li = %d  Lo = %d",Li,Lo);

							///Se abre el archivo binario
							db_open = fopen("Entrada_real_NCompuesta_C.bin","rb");
							dc_open = fopen("Entrada_imag_NCompuesta_C.bin","rb");

							

							suma=0.0;
							for(j=0;j<loop;j++)
							{
		
								//Comandos necesarios para medir el tiempo
								float elapsedTime_app;
								cudaEvent_t start_app, stop_app;
								cudaEventCreate(&start_app);
								cudaEventCreate(&stop_app);

								//Se generan en el host los valores del vector de entrada x[n] 
								vector_entrada_xn(Li);

								


								///Se genera el arreglo W[N]
								arreglo_W(N);
								
								
								//---------------------------------------------------------------------------------------------
								//Se empieza a medir el tiempo de ejecucion de la aplicacion
								cudaEventRecord(start_app,0);

								

								//Se generan en el host los factores Dip y Dop
								asign_rap(N,Li,Lo);

								

								//C�lculo en el host del factor P
								P = N/(Dip*Dop);

								

								//printf("\n\n FACTOR P:\n\n");
								//printf("\n Dip = %d Dop = %d P = %d ",Dip,Dop,P);

								//Funci�n auxiliar del host para ejecutar la etapa de entrada
								etapa_entrada();

								//Funci�n auxiliar del host para ejecutar la etapa intermedia
								etapa_intermedia();

								//Funci�n auxiliar del host para ejecutar la etapa de salida
								etapa_salida();

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
								free(W_host);
								free(X_host);
								cudaFree(x_device);
								cudaFree(W_device);
								cudaFree(y_device);
								cudaFree(z_device);
								cudaFree(X_device);

								

							}
							//printf("\n Dip = %d Dop = %d P = %d\n\n",Dip,Dop,P);

							promedio[cont_1-1] = suma/(float)loop;
							fclose(db_open);
							fclose(dc_open);
							

						}

					}
				}
			}
		}
	}
	fwrite(promedio,sizeof(float),13,da);
	printf("\n\nTIEMPOS:\n\n");
	int time_print;
	for(time_print = 0;time_print < cont_1;time_print++)
	{
		printf("\nTime (%d)= %f ms",time_print,promedio[time_print]);
	}
    fclose(da);				

	
}


//////////////////////////////////////////////////////////////////////////
/////////////////////////FUNCIONES SECUNDARIAS////////////////////////////
//////////////////////////////////////////////////////////////////////////

//�sta funci�n genera el vector de entrada x[n]
void vector_entrada_xn(int Li)
{
	//Declaraci�n de variables locales
	int k;
	float *buffer_real,*buffer_imag;

	//Se reserva memoria para xn_host en el host
	x_host = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*Li);
	buffer_real = (float*)malloc(sizeof(float)*N);
	buffer_imag = (float*)malloc(sizeof(float)*N);

	///Se lee el vector de entrada del archivo binario
	fread(buffer_real,sizeof(float),N,db_open);
    fread(buffer_imag,sizeof(float),N,dc_open);


	//Se dan valores a x[n]
	for(k = 0;k < Li; k++)
	{
		//x_host[k] = make_cuFloatComplex((double)(rand()%11),(double)(rand()%11));
		//x_host[k] = make_cuDoubleComplex((double)(k + 1),(double)(0.0));
		x_host[k] =   make_cuDoubleComplex((double)buffer_real[k],(double)buffer_imag[k]);
	}
	


	/*
	//Se imprimen los valores de entrada x[n]
	printf("\n---ELEMENTOS DE ENTRADA x[n]---\n\n");
	for(k=0;k<Li;k++) 
	{
		printf(" %d-> (%f) + (%f)\n",k+1,cuCreal(x_host[k]),cuCimag(x_host[k]));
	}
	*/
	
	free(buffer_real);
	free(buffer_imag);
}


//�sta funci�n genera el arreglo W
void arreglo_W(int N)
{
	//Declaraci�n de variables locales
	int n;

	//Se reserva memoria para W_host en el host
	W_host = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*N);

	//Se genera el arreglo W
	for(n = 1;n <= N;n++)
	{
		W_host[n-1] = make_cuDoubleComplex((double)cos((2*CUDART_PI*n)/N),(double)(-1)*sin((2*CUDART_PI*n)/N));
		
	}
	
	
	/*
	//Se imprimen los valores del arreglo W[N]
	printf("\n---ARREGLO W[N]---\n\n");
	for(n = 0;n < N; n++)
	{
		printf(" W[%d]-> (%f) + (%f)\n",n+1,cuCreal(W_host[n]),cuCimag(W_host[n]));
	}
	*/

}

//�sta funci�n genera los factores Dip y Dop
void asign_rap(int N,int Li,int Lo)
{
	//Declaraci�n de variables locales
	float NLi,NLo,Diprapt,Doprapt;
	int Nh[500];
	int k[500];
	int G;
	int g,i,t,ta;
	int Dipt[500],Dopt[500];
	float distrapt,distrap;
	int Pos,h,Poss;
	int nk[500];
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
	//svF almacena el n�mero de factores de "N"
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

//�sta funci�n encuentra los factores de "N"
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

//�sta funci�n encuentra todas las posibles combinaciones de factores que den como resultado "N"
void product(int vector_1[500],int vector_2[500],int valor)
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

//Funci�n auxiliar del host para calcular la etapa de entrada en el device
void etapa_entrada(void)
{
	//////////////////////////////////////////////////////////////////////////
	////////////////////////////ETAPA DE ENTRADA//////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//Declaraci�n de variables locales
	int k1,n1,n2;

	//Asignaci�n de memoria en el device para el arreglo "x_device"
	cudaMalloc((void**)&x_device,Li*sizeof(cuDoubleComplex));

	//Se reserva memoria en el device para el arreglo "W_device"
	cudaMalloc((void**)&W_device,N*sizeof(cuDoubleComplex));

	//Asignaci�n de memoria en el device para el arreglo "y"
	cudaMalloc((void**)&y_device,P*Dip*Dop*sizeof(cuDoubleComplex));

	//Se pasa el arreglo x_host a x_device
	cudaMemcpy(x_device,x_host,Li*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);

	//Env�o de los arreglos W hacia la memoria global del device
	cudaMemcpy(W_device,W_host,N*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);

	//Asignaci�n de memoria en el host para "y"
	//y_host = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*P*Dip*Dop);

	//Dimensionamiento del grid para la funci�n kernel "inputStage"
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
	inputStage_kernel<<<gridDim,blockDim>>>(N,Li,Dip,Dop,P,x_device,W_device,y_device);

	//Esperar que el kernel termine de ejecutarse totalmente
	cudaDeviceSynchronize();

	/*
	//Copia del arreglo "y" del device hacia el host
	cudaMemcpy(y_host,y_device,sizeof(cuDoubleComplex)*P*Dip*Dop,cudaMemcpyDeviceToHost);
	
	
	//Se imprimen los valores de "y"
	printf("\n\n--- ARREGLO y(n1,n2,k1) ---\n\n");
	for(k1 = 0;k1 < Dip;k1++) 
	{
		for(n1 = 0;n1 < Dop;n1++)
		{
			for(n2 = 0;n2 < P;n2++)
			{
				printf(" (%f) + (%f) ",cuCreal(y_host[(k1*Dop*P)+(n1*P)+n2]),cuCimag(y_host[(k1*Dop*P)+(n1*P)+n2]));
			}
			printf("\n");
		}
		printf("\n\n");
	}
	printf("\n");
	*/
	
}

//funci�n kernel que ejecuta la etapa de entrada en el device
__global__ void inputStage_kernel(int N, int Li,int Dip,int Dop,int P,cuDoubleComplex *x,cuDoubleComplex *W,cuDoubleComplex *y)
{
	int n1,n2;
	cuDoubleComplex t1;

	//Threads
	int n = blockDim.x *blockIdx.x + threadIdx.x;
	int k1 = blockDim.y *blockIdx.y + threadIdx.y;

	//Se resetean las flags
	//flag_inputstage_1_d[0] = 0;
	//flag_inputstage_2_d[0] = 0;
	//flag_inputstage_3_d[0] = 0;

	//printf("\n n = %d k1 = %d",n,k1);

	if( (n < (P*Dop)) && (k1 < Dip))
	{
		n2 = floorf(n/Dop);
		n1 = n - (Dop*n2);
		//Generaci�n de los elementos que dependen de x[0]
		if(n == 0)
		{
			y[(k1*Dop*P)+(0*P)+ 0] = x[0];

			///Flag
			//flag_inputstage_1_d[0] = 1;
			
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
				y[(k1*Dop*P)+(n1*P)+ n2] = cuCmul(W[((n*k1)%N)-1],t1);
			}

			///Flag
			//flag_inputstage_2_d[0] = 1;
		}
		//Rellenado de ceros para los elementos de "y" para Li <= n <= (P*Dop)-1
		if((n >= Li) && (n <= (P*Dop)-1))
		{
			y[(k1*Dop*P)+(n1*P)+ n2] = make_cuDoubleComplex(0.0,0.0);

			///Flag
			//flag_inputstage_3_d[0] = 1;
		}

		
		//printf("\n (%f) + (%f)\n ",cuCrealf(y[(k1*Dop*P)+(n1*P)+ n2]),cuCimagf(y[(k1*Dop*P)+(n1*P)+ n2]));
	}
}

//Funci�n auxiliar del host para calcular la etapa intermedia en el device
void etapa_intermedia(void)
{
	//////////////////////////////////////////////////////////////////////////
	////////////////////////////ETAPA INTERMEDIA//////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//Declaraci�n de variables locales
	int k1,k2,n1;
	int n[1] = {P};
	int inembed[1] = {P};
	int onembed[1] = {P};

	//Asignaci�n de memoria en el device para "z"
	cudaMalloc((void**)&z_device,P*Dip*Dop*sizeof(cuDoubleComplex));

	//Asignaci�n de memoria en el host para "z"
	//z_host = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*P*Dip*Dop);

	//Asignaci�n de memoria en el device para "in" y "out"
	cudaMalloc((void**)&in,sizeof(cufftDoubleComplex)*P*Dip*Dop);
	cudaMalloc((void**)&out,sizeof(cufftDoubleComplex)*P*Dip*Dop);

	//Se copia el arreglo "y" al arreglo "in"
	cudaMemcpy(in,y_device,sizeof(cuDoubleComplex)*P*Dip*Dop,cudaMemcpyDeviceToDevice);

	//Se crea un plan
	cufftHandle plan;
	cufftPlanMany(&plan,1,n,inembed,1,P,onembed,1,P,CUFFT_Z2Z,Dip*Dop);

	//Ejecuci�n del plan
	cufftExecZ2Z(plan,in,out,CUFFT_FORWARD);
	

	//Esperar que el kernel termine de ejecutarse totalmente
	cudaDeviceSynchronize();

	//Se copian los datos del arreglo "out" al arreglo "z_device"
	cudaMemcpy(z_device,out,sizeof(cufftDoubleComplex)*P*Dip*Dop,cudaMemcpyDeviceToDevice);

	//Se destruye el plan
	cufftDestroy(plan);

	//Se liberan los arreglos "in" y "out"
	cudaFree(in);
	cudaFree(out);

	/*
	//Se copian los datos del arreglo "z_device" al arreglo "z_host"
	cudaMemcpy(z_host,z_device,sizeof(cuDoubleComplex)*P*Dip*Dop,cudaMemcpyDeviceToHost);

	
	///Se imprimen los valores de z(n1,k2,k1)
	printf("\n\n--- ARREGLO z(n1,k2,k1) ---\n\n");
	for(k1 = 0;k1 < Dip;k1++) 
	{
		for(n1 = 0;n1 < Dop;n1++)
		{
			for(k2 = 0;k2 < P;k2++)
			{
				printf(" (%f) + (%f) ",cuCreal(z_host[(k1*Dop*P)+(n1*P)+k2]),cuCimag(z_host[(k1*Dop*P)+(n1*P)+k2]));
			}
			printf("\n");
		}
		printf("\n\n");
	}
	printf("\n");
	*/
	
}

//Funci�n auxiliar del host para calcular la etapa de salida en el device
void etapa_salida(void)
{
	//////////////////////////////////////////////////////////////////////////
	////////////////////////////ETAPA DE SALIDA///////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//Declaraci�n de variables locales
	int m;

	//Asignaci�n de memoria en el device para "X"
	cudaMalloc((void**)&X_device,Lo*sizeof(cuDoubleComplex));

	//Asignaci�n de memoria en el host para "X"
	X_host = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*Lo);

	//Dimensionamiento del grid para la funci�n kernel "outputStage"
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
	outputStage_kernel<<<gridDim,blockDim>>>(N,Lo,Dip,Dop,P,z_device,W_device,X_device);

	//Esperar que el kernel termine de ejecutarse totalmente
	cudaDeviceSynchronize();

	//Copia del arreglo "X" del device hacia el host
	cudaMemcpy(X_host,X_device,sizeof(cuDoubleComplex)*Lo,cudaMemcpyDeviceToHost);
			
	/*
	//Se imprimen los valores de "X_host"
    ///Imprimir X[k]
    printf("\n\n--- ARREGLO X[k] ---\n\n");
    for(m=0;m<=Lo-1;m++)
    {
        printf("\n X[%d] = %f + (%f)",m,cuCreal(X_host[m]),cuCimag(X_host[m]));
        //fprintf(da,"%.4f %.4f\n",creal(X[i]),cimag(X[i]));
    }
	*/
	


}

//funci�n kernel que ejecuta la etapa de salida en el device
__global__ void outputStage_kernel(int N,int Lo,int Dip,int Dop,int P,cuDoubleComplex *z,cuDoubleComplex *W,cuDoubleComplex *X)
{
	//Declaraci�n de variables locales
	int n1,k_aux,k1,k2,a,b;
	cuDoubleComplex	t1,t2,t3,t4,t5;

	//Threads
	int k = blockDim.x *blockIdx.x + threadIdx.x;

	//Se resetean las flags
	//flag_outputstage_1_d[0] = 0;
	//flag_outputstage_2_d[0] = 0;
	//flag_outputstage_3_d[0] = 0;

	if(k < Lo)
	{
		for(n1 = 0; n1 <= (Dop-1); n1 = n1+1)
		{
			if(Lo <= Dip)
			{
				//C�lculo de X(k) para 0<=k<=Lo-1.
				//printf("\n--- Caso (Lo <= Dip) ---\n");
				//En la descomposici�n k = k1 + Dipk2; k2 = 0, y por lo tanto, k = k1
				if(n1 == 0) //Caso para lograr que por lo menos ingrese una vez 
				{
					X[k] = z[(k*Dop*P)+(0*P) + 0];
					
					///Flag
					//flag_outputstage_1_d[0] = 1;
				}
				else
				{
					if(n1 == 1)
					{
						X[k] = z[(k*Dop*P)+(0*P) + 0];
					}
					X[k] = cuCadd(z[(k*Dop*P)+(n1*P) + 0],X[k]);

					///Flag
					//flag_outputstage_1_d[0] = 1;

				}

			}
			else
			{
				if((k >= 0) && (k <= (Dip-1)))
				{
					//C�lculo de X(k) para 0<=k<=Dip-1.
					//En la descomposici�n k = k1 + Dipk2; k2 = 0, y por lo tanto, k = k1
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
						X[k] = cuCadd(z[(k*Dop*P)+(n1*P) + 0],X[k]);

					}
					

				}
				else
				{
					
					if(Dop <= 4) 
					{
						//Usando el m�todo directo
						//printf("\n--- Caso (Metodo directo) ---\n");
						if(n1 == 0) //Caso para lograr que por lo menos ingrese una vez 
						{
							k_aux = k-((Dip*P)*floorf(k/(Dip*P)));
							k2 = floorf(k_aux/Dip);
							k1 = k_aux-(Dip*k2);
							X[k] = z[(k1*Dop*P)+(0*P)+ (k2%P)];
							//printf("\nk = %d,k_aux = %d,k2 = %d,k1 = %d",k,k_aux,k2,k1);
							///Flag
							//flag_outputstage_2_d[0] = 1;

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
							X[k] = cuCadd(X[k],cuCmul(z[(k1*Dop*P)+(n1*P)+ (k2%P)],W[((n1*(k2+P*(a))*Dip)%N)-1]));

							///Flag
							//flag_outputstage_2_d[0] = 1;

						}

					}
					
					
					
					else
					{
						//Usando el m�todo filtering 2BF
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
								t4 = cuCmul(t1,make_cuDoubleComplex(2*cuCreal(W[(((k2+(P*(b)))*Dip)%N)-1]),0.0));
								/*
								if(k == 256)
								{
									printf("\nW = %d, k = %d,k_aux = %d,k2 = %d,k1 = %d, b= %d,z= %d",(((k2+(P*(b)))*Dip)%N)-1,k,k_aux,k2,k1,b,(k1*Dop*P)+((Dop-1)*P)+ (k2%P));
								}
								*/

								///Flag
								//flag_outputstage_3_d[0] = 1;
							}

							if((n1 >= 1) && (n1 <= (Dop-2)))
							{
								t2 = t1;
								t1 = cuCadd(z[(k1*Dop*P)+((-(n1-(Dop-1)))*P)+ (k2%P)],t4); 
								t3 = cuCmul(t1,make_cuDoubleComplex(2*cuCreal(W[(((k2+(P*(b)))*Dip)%N)-1]),0.0));
								t4 = cuCsub(t3,t2);
								/*
								if(k == 256)
								{
									printf("\nW= %d",(((k2+(P*(b)))*Dip)%N)-1);
								}
								*/
							}

							if(n1 == (Dop-1))
							{
								t5 = cuCadd(z[(k1*Dop*P)+(k2%P)],t4);
								X[k] = cuCsub(t5,cuCmul(t1,cuConj(W[(((k2+(P*(b)))*Dip)%N)-1])));
								


								
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
								//flag_outputstage_3_d[0] = 1;
							}
							else
							{
								k_aux = k-((Dip*P)*floorf(k/(Dip*P)));
								k2 = floorf(k_aux/Dip);
								k1 = k_aux-(Dip*k2);
								t1 = z[(k1*Dop*P)+((Dop-1)*P)+ (k2%P)];
								b = floorf(k/(Dip*P));
								t4 = cuCmul(t1,make_cuDoubleComplex(2*cuCreal(W[(((k2+(P*(b)))*Dip)%N)-1]),0.0));
								t5 = cuCadd(z[(k1*Dop*P)+(0*P)+ (k2%P)],t4);
								X[k] = cuCsub(t5,cuCmul(t1,cuConj(W[(((k2+(P*(b)))*Dip)%N)-1])));

								///Flag
								//flag_outputstage_3_d[0] = 1;
							}

						}
						
					}
					
					

				}
				

			}
				
		}
	}
}