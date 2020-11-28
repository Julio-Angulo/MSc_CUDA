///Este programa calcula el algoritmo QR. Se tomaron 13 muestras de tiempo promedio (con 100 iteraciones) con matrices cuadradas de
///2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 y 16384
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>


void main(void)
{
    int i,j,loop,ii,jj,kk;
    unsigned long N,M;
	float r,c,s,temp;
    float suma;

    ///Número de iteraciones
    loop=100;

    ///Número de muestras
    int const muestras=6;


    float promedio[muestras];

    ///Se crean los archivos binarios donde se guardarán los datos
    FILE *da;
    da = fopen("Algoritmo_QR_version_C.bin","a+b"); //Crea o sobre escribe archivo

    ///Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();

    ///Se inicializa N
    N=1;

    for(i=1;i<=muestras;i++)
    {
        ///Tamaño de la matriz "A"
        N = pow(2,i*2);
        M = N;
        suma=0.0;
        for(j=0;j<loop;j++)
        {
            LARGE_INTEGER frequency; //Ticks por segundo
            LARGE_INTEGER t1,t2;     //Ticks
            float elapsedTime;


            ///Se obtienen los ticks por segundo
            QueryPerformanceFrequency(&frequency);

            ///Declaración de variables
			float** A;
			

            ///Se reserva memoria en host
            A = (float**) malloc(sizeof(float*)*N);
            for(ii=0;ii<N;ii++)
            {
                A[ii] = (float*) malloc(sizeof(float)*M);
                if(A[ii] == NULL)
                {
                    printf("\nMemoria insuficiente\n");
                }
            }
            
			///Se dan valores a la matriz "A"
            
			for(ii=0;ii<N;ii++)
            {
                for(jj=0;jj<M;jj++)
                {
                    //A[jj +(ii*N)]=rand()%10;
					A[ii][jj]=rand()%10;
					//A[ii][jj]= (ii + 1) + (jj + 1);
                }
            }

			/*
			///Se imprime la matriz A
            printf("MATRIZ A\n\n");
            for(ii=0;ii<N;ii++)
            {
                for(jj=0;jj<M;jj++)
                {
                    printf("%f ",A[ii][jj]);
                }
                printf("\n");
            }
			*/


            ///------------------------------------------------------------------------------
            ///Se empieza a medir el tiempo
            //tiempo_1=clock();
            //gettimeofday(&tini, NULL);
            QueryPerformanceCounter(&t1);

            ///Se ejecuta el algoritmo QR secuencial 
            for(kk = 0;kk <= N-1; kk++)
            {
                for(ii = M-1;ii >= kk + 1;ii--)
                {
					r = sqrt(pow(A[ii-1][kk],2) + pow(A[ii][kk],2));
					c = A[ii-1][kk]/r;
					s = A[ii][kk]/r;

					for(jj = kk;jj <= N - 1;jj++)
					{
						temp = A[ii-1][jj];
						A[ii-1][jj] = c*temp + s*A[ii][jj];
						A[ii][jj] = -s*temp + c*A[ii][jj];
					}
				}
            }

            ///------------------------------------------------------------------------------
            ///Se termina de medir el tiempo
            //gettimeofday(&tfin, NULL);
            //ltiempo= (tfin.tv_sec - tini.tv_sec)*1000000 + tfin.tv_usec - tini.tv_usec;
            //ftiempo= ltiempo/1000.0;
            QueryPerformanceCounter(&t2);

            ///Se calcula el tiempo en milisegundos
            elapsedTime = (t2.QuadPart - t1.QuadPart)* 1000.0/ frequency.QuadPart;

            /*
            ///Se imprime la matriz R
            printf("MATRIZ R\n\n");
            for(ii=0;ii<M;ii++)
            {
                for(jj=0;jj<N;jj++)
                {
                    printf("%f ",A[ii][jj]);
                }
                printf("\n");
            }
			*/

			///Se liberan las memorias
            for(ii=0;ii<N;ii++)
            {
                free(A[ii]);
            }
            
            free(A);
            
			suma = suma + elapsedTime;

        }
        promedio[i-1] = suma/(float)loop;
        printf("\n%d - Tiempo promedio para N = %ld >>> %f mS\n",i,N,promedio[i-1]);

    }
    fwrite(promedio,sizeof(float),muestras,da);
    fclose(da);

}
