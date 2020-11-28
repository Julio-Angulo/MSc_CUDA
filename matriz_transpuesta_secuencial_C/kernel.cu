///Este programa calcula la transpuesta de una matriz "A" y el resultado se guarda en la matriz "B"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>


void main(void)
{
    int i,j,loop,muestras;
    unsigned long N,M,k,r;
    float suma;

    ///Número de iteraciones
    loop=100;

    ///Número de muestras
    muestras=14;


    float promedio[14];

    ///Se crean los archivos binarios donde se guardarán los datos
    FILE *da;
    da = fopen("Matriz_transpuesta_secuencial_C.bin","a+b"); //Crea o sobre escribe archivo

    ///Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();

    ///Se inicializa N
    N=1;

    for(i=1;i<=muestras;i++)
    {
        ///Tamaño de la matriz "A"
        N = N * 2;
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
            int** A;
            int** B;

            ///Se reserva memoria en host
            A = (int**) malloc(sizeof(int*)*N);
            B = (int**) malloc(sizeof(int*)*N);

            
            for(k=0;k<N;k++)
            {
                A[k] = (int*) malloc(sizeof(int)*M);
                B[k] = (int*) malloc(sizeof(int)*M);

                if(A[k] == NULL)
                {
                    printf("\nMemoria insuficiente\n");
                }
            }
            
			///Se dan valores a la matriz "A"
            for(k=0;k<N;k++)
            {
                for(r=0;r<M;r++)
                {
                    //A[r +(k*N)]=rand()%10;
					A[k][r]=rand()%10;
                }
            }

            ///------------------------------------------------------------------------------
            ///Se empieza a medir el tiempo
            //tiempo_1=clock();
            //gettimeofday(&tini, NULL);
            QueryPerformanceCounter(&t1);

            ///Se saca la matriz transpuesta y se guarda en la matriz "B"
            for(k=0;k<N;k++)
            {
                for(r=0;r<M;r++)
                {
                    //B[k + (r*N)]=A[r +(k*N)];
					B[r][k] = A[k][r];
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
            ///Se imprimen las matrices A y B
            printf("MATRIZ A\n\n");
            for(k=0;k<M;k++)
            {
                for(r=0;r<N;r++)
                {
                    printf("%d ",A[k][r]);
                }
                printf("\n");
            }

            printf("\n\nMATRIZ B\n\n");
            for(k=0;k<M;k++)
            {
                for(r=0;r<N;r++)
                {
                    printf("%d ",B[k][r]);
                }
                printf("\n");
            }
            */

            ///Se liberan las memorias
            for(k=0;k<N;k++)
            {
                free(A[k]);
                free(B[k]);
            }
            
            free(A);
            free(B);

            suma = suma + elapsedTime;

        }
        promedio[i-1] = suma/(float)loop;
        printf("\n%d - Tiempo promedio para N = %ld >>> %f mS\n",i,N,promedio[i-1]);

    }
    fwrite(promedio,sizeof(float),muestras,da);
    fclose(da);

}




