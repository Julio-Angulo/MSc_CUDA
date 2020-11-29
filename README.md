# MSc_CUDA
This repository contains all CUDA C codes written during my MSc. Next, there is a brief description of what each code does. Further information about results, parallelization techniques, and data dependence theory found [**here**](https://www.researchgate.net/profile/Julio_Angulo_Rios/publication/316692653_A_GPU-Based_Implementation_of_the_Input_andor_Output_Pruning_of_Composite_Length_FFTs_Using_a_DIF-DIT_Transform_Decomposition/links/590cd2d0aca2722d185c131c/A-GPU-Based-Implementation-of-the-Input-and-or-Output-Pruning-of-Composite-Length-FFTs-Using-a-DIF-DIT-Transform-Decomposition.pdf).


| #| NAME| DESCRIPTION|
| -- | ----- | ---- |
| 1 | Algoritmo_QR_CUDA_1_thread | Find the QR algorithm of a matrix "A" (CUDA C 1 thread version) |
| 2 | Algoritmo_QR_CUDA_2D_version_2 | Find the QR algorithm of a matrix “A” (CUDA C version 2D parallel) |
| 3 | Algoritmo_QR_CUDA_pol_destino_1_thread_2D | Find the QR algorithm of an “A” matrix (CUDA C 1 thread version, 2D destination polytope) |
| 4 | Algoritmo_QR_version_C | Find the QR algorithm of a matrix “A” (13 samples of average time (100 iterations) were taken with square matrices) |
| 5 | AlgoritmoGoertzelDavidV3 | Calculate the DFT using the Goertzel algorithm |
| 6 | cufft_1D | Calculation of 1D FFT using "cufftPlan1d" |
| 7 | cufft_2D | 2D FFT calculation using "cufftPlanMany" |
| 8 | cufft_3D | 3D FFT calculation using "cufftPlanMany" |
| 9 | cufft_3D_2 | 3D FFT calculation using "cufftPlanMany" (no transpose) |
| 10 | CUFFT_NOPRUN_N20 | This program calculates the parallel version of the cuFFT library (without pruning) for N = 2 ^ 20 |
| 11 | CUFFT_NOPRUN_NCOMPUESTA | This program calculates the parallel version of the cuFFT library (without pruning) for N = 2 ^ 5 x 3 ^ 4 x 5 ^ 4 (N-Composite) |
| 12 | cufft_planmany_radix_2 | Measurement of time to execute a 1D FFT, radix-2. The command "cufftPlanMany" will be used |
| 13 | cufft_planmany_radix_3 | Measurement of time to execute a 1D FFT, radix-3. The command "cufftPlanMany" will be used |
| 14 | cufftPlan2D | Calculation of the 2D FFT using the cufftPlan2D () function |
| 15 | cufftPlan3D | Calculation of the 3D FFT using the cufftPlan3D () function |
| 16 | cufftw_1D | Calculation of 1D FFT using "fftwf_plan_dft_1d" |
| 17 | cufftw_2D | Calculation of the 2D FFT using the function "fftwf_plan_dft_2d" |
| 18 | cufftw_3D | Calculation of the 3D FFT using the function "fftwf_plan_dft_3d" |
| 19 | cufftw_planmany_3D | Calculation of 3D FFT using "cufftw_Plan_Many_3d" |
| 20 | cufftw_planmany_radix_2 | Measurement of time to execute a 1D FFT, radix-2. The command "fftwf_plan_many_dft" will be used |
| 21 | cufftw_planmany_radix_3 | Measurement of time to execute a 1D FFT, radix-3. The command "fftwf_plan_many_dft" will be used |
| 22 | ejemplo_1 | Calculate the sum of two vectors (a and b) and store the result in "c" |
| 23 | Ejemplo_2 | Calculate the number of multiprocessors of the device |
| 24 | ejemplo_3 | Program that adds two vectors (a and b) and stores the result in the vector "c" |
| 25 | ejemplo_4 | Calculate the sum of two vectors (a and b) and store the result in "c" |
| 26 | Ejemplo_texture_3D | Calculates the sum of two matrices (A and B) and stores the result in "C" (uses texture memory) |
| 27 | FFT_3D_paralelismo dinamico | Calculation of the 3D FFT using "cufftw_Plan_Many_dft" and dynamic parallelism |
| 28 | FFT_DIF_DIT_TD_N729_Li43_LoVARIA | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to find in matlab the necessary number of iterations, considering (RADIX-3) N = 729, Li = 43 and Lo = {3,9,27, ..., 729} |
| 29 | FFT_DIF_DIT_TD_N729_LiN_LoVARIA | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to find in matlab the necessary number of iterations, considering (RADIX-3) N = 729, Li = N and Lo = {3,9,27, ..., 729} |
| 30 | FFT_DIF_DIT_TD_N729_LiVARIA_Lo43 | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to find in matlab the necessary number of iterations, considering (RADIX-3) N = 729, Li = {3,9,27, ..., 729} and Lo = 43 |
| 31 | FFT_DIF_DIT_TD_N729_LiVARIA_LoN | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to find in matlab the necessary number of iterations, considering (RADIX-3) N = 729, Li = {3,9,27, ..., 729} and Lo = N |
| 32 | FFT_DIF_DIT_TD_N1024_Li33_LoVARIA | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to find in matlab the necessary number of iterations, considering (RADIX-2) N = 1024, Li = 33 and Lo = {2,4,8, ..., 1024} |
| 33 | FFT_DIF_DIT_TD_N1024_LiN_LoVARIA | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to find in matlab the necessary number of iterations, considering (RADIX-2) N = 1024, Li = N and Lo = {2,4,8, ..., 1024} |
| 34 | FFT_DIF_DIT_TD_N1024_LiVARIA_Lo33 | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to find in matlab the necessary number of iterations, considering (RADIX-2) N = 1024, Li = {2,4,8, ..., 1024} and Lo = 33 |
| 35 | FFT_DIF_DIT_TD_N1024_LiVARIA_LoN | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to find in matlab the necessary number of iterations, considering (RADIX-2) N = 1024, Li = {2,4,8, ..., 1024} and Lo = N |
| 36 | FFT_DIF_DIT_TD_VERSION_1THREAD_R2 | This program calculates the 1-thread version of the FFT_DIF_DIT_TD algorithm. This version is used to graph absolute and relative errors in matlab (RADIX-2) 2^1 - 2^10 |
| 37 | FFT_DIF_DIT_TD_VERSION_1THREAD_R2_2 | This program calculates the 1-thread version of the FFT_DIF_DIT_TD algorithm. This version is used to graph execution times in matlab, considering (RADIX-2) N = 16,384, Li = N and Lo = {2,4,8, ..., 16,384} |
| 38 | FFT_DIF_DIT_TD_VERSION_1THREAD_R2_3 | This program calculates the 1 thread version of the FFT_DIF_DIT_TD algorithm. This version is used to graph execution times in matlab, considering (RADIX-2) N = 16,384, Li = 33 and Lo = {2,4,8, ..., 16384} |
| 39 | FFT_DIF_DIT_TD_VERSION_1THREAD_R3 | This program calculates the 1-thread version of the FFT_DIF_DIT_TD algorithm. This version is used to graph absolute and relative errors in matlab (RADIX-3) 3^1 - 3^6 |
| 40 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to calculate the execution times for different numbers of samples and iterations |
| 41 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_2 | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to graph absolute and relative errors in matlab (RADIX-2) 2^1 - 2^10 |
| 42 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_3 | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to graph absolute and relative errors (RADIX-3) 3^1 - 3^6 in matlab |
| 43 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_4 | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to graph absolute and relative errors (RADIX-2) 2^1 - 2^10 in matlab. The x_host and W_host arrays are generated on the device and the assign_rap function is executed on the device |
| 44 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_6 | Graph the execution times in Matlab, considering Radix-2. N = 220, Li = N, Lo = {2, 4,…, N}. (simple precision) |
| 45 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_6_DO | Graph the execution times in Matlab, considering Radix-2. N = 220, Li = N, Lo = {2, 4,…, N}. (double precision) |
| 46 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_7 | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Li = 33, Lo = {2, 4,…, N}. (simple precision) |
| 47 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_7_DO | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Li = 33, Lo = {2, 4,…, N}. (double precision) |
| 48 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_8 | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Li = 524,000, Lo = {2, 4,…, N}. (simple precision) |
| 49 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_8_DO | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Li = 524,000, Lo = {2, 4,…, N}. (double precision) |
| 50 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_9 | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Lo = N, Li = {2, 4,…, N}. (simple precision) |
| 51 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_9_DO | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Lo = N, Li = {2, 4,…, N}. (double precision) |
| 52 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_10 | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Lo = 33, Li = {2, 4,…, N}. (simple precision) |
| 53 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_10_DO | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Lo = 33, Li = {2, 4,…, N}. (double precision) |
| 54 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_11 | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Lo = 524,000, Li = {2, 4,…, N}. (simple precision) |
| 55 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_11_DO | Graph the execution times in Matlab, considering Radix-2. N = 2^20, Lo = 524,000, Li = {2, 4,…, N}. (double precision) |
| 56 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_12 | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Li = N, Lo = {3, 9, ..., N}. (simple precision) |
| 57 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_12_DO | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Li = N, Lo = {3, 9, ..., N}. (double precision) |
| 58 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_13 | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Li = N, Lo = {3, 9, ..., N}. (double precision) |
| 59 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_13_DO | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Li = 43, Lo = {3, 9, ..., N}. (double precision) |
| 60 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_14 | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Li = 797,000, Lo = {3, 9,…, N}. (simple precision) |
| 61 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_14_DO | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Li = 797,000, Lo = {3, 9,…, N}. (double precision) |
| 62 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_15 | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Lo = N, Li = {3, 9, ..., N}. (simple precision) |
| 63 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_15_DO | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Lo = N, Li = {3, 9, ..., N}. (double precision) |
| 64 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_16 | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Lo = 43, Li = {3, 9, ..., N}. (simple precision) |
| 65 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_17 | Graph the execution times in Matlab, considering Radix-3. N = 3^13, Lo = 797,000, Li = {3, 9,…, N}. (simple precision) |
| 66 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_18 | This version is used to detect errors in the stages with the help of the flags, considering (RADIX-3) N = 3^13, Li = Varies and Lo = 43 | 67 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_19 | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to graph absolute and relative errors in matlab Case: N^20, Li = 307, Lo = N | 
| 68 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_19_DO | This program calculates the parallelized version of the algorithm FFT_DIF_DIT_TD (VARIABLES TYPE DOUBLE). This version is used to graph absolute and relative errors in matlab Case: N^20, Li = 524,000, Lo = N |
| 69 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_20 | This program calculates the parallelized version of the FFT_DIF_DIT_TD algorithm. This version is used to graph absolute and relative errors in matlab Case: N=3^13, Li = 264, Lo = N |
| 70 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_21 | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = 45, Lo = varies. (simple precision) |
| 71 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_21_DO | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = 45, Lo = varies. (double precision) |
| 72 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_22 | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = 800,000, Lo = varies. (simple precision) |
| 73 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_22_DO | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = 800,000, Lo = varies. (double precision) |
| 74 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_23 | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = N, Lo = varies. (simple precision) |
| 75 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_23_DO | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = N, Lo = varies. (double precision) |
| 76 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_24 | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Lo = 45, Li = varies. (simple precision) |
| 77 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_24_DO | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Lo = 45, Li = varies. (double precision) |
| 78 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_25 | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Lo = 800,000, Li = varies. (simple precision) |
| 79 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_25_DO | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Lo = 800,000, Li = varies. (double precision) |
| 80 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_26 | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Lo = N, Li = varies. (simple precision) |
| 81 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_26_DO | Graph the execution times in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Lo = N, Li = varies. (double precision) |
| 82 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_27 | Graph the relative and absolute errors in Matlab, considering Radix-2. N = 2^20, Li = N, Lo = 524,000. (simple precision) |
| 83 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_27_DO | Graph the relative and absolute errors in Matlab, considering Radix-2. N = 2^20, Li = N, Lo = 524,000. (double precision) |
| 84 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_28 | Graph the relative and absolute errors in Matlab, considering Radix-2. N = 2^20, Li = 524,000, Lo = 524,000. (simple precision) |
| 85 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_28_DO | Graph the relative and absolute errors in Matlab, considering Radix-2. N = 2^20, Li = 524,000, Lo = 524,000. (simple precision) |
| 86 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_29 | Graph the relative and absolute errors in Matlab, considering Radix-3. N = 3^13, Li = N, Lo = 797,000. (simple precision) |
| 87 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_30 | Graph the relative and absolute errors in Matlab, considering Radix-3. N = 3^13, Li = 797,000, Lo = 797,000. (simple precision) |
| 88 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_31 | Graph the relative and absolute errors in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = 800,000 Lo = N. (simple precision) |
| 89 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_31_DO | Graph the relative and absolute errors in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = 800,000 Lo = N. (double precision) |
| 90 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_32 | Graph the relative and absolute errors in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = N, Lo = 800,000. (simple precision) |
| 91 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_32_DO | Graph the relative and absolute errors in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = N, Lo = 800,000. (double precision) |
| 92 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_33 | Graph the relative and absolute errors in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = 800,000, Lo = 800,000. (simple precision) |
| 93 | FFT_DIF_DIT_TD_VERSION_PARALELIZADA_33_DO | Graph the relative and absolute errors in Matlab, considering N-Composite. N = (2^5) x (3^4) x (5^4), Li = 800,000, Lo = 800,000. (double precision) |
| 94 | fftw_guru_3D | Calculation of 3D FFT using "fftw_plan_guru_dft" ----- >>>> without transpose |
| 95 | matriz_transpuesta_2D | This program finds the transposed matrix of "A" and saves the result in matrix "B". (PARALLELISM VERSION IN TWO DIMENSIONS) |
| 96 | matriz_transpuesta_CUDA_1_thread | This program finds the transposed matrix of "A" and saves the result in matrix "B". (CUDA 1 THREAD SEQUENTIAL VERSION) |
| 97 | matriz_transpuesta_paralelismo_externo_M | This program finds the transposed matrix of "A" and saves the result in matrix "B". (EXTERNAL PARALLELISM VERSION WITH M THREADS (ROWS)) |
| 98 | matriz_transpuesta_paralelismo_externo_N | This program finds the transposed matrix of "A" and saves the result in matrix "B". (EXTERNAL PARALLELISM VERSION WITH N THREADS (COLUMNS)) |
| 99 | matriz_transpuesta_paralelismo_interno | This program finds the transposed matrix of "A" and saves the result in matrix "B". (INTERNAL PARALLELISM VERSION) |
| 100 | matriz_transpuesta_secuencial_C | This program calculates the transpose of a matrix "A" and the result is saved in matrix "B" |
| 101 | Mult_2D | Array Element Multiplication Using Surface 2D Memory |
| 102 | Mult_2D_Surface | Multiplication of two matrices (A and B) and the result is stored in "C" (Using Surface 2D memory) |
| 103 | Mult_2D_Texture | Multiplication of two matrices (A and B) and the result is stored in "C" (Using Texture 2D memory) |
| 104 | Mult_3D_Surface | Multiplication of two matrices (A and B) and the result is stored in "C" (Using Surface 3D memory) |
| 105 | Mult_3D_Surface2 | Multiplication of two matrices (A and B) and the result is stored in "C" (Using Surface 3D memory) (version 2) |
| 106 | Mult_3D_Texture | The elements of an “A” matrix are copied to a “C” matrix using the 3D texture memory |
| 107 | multiplicacion_matricial | Multiplication of two matrices (A and B) and the result is stored in "R" (Using Surface 2D memory) |
| 108 | multNoShare | Multiplication of two matrices (A and B) using linear memory |
| 109 | multShare | Multiplication of two matrices (A and B) using shared memory |
| 110 | parallelism_dynamic | Program that adds two vectors (a and b) and stores the result in the vector "c". Also, multiply two vectors (a and b) and stor the result in vector "d". (Use Dynamic Parallelism) |
| 111 | suma_vectores_1_thread | Sum of two vectors (a and b) using 1 thread |
| 112 | suma_vectores_paralelizado | Sum of two vectors (a and b) parallelized |