/***

This software preprocess data using Principal Component Analysis ( PCA ) exploiting CUDA.
Developed by Gianluca De Lucia ( gianluca.delucia.94@gmail.com ) and Diego Romano ( diego.romano@cnr.it )

***/

#include "kernel_pca.h"
#include <string>
#include <iostream>
#include <iomanip>


using namespace std;
void mainFunction(float* img, int K, int d0, int d1, int d2, float* imgT);

void mainFunction(float* img, int K, int d0, int d1, int d2, float* imgT){
		
		int M, N, m, n;

		// initialize srand and clock
	    srand (time(NULL));

		//from cube 3D to matrix 2D		
		M = d0*d1;
	    N = d2;
		double dtime;
		clock_t start;
		KernelPCA* pca;
		//float *T = (float*)malloc(sizeof(float)*d0*d1*K);
		float *T = (float*)malloc(sizeof(float)*d0*d1*K);
		float *T0 = (float*)malloc(sizeof(float)*d0*d1*d2);

		for (int i = 0; i < d0*d1*d2 ; ++i){
			T0[i] = img[i];
		}


		pca = new KernelPCA(K);

	    start=clock();

	    pca->fit_transform(M, N, img, 1,imgT);

	    dtime = ((double)clock()-start)/CLOCKS_PER_SEC;

		printf("\nTime for GS-PCA in CUBLAS: %f seconds\n", dtime);

	}


extern "C" {
    void cudaPCA(float* img,int K, int d0, int d1, int d2, float* imgT)
    {
        return mainFunction(img,K,d0,d1,d2, imgT);
    }
}



