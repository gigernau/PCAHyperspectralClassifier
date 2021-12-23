#include "kernel_pca.h"
// includes, GSL & CBLAS
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <string>
#include <iostream>
#include <vector>
#include <iomanip>
#include <matioCpp/matioCpp.h>

using namespace std;
void mainFunction(int K);

void mainFunction(int K){
		printf("\nStart PCA");
		int M, N, m, n;
		
	    // initialize srand and clock
	    srand (time(NULL));
		matioCpp::File input("PaviaU.mat");
		matioCpp::MultiDimensionalArray<double> matrix1 = input.read("paviaU").asMultiDimensionalArray<double>();

	        int d0 = matrix1.dimensions()[0];
		int d1 =matrix1.dimensions()[1];
		int d2 =matrix1.dimensions()[2];

		//from cube 3D to matrix 2D		
		M = d0*d1;
	        N = d2;

		double *T0 = (double*)malloc(sizeof(double)*d0*d1*d2);
		double *T = (double*)malloc(sizeof(double)*d0*d1*K);
		
		double max=8000;
		double min=0;

		for (size_t i = 0; i < d0 ; ++i)
		 for (size_t k = 0; k < d2; ++k)
		  for (size_t j = 0; j < d1; ++j)
		    
		      T0[(i*d1) + (j) + (k*d1*d0)] = matrix1({i,j,k});

		//Normalization
		for(int l =0; l<d0*d1*d2; l++){
			T0[l] = (T0[l] - min)/(max-min);
		}

		double dtime;
		clock_t start;

		
		KernelPCA* pca;

		pca = new KernelPCA(K);

	        start=clock();

	        T = pca->fit_transform(M, N, T0, 1);

	        dtime = ((double)clock()-start)/CLOCKS_PER_SEC;
		printf("\nTime for GS-PCA in CUBLAS: %f seconds\n", dtime);
		printf("\nStop PCA");
	        size_t d00,d11,dk;
	        d00 = d0;
	        d11 = d1;
	        dk = K;

		matioCpp::File file2 = matioCpp::File::Create("CublasPCA.mat");
		matioCpp::MultiDimensionalArray<double> mat("X", {d00,d11,dk});

	    	
	    	int o = 0;
		    for (size_t i = 0; i < d00; ++i)
		    {
			for (size_t k = 0; k < dk; ++k)
			{
			    for (size_t j = 0; j < d11; ++j)
			    {
			        mat({i,j,k}) = T[(i*d1) + (j) + (k*d1*d0)];
			        
			    }
			}
		    }

	    	file2.write(mat);

	}


extern "C" {
    void myFunction(int K)
    {
        return mainFunction(K);
    }
}



