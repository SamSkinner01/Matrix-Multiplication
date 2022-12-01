#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

#define threads 32 // In 2D array 32 * 32 = 1024 threads

int ct = 0;

void fill(int r, int c, int* o) {
    /*
        Fills matrix with r rows and c columns.
        Generates matrix in a 1D form.
        ct variable is used to change the seed for randomizing.
        Calling this function twice without it will cause both arrays
        to be filled with the same values.
    */
    srand(time(NULL) + ct);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            o[i * c + j] = rand() % 10;
        }
    }
    ct++;
}

void print(int r, int c, int* o) {
    /*
        Will print out the matrix given r rows and c columns.
        Assumes the matrix is stored as a 1D array.
    */

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            cout << o[i * c + j] << " ";
        }
        cout << endl;
    }
    cout << endl << endl;
}

__global__ void mult_2_matrix(int* a, int* b, int* c, int a_r, int a_c, int b_r, int b_c, int size) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;


    c[row * b_c + col] = 0;

    while (row < a_r && col < b_c) {

        for (int k = 0; k < a_c; k++) {
            c[row * b_c + col] += a[row * a_c + k] * b[k * b_c + col];
        }

        row += gridDim.y * blockDim.y;
        //col += gridDim.x * blockDim.x;
        
    }


}

void verify(int* mat1, int mat1_m, int mat1_n, int mat2_n, int mat2_p, int* mat2, int* output) {
    /*
        Multiplies two matricies together.
        This function assumes that the 2 matricies are
        stored as a 1D array.
    */
    for (int i = 0; i < mat1_m; i++) {
        for (int j = 0; j < mat2_p; j++) {
            int sum = 0;
            for (int k = 0; k < mat2_n; k++) {
                sum += mat1[i * mat1_n + k] * mat2[k * mat2_p + j];
            }
            output[i * mat2_p + j] = sum;
        }

        
    }

}

int main() {
    int* a, a_r, a_c, a_n, * dev_a;
    int* b, b_r, b_c, b_n, * dev_b;
    int* c, c_n, * dev_c;

    cout << "Please enter rows and cols for matrix 1: ";
    cin >> a_r >> a_c;
    a_n = a_r * a_c;

    cout << "Next matrix: ";
    cin >> b_r >> b_c;
    b_n = b_r * b_c;
    c_n = a_r * b_c;
    cout << endl;
    int* dev_col;

    a = new int[a_n];
    b = new int[b_n];
    c = new int[c_n];

    fill(a_r, a_c, a);
    fill(b_r, b_c, b);
     //cout << endl;
     //print(a_r, a_c, a);
     //cout << endl;
     //print(b_r, b_c, b);
      /*for (int i = 0; i < a_r; i++) {
          for (int j = 0; j < b_c; j++) {
              c[i * b_c + j] = 0;
          }
      }*/

    cudaMalloc(&dev_a, a_n * sizeof(int));
    cudaMalloc(&dev_b, b_n * sizeof(int));
    cudaMalloc(&dev_c, c_n * sizeof(int));
    cudaMalloc(&dev_col, sizeof(int));

    cudaMemcpy(dev_a, a, a_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, b_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, c_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col, &a_c, sizeof(int), cudaMemcpyHostToDevice);


    mult_2_matrix << < (c_n * 1023)/1024, threads >> > (dev_a, dev_b, dev_c, a_r, a_c, b_r, b_c, c_n);



    cudaMemcpy(c, dev_c, c_n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("CUDA FINISHED\n");
    int* ver = new int[c_n];
    verify(a, a_r, a_c, b_r, b_c, b, ver);

    /*printf("ANSWER: \n");
     print(a_r, b_c, ver);

     printf("GOT: \n");
     print(a_r, b_c, c);*/

    for (int i = 0; i < c_n; i++) {
        if (c[i] != ver[i]) {
            cout << "FAILED FAILED\n";
            cout << c[i] << "\t" << ver[i] << " " << i;
            break;
        }
    }

    cout << endl;
    //print(a_r, b_c, c);



}
