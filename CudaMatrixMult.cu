
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;


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

void verify(int* a,int a_r, int a_c, int b_r, int b_c, int* b, int* output) {
    /*
        Multiplies two matricies together.
        This function assumes that the 2 matricies are
        stored as a 1D array.
    */
    for (int i = 0; i < a_r; i++) {
        for (int j = 0; j < b_c; j++) {
            int sum = 0;
            for (int k = 0; k < b_r; k++) {
                sum += a[i * a_c + k] * b[k * b_c + j];
            }
            output[i * b_c + j] = sum;
        }
    }
    
}

__global__ void matrixMult(int*a, int a_r, int a_c, int b_r, int b_c, int* b, int* output) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.y;
    
    if (row < a_r && col < b_c) {
        int sum = 0;
        for (int k = 0; k < b_r; k++)
            sum += a[row * a_c + k] * b[k * b_c + col];

        output[row * b_c + col] = sum;
    }

}

int main(){
    // Create matrices a and b, and resulting matrix c
    int* a, a_r, a_c, a_n, *dev_a;
    int* b, b_r, b_c, b_n, *dev_b;
    int* c, c_n, *dev_c;

    // User input
    cout << "Please enter rows and cols for matrix 1: ";
    cin >> a_r >> a_c;  // Enter values for row and columns
    a_n = a_r * a_c;    // Calculate size
    cout << "Next matrix: ";
    cin >> b_r >> b_c;
    while (b_r != a_c) {
        cout << "Columns of Mat 1 and Rows of Mat 2 must match. Try Again: \n";
        cin >> b_r >> b_c;
    }
    b_n = b_r * b_c;
    c_n = a_r * b_c;
    cout << endl;

    // Initialize all matrices
    a = new int[a_n];
    b = new int[b_n];
    c = new int[c_n];

    // Fill a and b with random numbers
    fill(a_r, a_c, a);
    fill(b_r, b_c, b);

    // Initialize matrices on device
    cudaMalloc(&dev_a, a_n * sizeof(int));
    cudaMalloc(&dev_b, b_n * sizeof(int));
    cudaMalloc(&dev_c, c_n * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(dev_a, a, a_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, b_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, c_n * sizeof(int), cudaMemcpyHostToDevice);
   
    // Create Timer
    cudaEvent_t     start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Values for threads and blocks
    dim3 thread(32, 32, 1);
    dim3 block(100, 100, 1);

    // Mult Matr
    matrixMult << < block, thread >> > (dev_a, a_r, a_c, b_r, b_c, dev_b, dev_c);
    
    // Stop timer, calc time and print
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    printf("\nFinished Calculating With Cuda.\n");
    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to do calculate:  %3.1f ms\n", elapsedTime);
    cudaMemcpy(c, dev_c, c_n * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify
    printf("\nStarting Verification...\n");
    bool isSame = true;
    int* ver = new int[c_n];
    verify(a, a_r, a_c, b_r, b_c, b, ver);

    for (int i = 0; i < c_n; i++) {
        if (c[i] != ver[i]) {
            isSame = false;
            printf("i: %d, c[i]: %d, ver[i]: %d\n", i, c[i], ver[i]);
            break;
        }
    }

    // Print verification boolean
    if (isSame)
        cout << "Success.\n";
    else
        cout << "Failed...\n";
}
