#include <stdio.h>
#include <string.h>  /* For strlen             */
#include <mpi.h>     /* For MPI functions, etc */ 
#include <iostream>
#include <random>
#include <time.h>
using namespace std;

int ct = 0; // For creating random numbers on the same rank.


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
    cout << endl;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            cout << o[i * c + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void mult_2_matrix(int* mat1, int m1_r, int m1_c,int* mat2, int m2_r,int m2_c, int start, int end, int* output) {
    /*
        Multiplies two matricies together.
        This function assumes that the 2 matricies are
        stored as a 1D array.
    */
    for (int i = start; i < end; i++) {
        for (int j = 0; j < m2_c; j++) {
            int sum = 0;
            for (int k = 0; k < m2_r; k++) {
                sum += mat1[i * m1_c+  k] * mat2[k * m2_c + j];
            }
            output[i * m2_c + j] = sum;
        }
    }
}

int main(void) {
    int        comm_sz;               /* Number of processes    */
    int        my_rank;               /* My process rank        */
    MPI_Comm comm = MPI_COMM_WORLD;  

    // Initalize the two matricies and their output
    int* matrix_one, m1_rows, m1_columns;
    int* matrix_two, m2_rows, m2_columns;
    int* output;
    int remainder = 0;      // Remainder needed to add calculations for last core
                            // If not evenly divisble


    /* Start up MPI */
    MPI_Init(NULL, NULL);

    /* Get the number of processes */
    MPI_Comm_size(comm, &comm_sz);

    /* Get my rank among all the processes */
    MPI_Comm_rank(comm, &my_rank);

    if (my_rank == 0) {
        // Enter values
        cout << "Please enter the number of rows and columns for Matrix 1: \n";
        cin >> m1_rows >> m1_columns;
        cout << "Please enter the number of rows and columns for Matrix 2: \n";
        cin >> m2_rows >> m2_columns;

        if (m1_columns != m2_rows) {
            cout << "Values do not match.\n";
            return 0;
        }

        if (m1_rows < comm_sz) {
            cout << "The number of rows must be at least the number of cores you will use. \n";
            return 0;
        }
    }

    // Init each matrix for all cores
    MPI_Bcast(&m1_rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&m2_rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&m1_columns, 1, MPI_INT, 0, comm);
    MPI_Bcast(&m2_columns, 1, MPI_INT, 0, comm);
    matrix_one = new int[m1_rows * m1_columns];
    matrix_two = new int[m2_rows * m2_columns];
    output = new int[m1_rows * m2_columns];

    // Init output to 0
    // Allows us to use MPI_Reduce() on the sum of elements
    for (int i = 0; i < m1_rows * m2_columns; i++) {
        output[i] = 0;
    }

    if (my_rank == 0) {
        // Generate and print arrays
        fill(m1_rows, m1_columns, matrix_one);
        fill(m2_rows, m2_columns, matrix_two);

        remainder = m1_rows % comm_sz;
    }

    // Broadcast for the function later
    MPI_Bcast(matrix_one, m1_rows * m1_columns, MPI_INT, 0, comm);
    MPI_Bcast(matrix_two, m2_rows * m2_columns, MPI_INT, 0, comm);
    MPI_Bcast(output, m1_rows * m2_columns, MPI_INT, 0, comm);
    MPI_Bcast(&remainder, 1, MPI_INT, 0, comm);
    

    // Calculate starting and end point
    // based on an interval
    int local_rows = m1_rows / comm_sz;
    int local_a = my_rank * local_rows;
    int local_b = local_a + local_rows;
    int* local_output = new int [local_rows * m2_columns];

    
    // If not evenly divisble let last core handle the rest of the rows
    if (my_rank == comm_sz - 1)
        local_b += remainder;

    double local_start, local_finish, local_elapsed, elapsed;
    MPI_Barrier(comm);
    local_start = MPI_Wtime();

    // Call function
    mult_2_matrix(matrix_one, m1_rows, m1_columns, matrix_two, m2_rows, m2_columns, local_a, local_b, output);

    local_finish = MPI_Wtime();
    local_elapsed = local_finish - local_start;

    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if(my_rank == 0)
        printf("Elapsed time: %e seconds\n", elapsed);

    // Get result and print
    int* res = new int[m1_rows * m2_columns];
    MPI_Reduce(output, res, m1_rows * m2_columns, MPI_INT, MPI_SUM, 0, comm);

    if (my_rank == 0) {
        /*printf("MAT 1: \n");
        print(m1_rows, m1_columns, matrix_one);
        printf("MAT 2: \n");
        print(m2_rows, m2_columns, matrix_two);
        printf("RES MAT: \n");
        print(m1_rows, m2_columns, res);*/
    }
    /* Shut down MPI */
    MPI_Finalize();

    return 0;
}
