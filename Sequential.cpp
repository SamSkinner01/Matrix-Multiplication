#include <iostream>
#include <ctime>
using namespace std;



// (m x n) x (n x p) => (m x p)
int mat1_m, mat1_n; // Size of matrix 1, where m is row and n is columns
int mat2_n, mat2_p; // Size of matrix 2, where n is row and p is cols

// Counter used for genreating random numbers differently/
int ct = 0;

void mult_2_matrix(int* mat1, int*mat2, int* output) {
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

int main() {

	/*
		Get user input for the size of both matricies.
		If matrix 1's # of columns does not equal matrix 2's # of rows, then 
		multiplication is not possible, so end the program with that message.
	*/
	cout << "Please enter the number of rows and columns for Matrix 1: \n";
	cin >> mat1_m >> mat1_n;
	cout << "Please enter the number of rows and columns for Matrix 2: \n";
	cin >> mat2_n >> mat2_p;

	if (mat1_n != mat2_n) {
		cout << "Values do not match.\n";
		return 0;
	}

	/*
		Initalizing the 1D arrays that will store the matrix.
	*/
	int* m1;
	int* m2;
	int* output;

	m1 = new int[mat1_m * mat1_n];
	m2 = new int[mat2_n * mat2_p];
	output = new int[mat1_m * mat2_p];

	/*
		Fill the the two matricies with random values.
		Print out the two matricies.
	*/
	cout << "\nMatrix 1: \n";
	fill(mat1_m, mat1_n, m1);
	//print(mat1_m, mat1_n, m1);

	cout << "Matrix 2: \n";
	fill(mat2_n, mat2_p, m2);
	//print(mat2_n, mat2_p, m2);


	/*
		Perform calculation
	*/
	auto begin = clock();
	mult_2_matrix(m1, m2, output);
	auto end = clock();

	cout << "Elapsed time:" << 0.001 * double(std::difftime(end, begin)) << " seconds\n";
	/*
		Print output to command line.
	*/
	cout << "Matrix 3: \n";
	//print(mat1_m, mat2_p, output);
}
