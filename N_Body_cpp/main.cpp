#include <iostream>
#include <fstream>
#include <cmath>
#include <eigen-5.0.0/Eigen/Dense>
#include <format>
using namespace std;
using namespace Eigen;

Eigen::MatrixXd readFileToEigenMatrix(const string& filename) {
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return Eigen::MatrixXd();
    }

    // Read the file line by line to determine the dimensions
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        double value;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();

    // Check if the file is empty
    if (data.empty()) {
        std::cerr << "The file is empty." << std::endl;
        return Eigen::MatrixXd();
    }

    // Determine the number of rows and columns
    int rows = data.size();
    int cols = data[0].size();

    // Check if all rows have the same number of columns
    for (const auto& row : data) {
        if (row.size() != cols) {
            std::cerr << "Inconsistent number of columns in the file." << std::endl;
            return Eigen::MatrixXd();
        }
    }

    // Create an Eigen matrix and fill it with the data
    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}



int main() {
    string filename = "C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data0.txt"
    MatrixXd Matrix_test{readFileToEigenMatrix(filename)};
    cout << "test:" << Matrix_test << endl;
    return 0;
}