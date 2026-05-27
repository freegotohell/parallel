cuda_code = r'''#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;
using namespace chrono;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                 << " - " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

string intToString(int n) {
    stringstream ss;
    ss << n;
    return ss.str();
}

vector<double> generateMatrix(int n) {
    vector<double> matrix(n * n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 100.0);

    for (int i = 0; i < n * n; ++i)
        matrix[i] = dist(gen);

    return matrix;
}

void saveMatrix(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename);
    file << n << endl;
    file << fixed << setprecision(15);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

vector<double> multiplyCPU(const vector<double>& A, const vector<double>& B, int n) {
    vector<double> C(n * n, 0.0);

    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i * n + j] += A[i * n + k] * B[k * n + j];

    return C;
}

__global__ void matrixMultiplyKernel(double* A, double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

bool verifyResult(const vector<double>& cpuResult, 
                  const vector<double>& gpuResult, 
                  int n,
                  double tolerance = 1e-6) {
    for (int i = 0; i < n * n; i++) {
        if (abs(cpuResult[i] - gpuResult[i]) > tolerance) {
            cout << "Mismatch at [" << i / n << "][" << i % n << "]" << endl;
            return false;
        }
    }
    return true;
}

void printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    cout << "\nNumber of CUDA devices: " << deviceCount << endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        cout << "\nDevice " << i << ": " << prop.name << endl;
        cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;
    }
}

int main() {
    printDeviceInfo();
    
    // Размеры матриц как в примере
    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int numSizes = 6;
    
    // Конфигурации блоков как в примере (только размеры threadDim)
    struct GridConfig {
        string name;
        int threadX, threadY;
    };
    
    GridConfig configs[] = {
        {"8x8", 8, 8},
        {"16x16", 16, 16},
        {"32x8", 32, 8},
        {"64x4", 64, 4},
        {"128x2", 128, 2},
        {"8x32", 8, 32},
        {"4x64", 4, 64},
        {"2x128", 2, 128}
    };
    int numConfigs = 8;
    
    ofstream resultsFile("cuda_results.csv");
    // Формат заголовка как в примере
    resultsFile << "Size,BlockConfig,Time_seconds" << endl;
    
    cout << "========================================" << endl;
    cout << "CUDA Matrix Multiply - Google Colab" << endl;
    cout << "========================================" << endl;
    
    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];
        
        cout << "\n========== Size " << n << "x" << n << " ==========" << endl;
        
        string folder = "cuda_data_" + intToString(n);
        string cmd = "mkdir -p " + folder;
        int ret = system(cmd.c_str());
        (void)ret;
        
        cout << "Generating matrices..." << endl;
        auto A_cpu = generateMatrix(n);
        auto B_cpu = generateMatrix(n);
        
        saveMatrix(folder + "/A.txt", A_cpu, n);
        saveMatrix(folder + "/B.txt", B_cpu, n);
        
        double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        size_t size = n * n * sizeof(double);
        
        CUDA_CHECK(cudaMalloc(&d_A, size));
        CUDA_CHECK(cudaMalloc(&d_B, size));
        CUDA_CHECK(cudaMalloc(&d_C, size));
        
        cout << "Copying data to GPU..." << endl;
        CUDA_CHECK(cudaMemcpy(d_A, A_cpu.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B_cpu.data(), size, cudaMemcpyHostToDevice));
        
        cout << "CPU multiplication (for verification)..." << endl;
        auto C_cpu = multiplyCPU(A_cpu, B_cpu, n);
        
        for (int c = 0; c < numConfigs; c++) {
            cout << "\n--- Config: " << configs[c].name << " ---" << endl;
            
            dim3 threadDim(configs[c].threadX, configs[c].threadY);
            dim3 blockDim;
            blockDim.x = (n + threadDim.x - 1) / threadDim.x;
            blockDim.y = (n + threadDim.y - 1) / threadDim.y;
            
            cout << "Grid: (" << blockDim.x << ", " << blockDim.y << "), "
                 << "Block: (" << threadDim.x << ", " << threadDim.y << ")" << endl;
            
            // Один запуск для точного времени (как в примере)
            auto start = high_resolution_clock::now();
            matrixMultiplyKernel<<<blockDim, threadDim>>>(d_A, d_B, d_C, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end = high_resolution_clock::now();
            
            // Точное время в секундах с высокой точностью
            double gpu_time = duration_cast<nanoseconds>(end - start).count() / 1e9;
            
            cout << "GPU Time: " << fixed << setprecision(9) << gpu_time << " sec" << endl;
            
            resultsFile << n << "," << configs[c].name << "," << gpu_time << endl;
            
            // Верификация для первой конфигурации
            if (s == 0 && c == 0) {
                vector<double> C_gpu(n * n);
                CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, size, cudaMemcpyDeviceToHost));
                bool valid = verifyResult(C_cpu, C_gpu, n);
                cout << "Verification: " << (valid ? "PASSED" : "FAILED") << endl;
            }
        }
        
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        
        cout << "\nSaved in folder: " << folder << endl;
    }
    
    resultsFile.close();
    
    cout << "\n========================================" << endl;
    cout << "Results saved to: cuda_results.csv" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
'''

with open('matrix_multiply_cuda.cu', 'w') as f:
    f.write(cuda_code)

print("✅ CUDA program updated with correct format")
print("Changes:")
print("1. Sizes: 200, 400, 800, 1200, 1600, 2000")
print("2. Block configs: 8x8, 16x16, 32x8, 64x4, 128x2, 8x32, 4x64, 2x128")
print("3. CSV format: Size,BlockConfig,Time_seconds")
print("4. Time precision: nanoseconds for accuracy")