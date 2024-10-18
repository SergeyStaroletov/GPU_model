/*
@author sergey staroletov
OpenCL runner and ptx saver
Works on Linux 340.108 driver
*/

#include <CL/cl.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>


void printfinfo() {
    char *value;
    size_t valueSize;
    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);
    cl_platform_id *platforms =
        (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
    for (int p = 0; p < platformCount; p++) {
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        cl_device_id *devices =
            (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, deviceCount, devices,
                       NULL);
        for (int d = 0; d < deviceCount; d++) {
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. device: %s\n", d + 1, value);
            free(value);
            clGetDeviceInfo(devices[d], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[d], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.1 hardware standart: %s\n", d + 1, value);
            free(value);
            clGetDeviceInfo(devices[d], CL_DEVICE_OPENCL_C_VERSION, 0, NULL,
                            &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[d], CL_DEVICE_OPENCL_C_VERSION, valueSize, value,
                            NULL);
            printf(" %d.2 OpenCL C ver: %s\n", d + 1, value);
            free(value);
            cl_uint maxComputeUnits;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.3 compute units: %d\n", d + 1, maxComputeUnits);
        }
        free(devices);
    }
    free(platforms);
}



#define MAXSIZE 100000

void compute() {
    
    //big data to compute: array of MAXSIZE
    int *array = (int *)malloc(MAXSIZE * sizeof(int));
    for (int i = 0; i < MAXSIZE; i++) {
        array[i] = 2;
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue queue;
    FILE *programHandle;
    char *programBuffer;
    size_t programSize;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_uint units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units,
                    NULL);
    int sz = 20;//max units
    int *results = (int *)malloc(sz * sizeof(int));
    memset(results, sz * sizeof(int), 0);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    
    programHandle = fopen("kern.cl", "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);
    programBuffer = (char *)malloc(programSize + 1);
    memset(programBuffer, 0, programSize + 1);
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);
    program = clCreateProgramWithSource(context, 1, (const char **)&programBuffer,
                                        &programSize, NULL);
    free(programBuffer);
    int err = clBuildProgram(program, 1, &device, "-cl-std=CL1.0", NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("clBuildProgram: %d\n", err);
        char log[0x10000];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, log,
                              NULL);
        printf("\n%s\n", log);
        return;
    }


    //------- save ptx code
    cl_uint program_num_devices;
    clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &program_num_devices, NULL);
    size_t binaries_sizes[program_num_devices];
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, program_num_devices * sizeof(size_t), binaries_sizes, NULL);
    char **binaries = new char*[program_num_devices];
    for (size_t i = 0; i < program_num_devices; i++)
        binaries[i] = new char[binaries_sizes[i] + 1];
    clGetProgramInfo(program, CL_PROGRAM_BINARIES, program_num_devices * sizeof(size_t), binaries, NULL);
    for (size_t i = 0; i < program_num_devices; i++)
    {
        binaries[i][binaries_sizes[i]] = '\0';
        printf("Program %d:\n", i);
        //std::ofstream out_binary_file;
        char *fname = new char[20];
        strcpy(fname, "kernelX.ptx");
        fname[6] = '0' + i;
        FILE *f = fopen(fname, "w");
	fwrite(binaries[0], binaries_sizes[0], 1, f);
	fclose(f);
        delete []fname;
    }
    //------

    err = 0;
    kernel = clCreateKernel(program, "sum_even", &err);

    queue = clCreateCommandQueue(context, device, 0, NULL);
    cl_mem clArray = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    MAXSIZE * sizeof(int), NULL, NULL);
    clEnqueueWriteBuffer(queue, clArray, CL_TRUE, 0, MAXSIZE * sizeof(int), array,
                         0, NULL, NULL);
    cl_mem clResults = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sz * sizeof(int), NULL, NULL);

    printf("go!\n");


    int SM = 2;  //count of units
    int WG = 10; //WG size
    int TS = MAXSIZE / (WG * SM); //TS size - we split all the data 

    size_t global_item_size = SM * WG;
    size_t local_item_size = WG;


    clSetKernelArg(kernel, 0, sizeof(cl_mem), &clArray) ;
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &clResults);
    clSetKernelArg(kernel, 2, sizeof(cl_uint) * local_item_size, NULL);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &TS);


    struct timeval initial_time, final_time;
    gettimeofday(&initial_time, NULL);


    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size,
                           &local_item_size, 0, NULL, NULL);



    clEnqueueReadBuffer(queue, clResults, CL_TRUE, 0, units * sizeof(int), results, 0,
                        NULL, NULL);

    gettimeofday(&final_time, NULL);


    long long exec_time = ((long long)final_time.tv_sec * 1000000 + final_time.tv_usec) -
                          ((long long)initial_time.tv_sec * 1000000 + initial_time.tv_usec);
    printf("\nExecution time was %llu microseconds\n", exec_time);
    float bandwidth = 1e-9 * (float)(MAXSIZE*sizeof(cl_uint)) /
                      ((float)exec_time / 1e6);
    printf("Memory bandwidth %.2f GB/sec\n", bandwidth);


    clEnqueueReadBuffer(queue, clArray, CL_TRUE, 0, MAXSIZE * sizeof(int), array,
                         0, NULL, NULL);

    clFlush(queue);

    //final reduction 
    
    int sum_main = 0;
    for (int i = 0; i < SM; i++)
	sum_main += results[i];
    
    printf("result = %d!\n", sum_main);


    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(clArray);
    clReleaseMemObject(clResults);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, const char *argv[]) {

    printfinfo();
    compute();

    return 0;
}
