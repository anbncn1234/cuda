

#define CHECK(call) \
 {\
    const cudaError_t error = call; \
    if (error != cudaSuccess)\
    {\
        printf("Error: %s: %d\n", __FILE__, __LINE__);\
        printf("code :%d reason :%s\n", error , cudaGetErrorString(error));\
        exit(1);\
    }\
}

// 检查 CUDA 错误
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// 检查 cuSPARSE 错误
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "cuSPARSE Error: " << status << std::endl;                \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}


class Timer{
	cudaEvent_t _start;
	cudaEvent_t _stop;

	public:
	Timer(){
		cudaEventCreate(&_start);
		cudaEventCreate(&_stop);
	}

	void start(){
		cudaEventRecord(_start, 0);
	}




	void stop(){
		cudaEventRecord(_stop, 0);
		cudaEventSynchronize(_stop);
	}

	float elapsedms(){
		float out;
		cudaEventElapsedTime(&out, _start, _stop);
		return out;
	}

	~Timer(){
		cudaEventDestroy(_start);
		cudaEventDestroy(_stop);
	}
};


