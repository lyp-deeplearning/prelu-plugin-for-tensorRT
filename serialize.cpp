#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include "sys/time.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/imgproc/types_c.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvOnnxParserRuntime.h"
#include "NvOnnxConfig.h"
#include <time.h>
#include "Gplugin.h"
#include "GpluginGPU.h"
using namespace nvinfer1;

static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;
static int gUseDLACore{-1};

struct LayerInfo
{
	std::vector<int> dim;
	std::string name;
	int index;
	int size;
};
nvinfer1::IExecutionContext* context;
nvinfer1::IRuntime* runtime;
nvinfer1::ICudaEngine* engine;
cudaStream_t stream;
std::vector<LayerInfo> output_layer;
int input_size;
//std::vector<int> m_output_size;
void* buffers[10];
int inputIndex;


float m_nms_threshold = 0.4;
float data0[8] = { -248,-248,263,263,-120,-120,135,135 };
float data1[8] = { -56,-56,71,71,-24,-24,39,39 };
float data2[8] = { -8,-8,23,23,0,0,15,15 };


std::vector<int> get_dim_size(Dims dim)
{
	std::vector<int> size;
	for (int i = 0; i < dim.nbDims; ++i)
		size.emplace_back(dim.d[i]);
	return size;
}

int total_size(std::vector<int> dim)
{
	int size = 1 * sizeof(float);
	for (auto d : dim)
		size *= d;
	return size;
}



int caffeToGIEModel(const std::string& deployFile,const std::string& modelFile,Logger logger,nvcaffeparser1::IPluginFactory* pluginFactory) 
{
	// create the builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
 
	// parse the caffe model to populate the network, then set the outputs
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
        parser->setPluginFactory(pluginFactory);
       // const std::string deployFile={"/home/lyp/project/tf_insightface/deploy_insightface.prototxt"};
       // const std::string modelFile={"/home/lyp/project/tf_insightface/inference_insightface.caffemodel"};
        std::cout<<"model"<<std::endl;
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse("/home/lyp/project/tf_insightface/aa.prototxt", "/home/lyp/project/tf_insightface/inference_insightface.caffemodel", *network, nvinfer1::DataType::kFLOAT);
        const char* outputs={"fc5"};
	// specify which tensors are outputs
	
	network->markOutput(*blobNameToTensor->find("fc5"));
        std::cout<<"model-success"<<std::endl;
	// Build the engine
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 20);
 
	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	
 
	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();
 
        nvinfer1::IHostMemory* gieModelStream = engine->serialize(); // GIE model
	fprintf(stdout, "allocate memory size: %d bytes\n", gieModelStream->size());

        const std::string engine_file="/home/lyp/project/tf_insightface/insightface_v.engine";
	std::ofstream outfile(engine_file.c_str(), std::ios::out | std::ios::binary);
	if (!outfile.is_open()) {
		fprintf(stderr, "fail to open file to write: %s\n", engine_file.c_str());
        return -1;
    }
	unsigned char* p = (unsigned char*)gieModelStream->data();
	outfile.write((char*)p, gieModelStream->size());
	outfile.close();
 
	engine->destroy();
	builder->destroy();
	if (gieModelStream) gieModelStream->destroy();	
	nvcaffeparser1::shutdownProtobufLibrary();
	
	return 0;
}




void doInference()
{
	

}


int main(int argc, char** argv)
{
        std::cout<<1<<std::endl;
	const std::string deploy_file { "models/mnist.prototxt" };
	const std::string model_file { "models/mnist.caffemodel" };
	const std::string mean_file { "models/mnist_mean.binaryproto" };
	const std::string engine_file { "tensorrt_mnist.model" };
        Logger logger;
    // create a TensorRT model from the onnx model and serialize it to a stream
        IHostMemory* trtModelStream{nullptr};
        std::cout<<2<<std::endl;
        PluginFactory parserPluginFactory;
        caffeToGIEModel(deploy_file, model_file, logger,&parserPluginFactory);
        parserPluginFactory.destroyPlugin();
	std::cout<<3<<std::endl;
}
