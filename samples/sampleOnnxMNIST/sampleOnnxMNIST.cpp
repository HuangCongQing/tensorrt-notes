/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!
/* 
非常棒的解读！：https://blog.csdn.net/yanggg1997/article/details/111587687#t17
Related Docs：https://www.yuque.com/huangzhongqing/gk5f7m/tfyfx6#XNtir
*/

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

// using关键字是c++11中为类取别名的新关键字
// std::unique_ptr是智能指针的关键字
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
// 巨庞大的SampleOnnxMNIST类，这个就是我们程序的核心类了，封装了大量重要的功能。
class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params) // 各种初始化参数（构造函数）
        , mEngine(nullptr)
    {
    }

    //! 上面里的: mParams(params), mEngine(nullptr)是指初始化列表，
    //! 列表中有两个类成员分别为mParams和mEngine，前者值为初始化类SampleOnnxMNIST时传参params，后者则初始化为空指针
    //! \brief 构建引擎Function builds the network engine
    //!
    bool build();

    //!
    //! \brief 使用生成的Tensor网络进行推断 Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    // 继承自SampleParams结构体的，只不过新增了一个onxxFileName成员变量
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    // mInputDims和mOutputDims指的是输入和输出Tensor的维度信息，它们的类型是nvinfer1::Dims类型，
    // Dims类型的定义如下，在./include/NvInferRuntimeCommom.h文件下
    nvinfer1::Dims mInputDims;  //!< input输入维数 The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< output输出维数 The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify GT主要用于验证~~~

    // 定义的是一个用来run网络的engine，是一个指向nvinfer1::IcudaEngine类型的智能指针，它是具体的网络结构以及参数设定的更上层的封装。
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< 转换后的TensorRT网络 The TensorRT engine used to run the network

    //!
    //! \brief 将onnx模型转化为TensorRT网络 Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    // 将onnx模型转化为TensorRT网络
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    //  实现输入的读取和处理 读取并缓存input到buffer
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //! 对推理结果的输出进行验证
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!

// step1: 构建
bool SampleOnnxMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 生成network， config， parser
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    //构建网络（ 将onnx模型转化为TensorRT网络）
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    // 得到Engine！！！！！！！！===========================
    // 对network进行build操作，根据在前面constructNetwork中设定了的config来生成TensorRT的网络。
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!

// 构建网络
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    // 解析得到parse
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(16_MiB);
    // 设置精度
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    // 设置是否支持DLA   DLA：一种深度网络特征融合方法  https://zhuanlan.zhihu.com/p/364196632
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!

// step2: 推理inference
// 进行TensorRT预测，先申请缓存，然后设定输入，最后执行engine
bool SampleOnnxMNIST::infer()
{
    // Create RAII buffer manager object
    // 根据engine和batchsize自动生成一块输入的数据和输出的数据====================================
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    // 同步推理=====================================================================
    // https://www.yuque.com/huangzhongqing/gk5f7m/ysgfhl#IIXGh
    // 同步接口：execute()/executeV2()
    // 异步接口：enqueue()/enqueueV2()
    bool status = context->executeV2(buffers.getDeviceBindings().data());  // getDeviceBindings：直接获得输入和输出的指针的值
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost(); // 把数据从主机端拷贝到设备端，在设备端执行运算，然后把结果再从设备端拷贝到主机端。

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2]; // 获取网络输入层中定义的图像的高和宽
    const int inputW = mInputDims.d[3];

    // Read a random digit file
    srand(unsigned(time(nullptr))); // 设定随机数，用来随机读取一张图像
    std::vector<uint8_t> fileData(inputH * inputW); // 创建一个vector存储读入的图像
    // 获得从0~9范围内的随机数，选择一张这样的图像作为输入，并传给SampleOnnxMNIST类的mNumber成员变量，作为gt存储着，后面会用来判断预测值和gt是否相同。
    mNumber = rand() % 10; 
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print an ascii representation
    // 使用ascii码在终端拼图片（实际应用不必）
    sample::gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    // 把数字填充到buffer中input的相应位置
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        // 原始图像是8位黑白图像，且是白底黑字的，将它转换到0~1且是黑底白字。
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0); // 最终将图像数据赋值给hostDataBuffer(buffers)<<<<<<<<<<<<<<<
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//! 验证结果是否正确
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1]; // 获得网络的输出层总共有多少个输出（即多少类）
    // // 获取存储在buffers中的输出结果
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax 把输出用softmax转换成置信度，并打印出来
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
                         << std::endl;
    }
    sample::gLogInfo << std::endl;

    return idx == mNumber && val > 0.9f; // 如果预测结果和实际相同，并且置信度大于0.9，则返回true
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{   
    //输入参数解析
    samplesCommon::Args args;	// 接收用户传递参数的变量
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);	// 将main函数的参数argc和argv解释成args，返回转换是否成功的bool值
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help) // 如果接收的参数是请求打印帮助信息，则打印帮助信息，退出程序。
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    //定义一个Logger用于记录和打印输出
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);	// 定义一个日志类

    // ==========================记录日志的开始==========================================
    sample::gLogger.reportTestStart(sampleTest);	// 记录日志的开始

    // 使用initializeSampleParams解析并传入参数，初始化SampleOnnxMNIST sample<<<<<<<<<<<<<<<<<<<<<<<<
    SampleOnnxMNIST sample(initializeSampleParams(args)); 	// 定义一个sample实例<<<<<<<<<<<<<<<<,

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    // step1 build
    if (!sample.build()) // 【主要】在build方法中构建网络，返回构建网络是否成功的状态
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    // step2 推理inference
    if (!sample.infer()) 	// 【主要】读取图像并进行推理，返回推理是否成功的状态
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    // 结束
    return sample::gLogger.reportPass(sampleTest);	// 报告结束
}
