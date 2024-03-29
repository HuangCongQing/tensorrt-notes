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

#include "NvInfer.h"
#include "NvUffParser.h"
#include <chrono>
#include <cudnn.h>
#include <iostream>
#include <map>
#include <string.h>
#include <unordered_map>
#include <vector>

#include "NvUtils.h"
#include "argsParser.h"
#include "common.h"
#include "half.h"
#include "logger.h"

using namespace nvuffparser;
using namespace nvinfer1;
using namespace samplesCommon;

const std::string gSampleName = "TensorRT.sample_uff_plugin_v2_ext";
samplesCommon::Args gArgs;

template <DataType in, DataType out>
void transform(const void* src, void* dst, int count)
{
    ASSERT(in == out);
    memcpy(dst, src, count * elementSize(in));
}

template <>
void transform<DataType::kHALF, DataType::kFLOAT>(const void* src, void* dst, int count)
{
    const auto* srcPtr = static_cast<const half_float::half*>(src);
    auto* dstPtr = static_cast<float*>(dst);
    std::transform(srcPtr, srcPtr + count, dstPtr, [](half_float::half in) { return static_cast<float>(in); });
}

template <>
void transform<DataType::kINT8, DataType::kFLOAT>(const void* src, void* dst, int count)
{
    const auto* srcPtr = static_cast<const int8_t*>(src);
    auto* dstPtr = static_cast<float*>(dst);
    std::transform(srcPtr, srcPtr + count, dstPtr, [](int8_t in) { return static_cast<float>(in); });
}

template <>
void transform<DataType::kFLOAT, DataType::kHALF>(const void* src, void* dst, int count)
{
    const auto* srcPtr = static_cast<const float*>(src);
    auto* dstPtr = static_cast<half_float::half*>(dst);
    std::transform(srcPtr, srcPtr + count, dstPtr, [](float in) { return static_cast<half_float::half>(in); });
}

template <>
void transform<DataType::kFLOAT, DataType::kINT8>(const void* src, void* dst, int count)
{
    const auto* srcPtr = static_cast<const float*>(src);
    auto* dstPtr = static_cast<int8_t*>(dst);
    std::transform(srcPtr, srcPtr + count, dstPtr, [](float x) {
        x = std::max(x, float(INT8_MIN));
        x = std::min(x, float(INT8_MAX));
        return static_cast<int8_t>(x);
    });
}

static const int INPUT_H = 28;
static const int INPUT_W = 28;

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(locateFile(filename, gArgs.dataDirs), buffer, INPUT_H, INPUT_W);
}

std::vector<std::pair<size_t, DataType>> calculateBindingBufferSizes(
    const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<size_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        size_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }
    return sizes;
}

void* createMnistCudaBuffer(int64_t eltCount, DataType dtype, int num)
{
    // in that specific case, eltCount == INPUT_H * INPUT_W
    ASSERT(eltCount == INPUT_H * INPUT_W);
    ASSERT(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    std::vector<float> inputs(eltCount);

    // read PGM file
    uint8_t fileData[INPUT_H * INPUT_W];
    readPGMFile(std::to_string(num) + ".pgm", fileData);

    // display the number in an ascii representation
    sample::gLogInfo << "Input:\n";
    for (int i = 0; i < eltCount; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    // initialize the inputs buffer
    for (int i = 0; i < eltCount; i++)
    {
        inputs[i] = 1.0 - float(fileData[i]) / 255.0;
    }

    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs.data(), memSize, cudaMemcpyHostToDevice));

    return deviceMem;
}

bool verifyOutput(int64_t eltCount, DataType dtype, void* buffer, int num)
{
    ASSERT(elementSize(dtype) == sizeof(float));

    bool pass = false;

    size_t memSize = eltCount * elementSize(dtype);
    std::vector<float> outputs(eltCount);
    CHECK(cudaMemcpy(outputs.data(), buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));

    std::ios::fmtflags prevSettings = sample::gLogInfo.flags();
    sample::gLogInfo.setf(std::ios::fixed, std::ios::floatfield);
    sample::gLogInfo.precision(6);
    sample::gLogInfo << "Output:\n";
    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        sample::gLogInfo << eltIdx << " => " << std::setw(10) << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
        {
            sample::gLogInfo << "***";
            pass = eltIdx == num;
        }
        sample::gLogInfo << "\n";
    }
    sample::gLogInfo.flags(prevSettings);
    sample::gLogInfo << std::endl;
    return pass;
}

struct PoolParameters
{
    // Input dimensions
    int mC, mH, mW;
    // Output dimensions
    int mP, mQ;
    // Kernel size
    int mR, mS;
    // Stride
    int mU, mV;
    // Padding
    int pH, pW;
    // Pooling Function
    PoolingType pType;
};

class SampleUffPluginV2Ext
{
public:
    explicit SampleUffPluginV2Ext(const UffSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Creates the network, configures the builder and creates the network engine
    //!
    bool build()
    {
        SampleUniquePtr<IUffParser> parser{createUffParser()};
        parser->registerInput("in", Dims3(1, 28, 28), UffInputOrder::kNCHW);
        parser->registerOutput("out");

        SampleUniquePtr<IBuilder> builder{createInferBuilder(sample::gLogger.getTRTLogger())};
        if (!builder.get())
        {
            sample::gLogError << "Failed to create infer builder. " << std::endl;
            return false;
        }

        SampleUniquePtr<INetworkDefinition> network{builder->createNetworkV2(0)};
        if (!network.get())
        {
            sample::gLogError << "Failed to create network. " << std::endl;
            return false;
        }

        if (!parser->parse(mParams.uffFileName.data(), *network, nvinfer1::DataType::kFLOAT))
        {
            sample::gLogError << "Failure while parsing UFF file" << std::endl;
            return false;
        }

        if (gArgs.runInInt8)
        {
            samplesCommon::setAllDynamicRanges(network.get(), 25.0F, 25.0F);
        }

        SampleUniquePtr<IBuilderConfig> networkConfig{builder->createBuilderConfig()};
        networkConfig->setMaxWorkspaceSize(1_GiB);
        if (gArgs.runInFp16)
        {
            networkConfig->setFlag(BuilderFlag::kFP16);
        }
        if (gArgs.runInInt8)
        {
            networkConfig->setFlag(BuilderFlag::kINT8);
        }
        if (gArgs.useDLACore >= 0)
        {
            networkConfig->setDLACore(gArgs.useDLACore);
        }

        const int maxBatchSize = 1;
        builder->setMaxBatchSize(maxBatchSize);
        samplesCommon::enableDLA(builder.get(), networkConfig.get(), gArgs.useDLACore);

        // CUDA stream used for profiling by the builder.
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream)
        {
            return false;
        }
        networkConfig->setProfileStream(*profileStream);

        SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *networkConfig)};
        if (!plan)
        {
            sample::gLogError << "Unable to create serialized engine. " << std::endl;
            return false;
        }

        SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
        if (!runtime)
        {
            sample::gLogError << "Unable to create runtime. " << std::endl;
            return false;
        }

        mEngine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!mEngine.get())
        {
            sample::gLogError << "Unable to create engine. " << std::endl;
            return false;
        }
        return true;
    }

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer()
    {
        bool pass{true};
        SampleUniquePtr<IExecutionContext> context{mEngine->createExecutionContext()};

        const int batchSize{1};
        const int nbBindings = mEngine->getNbBindings();
        ASSERT(nbBindings == 2);

        std::vector<void*> buffers(nbBindings);
        auto buffersSizes = calculateBindingBufferSizes(*mEngine, nbBindings, batchSize);

        const int bindingIdxInput = mEngine->bindingIsInput(0) ? 0 : 1;
        const int bindingIdxOutput = mEngine->bindingIsInput(0) ? 1 : 0;
        auto bufferSizesOutput = buffersSizes[bindingIdxOutput];
        buffers[bindingIdxOutput] = safeCudaMalloc(bufferSizesOutput.first * elementSize(bufferSizesOutput.second));

        auto bufferSizesInput = buffersSizes[bindingIdxInput];

        const int iterations{1};
        const int numberRun{10};
        for (int i = 0; i < iterations; i++)
        {
            float total{0.0F}, ms{0.0F};
            for (int num = 0; num < numberRun; num++)
            {
                buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first, bufferSizesInput.second, num);
                auto t_start = std::chrono::high_resolution_clock::now();
                ASSERT(context->execute(batchSize, &buffers[0]));
                auto t_end = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
                total += ms;

                for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
                {
                    if (mEngine->bindingIsInput(bindingIdx))
                    {
                        continue;
                    }
                    auto bufferSizesOutput = buffersSizes[bindingIdx];
                    pass &= verifyOutput(bufferSizesOutput.first, bufferSizesOutput.second, buffers[bindingIdx], num);
                }
                CHECK(cudaFree(buffers[bindingIdxInput]));
            }
            total /= numberRun;
            sample::gLogInfo << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
        }

        for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        {
            if (!mEngine->bindingIsInput(bindingIdx))
            {
                CHECK(cudaFree(buffers[bindingIdx]));
            }
        }
        return pass;
    }

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown()
    {
        nvuffparser::shutdownProtobufLibrary();
        return true;
    }

private:
    SampleUniquePtr<nvinfer1::ICudaEngine> mEngine;
    samplesCommon::UffSampleParams mParams;
};

// main入口==========================================
class UffPoolPluginV2 : public IPluginV2IOExt
{
public:
    // 系列化使用
    UffPoolPluginV2(const PluginFieldCollection& fc)
    {
        // To do: TRT-TRT-8010 Populate Parameters from fc object w/ hard code
        mPoolingParams.pType = PoolingType::kMAX;
        mPoolingParams.mU = 2;
        mPoolingParams.mV = 2;
        mPoolingParams.mR = 2;
        mPoolingParams.mS = 2;
        mPoolingParams.pH = 0;
        mPoolingParams.pW = 0;
        mMode = CUDNN_POOLING_MAX;
        (void) fc;
    }

    // 反序列化使用
    UffPoolPluginV2(const void* data, size_t length)
    {
        const char* d = static_cast<const char*>(data);
        const char* const a = d;
        mPoolingParams = read<PoolParameters>(d);
        mInputDims.nbDims = read<int>(d);
        for (int i = 0; i < mInputDims.nbDims; ++i)
        {
            mInputDims.d[i] = read<int>(d);
        }
        mOutputDims.nbDims = read<int>(d);
        for (int i = 0; i < mOutputDims.nbDims; ++i)
        {
            mOutputDims.d[i] = read<int>(d);
        }
        mDataType = static_cast<DataType>(read<int>(d));
        mMode = mPoolingParams.pType == PoolingType::kMAX ? CUDNN_POOLING_MAX
                                                          : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        if (mDataType == DataType::kINT8)
        {
            mInHostScale = read<float>(d);
            mOutHostScale = read<float>(d);
        }
        ASSERT(d == a + length);
    }

    // It makes no sense to construct UffPoolPluginV2 without arguments.
    UffPoolPluginV2() = delete;

    virtual ~UffPoolPluginV2() {}

public:
    // 获得输出的个数
    // 插件op返回多少个Tensor，比如MyCustomPlugin这个操作只输出一个Tensor(也就是一个output)，所以直接return 1：
    int getNbOutputs() const noexcept override
    {
        return 1;
    }


    // 获得输出的dim
    // TensorRT支持Dynamic-shape的时候，batch这一维度必须是explicit的，也就是说，TensorRT处理的维度从以往的三维[3,-1,-1]变成了[1,3,-1,-1]。
    // 最新的onnx-tensorrt也必须设置explicit的batchsize，而且这个batch维度在getOutputDimensions中是可以获取到的。
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override
    {
        ASSERT(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        int height = (inputs[0].d[1] + mPoolingParams.pH * 2 - mPoolingParams.mR) / mPoolingParams.mU + 1;
        int width = (inputs[0].d[2] + mPoolingParams.pW * 2 - mPoolingParams.mS) / mPoolingParams.mV + 1;
        DimsHW outDims(height, width);
        return Dims3(inputs[0].d[0], outDims.h(), outDims.w());
    }

    // initialize和terminate这两个比较重要！！！！！！！！！！！！！！
    // 初始化
    /* 
    初始化函数，在这个插件准备开始run之前执行。

主要初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数(例如conv操作需要执行卷积操作，我们就需要提前开辟weight和bias的显存)，假如我们的算子需要这些参数，则在这里需要提前开辟显存。

需要注意的是，如果插件算子需要开辟比较大的显存空间，不建议自己去申请显存空间，可以使用Tensorrt官方接口传过来的workspace指针来获取显存空间。因为如果这个插件被一个网络调用了很多次，而这个插件op需要开辟很多显存空间，那么TensorRT在构建network的时候会根据这个插件被调用的次数开辟很多显存，很容易导致显存溢出。
    
     */
    int initialize() noexcept override
    {
        CHECK(cudnnCreate(&mCudnn));
        CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));
        CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
        CHECK(cudnnCreatePoolingDescriptor(&mPoolingDesc));
        CHECK(cudnnSetPooling2dDescriptor(mPoolingDesc, mMode, CUDNN_NOT_PROPAGATE_NAN, mPoolingParams.mR,
            mPoolingParams.mS, mPoolingParams.pH, mPoolingParams.pW, mPoolingParams.mU, mPoolingParams.mV));
        return 0;
    }

    // 比较重要!!!!
    void terminate() noexcept override
    {
        CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
        CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
        CHECK(cudnnDestroyPoolingDescriptor(mPoolingDesc));
        CHECK(cudnnDestroy(mCudnn));
    }

    // 显存大小不可控????????????????///
    // 这个函数需要返回这个插件op需要中间显存变量的实际数据大小(bytesize)，这个是通过TensorRT的接口去获取，是比较规范的方式。
    // 我们需要在这里确定这个op需要多大的显存空间去运行，在实际运行的时候就可以直接使用TensorRT开辟好的空间而不是自己去申请显存空间。
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override
    {
        return 0;
    }

    //===================================================================================
    // 实际插件op的执行函数，我们自己实现的cuda操作就放到这里(当然C++写的op也可以放进来，不过因为是CPU执行，速度就比较慢了)，
    // 与往常一样接受输入inputs产生输出outputs，传给相应的指针就可以。

    // 需要注意的是，如果我们的操作需要一些分布在显存中的中间变量，可以通过传过来的指针参数workspace获取，上述代码简单说明了一下使用方法。

    // 再多说一句，我们默认写的.cu是fp32的，TensorRT在fp16运行模式下，运行到不支持fp16的插件op时，会自动切换到fp32模式，等插件op运行完再切换回来。
    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        const float kONE = 1.0F, kZERO = 0.0F;
        cudnnSetStream(mCudnn, stream);

        const int N = 1;
        // Use float to simulate int8 calculation
        std::map<DataType, cudnnDataType_t> typeMap = {{DataType::kFLOAT, CUDNN_DATA_FLOAT},
            {DataType::kHALF, CUDNN_DATA_HALF}, {DataType::kINT8, CUDNN_DATA_FLOAT}};
        ASSERT(mDataType != DataType::kINT32);
        CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, typeMap[mDataType], N, mPoolingParams.mC,
            mPoolingParams.mH, mPoolingParams.mW));
        CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, typeMap[mDataType], N, mPoolingParams.mC,
            mPoolingParams.mP, mPoolingParams.mQ));
        void* input{nullptr};
        void* output{nullptr};
        if (mDataType == DataType::kINT8)
        {
            copyDeviceInputToFP32(inputs[0], input);
            size_t outCount = getC(mOutputDims) * getH(mOutputDims) * getW(mOutputDims);
            CHECK(cudaMalloc(&output, outCount * elementSize(DataType::kFLOAT)));
        }
        else
        {
            input = const_cast<void*>(inputs[0]);
            output = const_cast<void*>(outputs[0]);
        }
        CHECK(cudnnPoolingForward(mCudnn, mPoolingDesc, &kONE, mSrcDescriptor, input, &kZERO, mDstDescriptor, output));
        if (mDataType == DataType::kINT8)
        {
            copyDeviceToInt8Output(output, outputs[0]);
        }
        return 0;
    }

    // 得到序列化的大小
    // 返回序列化时需要写多少字节到buffer中。
    size_t getSerializationSize() const noexcept override
    {
        size_t serializationSize = 0;
        serializationSize += sizeof(mPoolingParams);
        serializationSize += sizeof(mInputDims.nbDims);
        serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
        serializationSize += sizeof(mOutputDims.nbDims);
        serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
        serializationSize += sizeof(static_cast<int>(mDataType));
        if (mDataType == DataType::kINT8)
        {
            serializationSize += sizeof(float) * 2;
        }
        return serializationSize;
    }

    // 序列化代码
    // 把需要用的数据按照顺序序列化到buffer里头。
    void serialize(void* buffer) const noexcept override
    {
        char* d = static_cast<char*>(buffer);
        const char* const a = d;
        write(d, mPoolingParams);
        write(d, mInputDims.nbDims);
        ASSERT(mInputDims.nbDims <= mInputDims.MAX_DIMS);
        for (int i = 0; i < mInputDims.nbDims; ++i)
        {
            write(d, mInputDims.d[i]);
        }
        write(d, mOutputDims.nbDims);
        ASSERT(mOutputDims.nbDims <= mOutputDims.MAX_DIMS);
        for (int i = 0; i < mOutputDims.nbDims; ++i)
        {
            write(d, mOutputDims.d[i]);
        }
        write(d, static_cast<int>(mDataType));
        if (mDataType == DataType::kINT8)
        {
            write(d, mInHostScale);
            write(d, mOutHostScale);
        }
        ASSERT(d == a + getSerializationSize());
    }

    // 判断输入是否符合标准 ,判断异常情况
    // 配置这个插件op，判断输入和输出类型数量是否正确。官方还提到通过这个配置信息可以告知TensorRT去选择合适的算法(algorithm)去调优这个模型。

    // 但自动调优目前还没有尝试过，我们一般自己写的plugin执行代码都是定死的，所谓的调优步骤可能更多地针对官方的op。

    // 下面的plugin中configurePlugin函数仅仅是简单地确认了下输入和输出以及类型。
    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept override
    {
        ASSERT(in && nbInput == 1);
        ASSERT(out && nbOutput == 1);
        ASSERT(in[0].type == out[0].type);
        ASSERT(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);

        mDataType = in[0].type;
        mInputDims = in[0].dims;
        mOutputDims = out[0].dims;
        mPoolingParams.mC = mInputDims.d[0];
        mPoolingParams.mH = mInputDims.d[1];
        mPoolingParams.mW = mInputDims.d[2];
        mPoolingParams.mP = mOutputDims.d[1];
        mPoolingParams.mQ = mOutputDims.d[2];
        mInHostScale = in[0].scale >= 0.0F ? in[0].scale : -1.0F;
        mOutHostScale = out[0].scale >= 0.0F ? out[0].scale : -1.0F;
    }

    //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override
    {
        ASSERT(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        condition &= inOut[pos].type != DataType::kINT32;
        condition &= inOut[pos].type == inOut[0].type;
        return condition;
    }

    // 返回结果的类型，一般来说我们插件op返回结果类型与输入类型一致：
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        ASSERT(inputTypes && nbInputs == 1);
        (void) index;
        return inputTypes[0];
    }

    const char* getPluginType() const noexcept override
    {
        return "MaxPool";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "2";
    }

    void destroy() noexcept override
    {
        delete this;
    }

    // 这玩意儿干嘛的，顾名思义，就是克隆嘛，将这个plugin对象克隆一份给TensorRT的builder、network或者engine。
    // 这个成员函数会调用上述说到的第二个构造函数：
    IPluginV2Ext* clone() const noexcept override
    {
        auto* plugin = new UffPoolPluginV2(*this);
        return plugin;
    }

    // set/getPluginNamespace
    // 为这个插件设置namespace名字，如果不设置则默认是""，需要注意的是同一个namespace下的plugin如果名字相同会冲突。
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.data();
    }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override
    {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override
    {
        return false;
    }

private:
    template <typename T>
    void write(char*& buffer, const T& val) const noexcept
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer) const noexcept
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    void copyDeviceInputToFP32(const void* src, void*& dst) noexcept
    {
        ASSERT(mDataType == DataType::kINT8);
        size_t inCount = getC(mInputDims) * getH(mInputDims) * getW(mInputDims);
        std::vector<char> inputTmp(inCount * elementSize(mDataType));
        CHECK(cudaMemcpy(inputTmp.data(), src, inCount * elementSize(mDataType), cudaMemcpyDeviceToHost));
        std::vector<float> inputFP32(inCount);
        transform<DataType::kINT8, DataType::kFLOAT>(inputTmp.data(), inputFP32.data(), inCount);
        // int8 scale
        int hw = mInputDims.d[1] * mInputDims.d[2];
        for (int j = 0; j < mInputDims.d[0]; ++j)
        {
            std::transform(inputFP32.data() + hw * j, inputFP32.data() + hw * (j + 1), inputFP32.data() + hw * j,
                [&](float in) -> float { return in * mInHostScale; });
        }
        CHECK(cudaMalloc(&dst, inCount * elementSize(DataType::kFLOAT)));
        CHECK(cudaMemcpy(dst, inputFP32.data(), inCount * elementSize(DataType::kFLOAT), cudaMemcpyHostToDevice));
    }

    void copyDeviceToInt8Output(const void* src, void* dst) noexcept
    {
        size_t outCount = getC(mOutputDims) * getH(mOutputDims) * getW(mOutputDims);
        std::vector<float> outTmp(outCount);
        CHECK(cudaMemcpy(outTmp.data(), src, outCount * elementSize(DataType::kFLOAT), cudaMemcpyDeviceToHost));
        std::vector<char> outInt8(outCount * elementSize(DataType::kINT8));
        // int8 + scale
        int hw = mOutputDims.d[1] * mOutputDims.d[2];
        for (int j = 0; j < mInputDims.d[0]; ++j)
        {
            std::transform(outTmp.data() + hw * j, outTmp.data() + hw * (j + 1), outTmp.data() + hw * j,
                [&](float in) -> float { return in / mOutHostScale; });
        }
        transform<DataType::kFLOAT, DataType::kINT8>(outTmp.data(), outInt8.data(), outCount);
        CHECK(cudaMemcpy(dst, outInt8.data(), outCount, cudaMemcpyHostToDevice));
    }

private:
    cudnnHandle_t mCudnn;
    cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
    cudnnPoolingDescriptor_t mPoolingDesc;
    PoolParameters mPoolingParams;
    cudnnPoolingMode_t mMode;
    DataType mDataType;

    Dims mInputDims;
    Dims mOutputDims;
    float mInHostScale{-1.0F};
    float mOutHostScale{-1.0F};
    std::string mNamespace;
};

// 构造函数
// 反序列化: 从文件中的数据得到plugin
class UffPoolPluginV2Creator : public IPluginCreator
{
public:
    const char* getPluginName() const noexcept override
    {
        return "MaxPool";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "2";
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &mFieldCollection;
    }

    // 核心==================================================================================
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        auto* plugin = new UffPoolPluginV2(*fc);
        mFieldCollection = *fc;
        mPluginName = name;
        return plugin;
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        auto* plugin = new UffPoolPluginV2(serialData, serialLength);
        mPluginName = name;
        return plugin;
    }

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    std::string mPluginName;
    PluginFieldCollection mFieldCollection{0, nullptr};
};

REGISTER_TENSORRT_PLUGIN(UffPoolPluginV2Creator);

// This function prints the help information for running this sample
void printHelpInfo()
{
    std::cout << "Usage: ./sample_uff_plugin_v2_ext [-h or --help] [-d or --datadir=<path to data directory>] "
                 "[--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode.\n";
}

int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "data/mnist/"};
    }
    auto sampleTest = sample::Logger::defineTest(gSampleName, argc, argv);

    sample::Logger::reportTestStart(sampleTest);

    samplesCommon::UffSampleParams params;
    params.uffFileName = locateFile("lenet5_custom_pool.uff", gArgs.dataDirs);
    sample::gLogInfo << params.uffFileName << std::endl;
    SampleUffPluginV2Ext sample(params);

    if (!sample.build())
    {
        return sample::Logger::reportFail(sampleTest);
    }

    if (!sample.infer())
    {
        return sample::Logger::reportFail(sampleTest);
    }

    if (!sample.teardown())
    {
        return sample::Logger::reportFail(sampleTest);
    }

    return sample::Logger::reportPass(sampleTest);
}
