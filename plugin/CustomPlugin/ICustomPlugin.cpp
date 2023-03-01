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
#include "CustomPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::CustomPluginCreator;
using nvinfer1::plugin::Custom;

static const char* Custom_PLUGIN_VERSION{"1"};
static const char* Custom_PLUGIN_NAME{"Custom_TRT"};
PluginFieldCollection CustomPluginCreator::mFC{};
std::vector<PluginField> CustomPluginCreator::mPluginAttributes;

// LeakyReLU {{{
Custom::Custom(float negSlope)
    : mNegSlope(negSlope)
    , mBatchDim(1)
{
}

Custom::Custom(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int Custom::getNbOutputs() const noexcept
{
    return 1;
}

Dims Custom::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

int Custom::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = CustomInference(stream, mBatchDim * batchSize, mNegSlope, inputData, outputData);
    return status;
}

size_t Custom::getSerializationSize() const noexcept
{
    // mNegSlope, mBatchDim
    return sizeof(float) + sizeof(int);
}

void Custom::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void Custom::configureWithFormat(
    const Dims* inputDims, int /* nbInputs */, const Dims* /* outputDims */, int nbOutputs, DataType type, PluginFormat format, int) noexcept
{
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    ASSERT(mBatchDim == 1);
    ASSERT(nbOutputs == 1);
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool Custom::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int Custom::initialize() noexcept
{
    return 0;
}

void Custom::terminate() noexcept {}

size_t Custom::getWorkspaceSize(int /* maxBatchSize */) const noexcept
{
    return 0;
}

const char* Custom::getPluginType() const noexcept // 对应
{
    return Custom_PLUGIN_NAME;
}

const char* Custom::getPluginVersion() const noexcept // 对应
{
    return Custom_PLUGIN_VERSION;
}

void Custom::destroy() noexcept
{
    delete this;
}

IPluginV2* Custom::clone() const noexcept
{
    IPluginV2* plugin = new Custom(mNegSlope);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

CustomPluginCreator::CustomPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CustomPluginCreator::getPluginName() const noexcept
{
    return Custom_PLUGIN_NAME;
}

const char* CustomPluginCreator::getPluginVersion() const noexcept
{
    return Custom_PLUGIN_VERSION;
}

const PluginFieldCollection* CustomPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* CustomPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    float negSlope = *(static_cast<const float*>(fields[0].data));

    return new Custom(negSlope);
}

IPluginV2* CustomPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call CustomPlugin::destroy()
    return new Custom(serialData, serialLength);
}
// LeakReLU }}}
