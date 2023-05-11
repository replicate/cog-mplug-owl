# Cog template for mPlUG-Owl


[![Replicate](https://replicate.com/replicate/mplug-owl/badge)](https://replicate.com/replicate/mplug-owl) 

This repo demonstrates how to push an `mPLUG-Owl` model to replicate. 

mPLUG-Owl is a training paradigm designed to equip large language models (LLMs) with multi-modal abilities, utilizing a modular approach that integrates visual knowledge and abstracting capabilities. This enables diverse unimodal and multimodal abilities through the collaborative interplay of different modalities.

It was developed by Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chaoya Jiang, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang.

For more details, please refer to the original [paper](https://arxiv.org/abs/2304.14178) and Github [repository](https://github.com/X-PLUG/mPLUG-Owl).

## Prerequisites

- **GPU machine**. You'll need a Linux machine with an NVIDIA GPU attached and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. If you don't already have access to a machine with a GPU, check out our [guide to getting a 
GPU machine](https://replicate.com/docs/guides/get-a-gpu-machine).

- **Docker**. You'll be using the [Cog](https://github.com/replicate/cog) command-line tool to build and push a model. Cog uses Docker to create containers for models.

## Step 0: Install Cog

First, [install Cog](https://github.com/replicate/cog#install):

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

## Step 1: Set up weights

First, you need to download the model weights. From the root directory of this project, run: 

```
wget -P model/ http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/instruction_tuned.pth
wget -P model/ http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/tokenizer.model
```

Next, we recommend tensorizing the model, which will dramatically decrease the time it takes to load the model. It will also allow you to load the model directly to GPU; however, this requires a GPU large enough to store the model weights. If you don't not have a sufficiently large GPU on hand, you can remove the `to('cuda') call and then handle device transfer in `predict.py`.

```
chmod +x tensorize_model.py
cog run python tensorize_model.py
```

## Step 2: Run the model

You can run the model locally to test it:

```
cog predict -i prompt="What's in this image?" -i img="https://replicate.delivery/pbxt/Io3tVPIOTuYNQhEoYbl1JS7fi7NzZeIr2MgPnbLiFX3nP3t9/mplug-owl-llama-3.png"
```

## Step 3: Push your model weights to cloud storage

If you want to deploy your own cog version of this model, we recommend pushing the tensorized weights to a public bucket. You can then configure the `setup` method in `predict.py` to pull the tensorized weights. 

Currently, we provide boiler-plate code for pulling weights from GCP. To use the current configuration, simply set `TENSORIZER_WEIGHTS_PATH` to the public Google Cloud Storage Bucket path of your tensorized model weights. At setup time, they'll be downloaded and loaded into memory. 

Alternatively, you can implement your own solution using your cloud storage provider of choice. 

To see if the remote weights configuration works, you can run the model locally.

## Step 4: Create a model on Replicate

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model.

Make sure to specify "private" to keep the model private.

## Step 5: Configure the model to run on A100 GPUs

Replicate supports running models on a variety of GPUs. The default GPU type is a T4, but for best performance you'll want to configure your model to run on an A100.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 6: Push the model to Replicate

Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 3:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)