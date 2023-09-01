# llama2j

The llama2j project is based on https://github.com/karpathy/llama2.c

This is a pure Java implementation of standalone LLama 2 inference, without any dependencies.

In addition, we implement CUDA version, where the transformer is implemented
as a number of CUDA kernels. Java code runs the kernels on GPU using JCuda.

The purpose of this project is to provide good-performance inference for LLama 2 models
that can run anywhere, and integrate easily with Java code. We desire to enable the LLM
locally available for backend code. LLM becomes a seamless and integrated part
of application backend functionality, and can be scaled together with the backend.

Features:
- 4 second startup time for LLama 7B model
- CPU support
- Single or multiple Nvidia GPU support
- I8 quantization of weights on the fly
- Caching of I8 weights
- Activations are FP32 (this is W8A32 quantization)
- CPU and CUDA implementations are identical and validatable against each other

Tested on:
- Ubuntu 22.04.02 and 22.04.03
- Windows 11 Version 10.0.22621 Build 22621
- LLama 7B model and smaller models
- Intel and AMD CPUs
- Java 20
- CUDA 11.2
- JCuda 11.2.0
- 1-4x RTX 4090

# Performance

Tokens per second is printed out at the end of the run, and it excludes model checkpoint
loading time.

NOTE: llama2.c has been compiled as 'make runomp' for the fastest performance.

| Command                                              | Configuration 1 | Configuration 2 | Configuration 3 |
|------------------------------------------------------|-----------------|---------------|----------------|
| llama2j --mode CPU --checkpoint Llama-2-7b-chat.bin  | 6.6 tok/s       | 4.0 tok/s     | 1.8 tok/s      |
| llama2j --mode CUDA --checkpoint Llama-2-7b-chat.bin | 20.9 tok/s      | 21.0 tok/s    | 17.0 tok/s     |
| llama2.c (OMP_NUM_THREADS=32)                        | 12.0 tok/s      | 2.3 tok/s     | -              |
| llama2.c (OMP_NUM_THREADS=64)                        | 9.5 tok/s       | 2.2 tok/s     | -              |

The duration of a model checkpoint loading depends on if the model is being loaded for the first
time, or if it already has been processed and cached. The time includes allocating memory, loading weighs
from the disk, if necessary, quantifying the weights, and transferring the data to GPU devices.

| Command                                                                  | Configuration 1 | Configuration 2 | Configuration 3 |
|--------------------------------------------------------------------------|-----------------|-----------------|-----------------|
| Load Llama-2-7b-chat for the first time, quantize, and store quant files | 15.3 s          | 27.4 s          | 38.0 s          |
| Load Llama-2-7b-chat from cached quant files                             | 0.8 s           | 1.3 s           | 1.7 s           |

The test system configurations are:

| Configuration   | System                                                                                                              |
|-----------------|---------------------------------------------------------------------------------------------------------------------|
| Configuration 1 | Ubuntu 22.04.3, MZ33-AR0-000, AMD EPYC 9374F 32-core processor, (1 of 4) * Nvidia 4090, 368GB 4800 DDR5             |
| Configuration 2 | Ubuntu 22.04.3, ROG CROSSHAIR X670E EXTREME, AMD 9750x 16-core processor, 1 * Nvidia 4090, 64GB 4800 DDR5           |
| Configuration 3 | Windows 11 Pro Build 22621, ROG MAXIMUS Z790 APEX, Intel 13900KS 24-core processor, 1 * Nvidia 4090, 32BG 7600 DDR5 |

# Quick and Easy Installation

For Ubuntu, follow these instructions. For Windows 11, see the section below.

## Install dependencies

This also provides dependencies for using llama2.c for converting models to a llama2.c format
that llama2j can use.

```console
# install dependencies needed to also run llama2.c and
# to process models from hugging face and to set system performance configuration

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential git cmake python3 clang libomp-dev git-lfs python3-pip maven tuned \
linux-tools-6.2.0-26-generic linux-cloud-tools-6.2.0-26-generic \
linux-tools-6.2.0-31-generic linux-cloud-tools-6.2.0-31-generic \
linux-tools-generic \
linux-cloud-tools-generic \
-y
git config --global credential.helper store
pip install --upgrade huggingface_hub
pip install transformers

# Install Java 20

wget https://download.oracle.com/java/20/latest/jdk-20_linux-x64_bin.deb
sudo dpkg -i jdk-20_linux-x64_bin.deb
# make sure JDK 20 java is first in your path
export PATH=/usr/lib/jvm/jdk-20/bin/:$PATH
# add the same path setting also ~/.bashrc if you prefer
# check you have the correct Java, e.g java 20.0.2 or later
java --version
```

## Set up CUDA

First check that you have NVIDIA drivers installed. If not, download and install them from nvidia site. And Good luck!

```console
nvidia-smi 
```
should show a Driver Version that is >= 525.00

```console
Thu Aug 31 13:04:25 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0  On |                  Off |
|  0%   43C    P8              28W / 450W |   1294MiB / 24564MiB |      2%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

Check CUDA version carefully. If it is anything else than 12.0
(for example 12.2 is not compatible), install CUDA 12.0 following
the exact instructions below. This will help you not to break any
other dependencies you might have to your current drivers.

NOTE: do not replace the drivers (unless you want to).
- Uncheck 'install drivers'!
- Accept other defaults & install

```console
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run
```

Installer will complain "***WARNING: Incomplete installation!" which
is not an error condition. You have your drivers and are good to go.

## Download and build llama2j

On Ubuntu, now everything is ready for cloning and building the project.

```console
git clone https://github.com/LastBotInc/llama2j.git
cd llama2j
mvn clean package
```

On Windows, just clone the project.

```console
git clone https://github.com/LastBotInc/llama2j.git
```

## Windows 11 quick installation guide

- Install git from https://git-scm.com/download/win
- Install maven from https://maven.apache.org/download.cgi
- Open PowerShell Prompt
- Go to the 'model' subdirectory under 'llama2j' directory, for example

```console
cd .\IdeaProjects\llama2j\models
```

Download a test model file

```console
curl https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -OutFile stories15M.bin
```
CUDA toolkit

- Install Anaconda on Windows
- Open 'Anaconda Prompt (miniconda3)' from the start menu
- Install CUDA 12.0 on conda

```console
conda install cuda -c nvidia/label/cuda-12.0.0
```

Microsoft C++ compiler

- Install Microsoft Visual Studio or Build Tools.
- Add Microsoft C++ compiler "cl.exe" into your PATH variable. Location varies, but on the test computer
it is:
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64

Now, everything should be ready to build on Windows.

```console
mvn clean
mvn package
```

That's all for Windows. For any further steps, follow the instructions below.

NOTE: Model conversion is only supported under Ubuntu. On Windows you can use WSL.

## Back to Ubuntu, set up your system for performance

On Ubuntu, go to directory "llama2j/environment". Review the script setup_amd.sh. The script sets system-wide
parameters for better performance. It is geared towards AMD processors, but likely will benefit Intel as well.

NOTE: the script sets system-wide kernel parameters permanently. In case you are using anything other than
a disposable instance, please review the settings carefully, and remove any setting you are not comfortable with.

```console
cd llama2j/environment
sh ./setup_amd.sh
```

## Download a test model

Get a 15M test model:

```console
cd models
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

## Give it a spin on CPU!

```console
./run.sh --mode CPU --checkpoint stories15M.bin --prompt "One day, Lily met a Shoggoth"
```

On Windows, please use run.cmd.

```console
run.cmd --mode CPU --checkpoint stories15M.bin --prompt "One day, Lily met a Shoggoth"
```

## Run for your money on GPU!

NOTE: configure the --gpuMem according to how much GPU memory you want to allocate
on each device. For small models (up to 7B) configuring only one device is sufficient
and provides better latency than using more devices.

To use 24G of the first CUDA device (0):

```console
./run.sh --mode CUDA --gpuMem 24 --checkpoint stories15M.bin --prompt "One day, Lily met a Shoggoth"
```
To use 17G of the first CUDA device (0), and 24G on the devices (1), (2) and (3):

```console
./run.sh --mode CUDA --gpuMem 17,24,24,24 --checkpoint stories15M.bin --prompt "One day, Lily met a Shoggoth"
```
# Take Advantage of Hugging Face LLama2 models

## Log in to your Hugging Face account

NOTE: if you installed in a virtual environment, change
the path to huggingface-cli correspondingly.

```console
~/.local/bin/huggingface-cli login
```

Provide your Hugging Face token (see https://huggingface.co/settings/tokens)
and select Y on:

```console
Add token as git credential? (Y/n) Y
```

## Clone llama2.c

Clone llama2.c to a separate directory. We need it to convert Hugging Face 
models to a llama2.c format that llama2j supports natively.

```console
git clone https://github.com/karpathy/llama2.c.git
```
Then, open the repository folder:

```console
cd llama2.c
```
Install requirements:

```console
pip install -r requirements.txt
```

## Download and convert your favorite LLama 7B based model checkpoints from Hugging Face

The example below is for llama-27b-chat model.

NOTE: Only 7B models have been tested.
The checkpoint conversion depends on the code path of  llama2.c,
where it appears larger models may yet not be support. In llama2j,
there are also some points where the code used ints instead of longs
to address weight data.

NOTE: Do not download "hf" version of the models.

Enter credentials when prompted.

```console
git clone https://hf.co:/meta-llama/Llama-2-7b-chat
```

Convert the model. This may take some 5-10 minutes! Yes, it's single-threaded and slow. This is one-off per model.

NOTE: replace the model name and model destination path to your selection.

```console
python3 export.py --meta-llama Llama-2-7b-chat <YOUR_PATH>/llama2j/models/Llama-2-7b-chat.bin
```

Check that the model file looks fine:

```console
ls -l <YOUR_PATH>/llama2j/models/Llama-2-7b-chat.bin
```

The file should be in your llama2j/models directory and likely be 26GB in size. An example below.

```console
-rw-rw-r-- 1 tero tero 26G Aug 31 14:12 /home/tero/llama2j/models/Llama-2-7b-chat.bin
```

## Now, run the model in CUDA.

At the first run, the startup takes ca. 20-30 seconds when the model checkpoint is automatically converted to quant files, which are cached to local files.

```console
./run.sh --mode CUDA --checkpoint Llama-2-7b-chat.bin --gpuMem 17 --prompt "One day, Lily met a Shoggoth"
```

# Future development ideas

PRs are welcome!

| Topic                                                       | Work estimate |
|-------------------------------------------------------------|---------------|
| Test the conversion of larger LLama2 models up to 70B       | 16-40 hours   |
| Implement and test grouped query attention for 70B          | 16 hours      |
| Implement and optimize I4 weights (today I8)                | 16 hours      |
| Implement and optimize FP16 activations (today FP32)        | 16 hours      |
| Implement optimize parallel processing of multiple queries  | 32 hours      |
| Implement optimized tensor core kernels for MatMul          | 36 hours      |
| Implement OpenAI compatible API                             | 8 hours       |

We expect the I4 will cost 5-10% of performance
(while the memory bandwidth requirement will be reduced, there is slightly
more computational load). FP32 to FP16 will improve performance.
It is hard to estimate how that will interplay together with I4. Multiple
queries (large batch size) will help to maximize the use of GPU resources,
and performance can greatly benefit from the larger batch size. When the weights are
loaded to the shared memory, they can be efficiently applied to multiple queries that
run in parallel. This may also be the only reasonable way to overcome the
eventual ultimate memory bandwidth issue. There are other options too, such as
weight compression (in addition to quantization) but there are substantial
challenges to optimize the decompression to the speed level it would not
actually crash performance.

Tensor core kernels might accelerate matrix-vector multiplication, but the real
benefit also requires fragmenting and aligning the quantized weights at processing
time so that they fit the tensor core operation sizes naturally for the selected
primitive types.

On a device with 4 * 4090, today this code runs 4 parallel queries each
at 20 tokens/s, which is already a good speed for many applications,
including interactive voice applications. We should be able to reach
total of 200 tokens/s on that platform, and 50 tokens on a single 4090.

Please let us know what do you think!

# What is LastBot

We are building the ultimate conversation machine, which automates
engagement throughout the customer journey. All communication is thoughtful,
efficient, and impactful. LastBot always knows how to choose the right words,
and continuously online-learns from the real-life events to make even better
choices.

# Contact

tero@lasbot.com
