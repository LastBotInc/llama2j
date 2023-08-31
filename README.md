# llama2j

Project is based on https://github.com/karpathy/llama2.c

This is a pure Java implementation of LLama 2 inference, without any dependencies.

In addition, we implement CUDA version of the same, where the transformer is implemented
as a number of kernels. Java code runs the kernels on GPU using JCuda.

The purpose of this project is to provide good-performance inference for LLama 2 models
that can run anywhere, and integrate easily with Java code. We desire to enable the LLM
locally available for backend code, and to scale horizontally to more nodes that each
can include both backend logic and local LLM inference.

Features:
- < 1 second startup time for LLama 7B model
- CPU support
- Single or multiple GPU support
- I8 quantization of weights on the fly
- Caching of I8 weights
- Activations are FP32, so this is W8A32 quantization
- CPU and CUDA are identical and validatable against each other

Tested on:
- LLama 7B model and smaller models
- CPU and GPU
- Windows and Linux
- Java 20
- CUDA 11.2
- JCuda 11.2.0
- 1-4x RTX 4090

## Performance

| Command                                                                                                                | Configuration 1 | Configuration 2 | Configuration 3 |
|------------------------------------------------------------------------------------------------------------------------|-----------------|-----------------|-----------------|
| llama2j --mode CPU --checkpoint Llama-2-7b-chat.bin                                                                    | 6.55 tok/s      | TBD             | TBD |
| llama2j --mode CUDA --checkpoint Llama-2-7b-chat.bin                                                                   | 19.77 tok/s     | TBD             | TBD |
| llama2.c compiled as make runomp and run as ./run lLlama-2-7b-chat.bin -t 1.0 -n 256 -i "One day, Lily met a Shoggoth" | 12.0 tok/s      | TBD| TBD|

## Usage

### Install dependencies

This also provides dependencies for using llama2.c for converting models to a llama2.c format
that llama2j can use.

```console
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential git cmake python3 clang libomp-dev git-lfs python3-pip maven -y
git config --global credential.helper store
pip install --upgrade huggingface_hub
pip install transformers
# Install Java 20
wget https://download.oracle.com/java/20/latest/jdk-20_linux-x64_bin.deb
sudo dpkg -i jdk-20_linux-x64_bin.deb

```

### Download and install

git clone https://github.com/LastBotInc/llama2j.git
cd llama2j
mvn clean build

### Download models

Get a 15M test model:

wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

Get your favorite LLama 7B based model from huggingface e.g.




run llama2j