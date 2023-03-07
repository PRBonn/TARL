ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Receive it from host env variables build with:
# docker build . -t nuneslu/tarl:latest --build-arg UID=$(id -u) --build-arg GID=$(id -g)
#ARG UID
#ARG GID

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Add user to share files and folder without root permissions
#ENV GROUP_ID=$GID
#ENV USER_ID=$UID
#RUN addgroup --gid $GROUP_ID tarl && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID tarl
RUN adduser --disabled-password --gecos '' tarl
USER tarl

WORKDIR /home/TARL/tarl/
# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
