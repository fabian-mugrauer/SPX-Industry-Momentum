# Use a base image with Miniconda installed
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Git
RUN apt-get update && apt-get install -y \
    git \
    docker.io 

# Clone the GitHub repository
RUN git clone https://github.com/fabian-mugrauer/SPX-Industry-Momentum.git .

# Add conda-forge to the list of channels
RUN conda config --add channels conda-forge

# Install Mamba from Conda-Forge
RUN conda install mamba -c conda-forge

# Initialize Mamba. This is important to ensure that the shell is aware of Mamba.
RUN mamba init bash

# Use Mamba to create the environment from the .yaml file
RUN mamba env create -f spx_industry_mom.yaml

# (Optional) Expose Docker socket for Docker-in-Docker (DinD)
VOLUME /var/run/docker.sock

# Activate the environment (this is just for documentation, 
# you'll need to activate it when running the container)
# Note that Docker containers do not maintain environment state between RUN commands
# Therefore, we activate the environment when running commands like ENTRYPOINT or CMD
ENV PATH /opt/conda/envs/spx_industry_mom/bin:$PATH