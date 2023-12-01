# Use a base image with Miniconda installed
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Git
RUN apt-get update && apt-get install -y \
    git \
    docker.io 

# Clone the GitHub repository into a specific folder
RUN mkdir SPX-Industry-Momentum && \
    git clone https://github.com/fabian-mugrauer/SPX-Industry-Momentum.git SPX-Industry-Momentum

# Change working directory to the cloned repository
WORKDIR /usr/src/app/SPX-Industry-Momentum

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

# Create environment_variables.env file in the specified directory
RUN mkdir -p /usr/src/app/SPX-Industry-Momentum/notebooks && \
    echo "PROJECT_ROOT=/usr/src/app" > /usr/src/app/SPX-Industry-Momentum/notebooks/environment_variables.env

# Activate the environment (this is just for documentation, 
# you'll need to activate it when running the container)
# Note that Docker containers do not maintain environment state between RUN commands
# Therefore, we activate the environment when running commands like ENTRYPOINT or CMD
ENV PATH /opt/conda/envs/spx_industry_mom/bin:$PATH

# Install Jupyter
RUN mamba install -n spx_industry_mom jupyter

# Expose the port Jupyter will run on
EXPOSE 8888

# Run Jupyter notebook
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
