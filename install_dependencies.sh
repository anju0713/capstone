#!/bin/bash

# Install dependencies from source (PortAudio as an example)
echo "Installing PortAudio from source..."

# Install necessary tools
apt-get update && apt-get install -y build-essential

# Download and build PortAudio
curl -O http://files.portaudio.com/archives/pa_stable_v190600_20161030.tgz
tar -xvzf pa_stable_v190600_20161030.tgz
cd portaudio
./configure
make
make install

# Move back to your project directory
cd ..

# Continue with the normal Python dependencies
pip install -r requirements.txt
