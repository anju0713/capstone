echo "Installing PortAudio from source..."

# Fix the file format if needed
dos2unix install_dependencies.sh

# Update and install necessary packages
apt-get update && apt-get install -y build-essential

# Download PortAudio
curl -O http://files.portaudio.com/archives/pa_stable_v190600_20161030.tgz

# Extract the tarball
tar -xvzf pa_stable_v190600_20161030.tgz

# Change to the extracted directory
cd portaudio

# Configure, make, and install
./configure
make
make install

# Move back to the original directory
cd ..

# Install Python dependencies
pip install -r requirements.txt
