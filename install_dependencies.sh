echo "Installing PortAudio from source..."
apt-get update && apt-get install -y build-essential
curl -O http://files.portaudio.com/archives/pa_stable_v190600_20161030.tgz
tar -xvzf pa_stable_v190600_20161030.tgz
cd portaudio
./configure
make
make install
cd ..
pip install -r requirements.txt
