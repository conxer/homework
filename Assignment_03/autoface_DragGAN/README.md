# Clone the DragGAN repository and install dependencies
git clone https://github.com/Zeqiang - Lai/DragGAN.git
cd DragGAN
conda create -n draggan python=3.7
conda activate draggan
pip install -r requirements.txt

# Solve the gradio connection error (if needed)
pip install pydantic==1.10.11

# Install face - alignment related
git clone https://github.com/1adrianb/face - alignment
cd face - alignment
pip install -r requirements.txt
python setup.py install

# Run
python gradio_app.py
# Result
