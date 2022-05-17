# word2vec
This project has been written in tensorflow and is designed to run in the Rosie tensorflow singularity container.


Instructions for running
1. Download this repository 
2. Upload it to Rosie and Unzip
3. Open a Rosie VS Code instance with 1 gpu
4. Navigate to the root directory of this folder in the terminal associated with the vscode instance
3. Install the required pip packages with `pip install -r requirements.txt` from the root of the repo
4. cd to `word2vec` with `cd ./word2vec`
5. Start an interactive tensorflow container session session with `singularity shell /data/containers/msoe-tensorflow-20.07-tf2-py3.sif` 
5. Run the example script with `python example.py`
6. View results