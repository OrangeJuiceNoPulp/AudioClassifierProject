# Audio Classifier Project
 Group 9's Final Project for CSI-4650 Parallel and Distributed Computing, Fall 2024


# Chord Classification

## Introduction
This project focuses on classifying audio samples of musical chords. The approach of this project involves converting `.wav` files into mel spectrograms and then using them to train two convolutional neural networks for classification, one for the type of chord and another for the chord's root.

This project has applications in music transcription. A successful model will be able to assist in analyzing music, which may help musicians to play or remix music, as detecting the chords of a song by ear can be a difficult task.

## Dataset
- The first half of the `.wav` audio samples were downloaded from the [Audio Piano Triads Dataset] (https://zenodo.org/records/4740877) created by Agustín Macaya Valladares, which contains 43,200 audio samples.
- Additional audio samples were generated from this original set of data, resulting in additional 43,200 audio samples.
- The process used to generate the additional audio samples is as follows:
	- Run the DataSorter Notebook file to sort the audio files by Chord Root and Chord Type. There will then be 48 folders containing 900 audio samples.
	- For all the audio samples in a given folder, select them all with `CTRL + A`, then drag them into Mixcraft 9 Recording Studio and drop them in an audio track while holding `SHIFT`.
	- Apply the Shred Amp Simulator `fx` to the audio track with the default settings (which should be the Marvel Crunch Amplifier).
	- Mixdown the project to `.wav`, name the file with the first 2 letters as the Chord Root, and the 3rd letter as the Chord Type, and do this for all the 48 folders.
	- Once all the files are created, run the `WavSplitter.ipynb` notebook file after updating the file paths to separate the large `.wav` files into smaller ones similar to those of the original dataset. (Running the `WaveSplitter.ipynb` notebook file requires FFmpeg to be installed and added to PATH.)
	- Finally, move all of the `.wav` files (both from the original dataset and the newly created ones) into a single folder and run the `DatasetMetadataGenerator.ipynb` notebook file after updating the file path in order to generate `.csv` files for training the models.
 - The additional audio samples will temporarily be available at: https://drive.google.com/file/d/1HsfatsMhJLSiQZRQ5mJgGFL-GYHxXwG5/view?usp=drive_link

## Installation and Quick Startup
1. Clone the repository (or download it from Github as a `.zip` file and extract it to a suitable location on your local device).
2. Create a Python virtual environment. (Python 3.10 was used when creating this project. Please use a version of Python that supports Pytorch.)
   ```bash
   python -m venv myenv
   ```
3. Activate the newly created virtual environment.
   ```bash
   source myenv/Scripts/activate
   ```
4. Run the pip install commands to install the dependencies.
   ```bash
   pip install notebook
   pip install jupyter
   pip install pandas
   pip install numpy
   pip install scikit-learn
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install PySoundFile
   pip install pydub
   pip install torch-summary
   pip install matplotlib
   pip install gradio
   ```
5. Run Jupyter Notebook.
   ```bash
   jupyter notebook
   ```
6. Use the available `.ipynb` files for your intended task.

## Prediction
1. Open the `PerformPrediction.ipynb` notebook.
2. Run all cells in the notebook
3. The last cell creates a Gradio interface for using the model to predict the chord type and root, so click on the link specified in the output of that cell.
4. Upload a `.wav` file to classify and trim the desired audio to 4 seconds if necessary.
5. Press the button to perform the prediction.

## Training
1. To train the chord root classifier, open the `RootTrainCNN.ipynb` notebook. For the chord type classifier, open the `TypeTrainCNN.ipynb` notebook.
2. Ensure that you have the full dataset downloaded and placed in a single folder. If you are using only the first half of the dataset (the portion from Zenodo.org), it will be necessary to regenerate the CSV files using the `DatasetMetadataGenerator.ipynb` notebook. (Simply update the paths and run all the cells to create the new `.csv` files.) Additionally, it will be necessary to change 86400 to the number of audio samples you are using in the line of code:
```python
ctd, unused = torch.utils.data.random_split(ChordTypeDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device), [NUM_DATA_ITEMS, 86400 - NUM_DATA_ITEMS])
```
3. Update the paths in the CNN training notebooks, ensuring that the `.csv` files and the audio dataset are in the correct places.
4. Update any parameters that need to be changed, such as the number of epochs, batch size, etc...
5. Run all of the cells from top to bottom to train a new network from scratch.
6. If you are training on CPU where CUDA is not available, it may be necessary to comment out the lines "torch.cuda.current_stream().synchronize()" which are present throughout the program.
7. To continue training an existing model, convert the bottom cell from raw text to Python code and run it.

- Training on the entire dataset takes a long time. The resulting epoch times for training the two CNN models are present in the `ChordRootFinalOutput.txt` and the `ChordTypeFinalOutput.txt` files respectively. The checkpoints for these models are present in the repository as `ChordRootCNN.pth` and `ChordTypeCNN.pth` respectively.

## Reproducing Experimental Results
The experiments were run on an Intel Core i7-10750H processor (2.60 GHz, 6 Cores) for the CPU and an NVIDIA GeForce RTX 2070 with Max-Q Design for the GPU.

Experimental results are available in the project presentation slides file.

Notice: These experiments took more than an hour to conclude. If you have limited time, consider modifying the code to reduce the number of iterations.

### Reproducing Experiment 1
- Ensure that the dataset is properly prepared and that the `.csv` files are correct, updating the necessary file paths in the `BenchmarkingBatchSizes.ipynb` notebook if necessary. If there are a different amount of audio samples present in the `.csv` file, it will be necessary to apply the same fixes to the code as were done in Step 2 of training.
- On a device with CUDA available, run all the cells in the `BenchmarkingBatchSizes.ipynb` notebook. The currently set parameters are the ones which were used in the experiment.
	- This code will output tables containing the average epoch times for training/testing with a 70/30 train-test split with various batch sizes [32, 64, 128, 256] across five iterations of 5 epochs each. (A 0-th iteration is performed as well to warm-up the machine, but its results are not kept or displayed at all.)
	- Additionally, graphs displaying the overall average epoch times across the various batch sizes are displayed.
	- All of the above is done between the two configurations of CUDA (GPU) and CPU for both the chord type CNN and the chord root CNN.

### Reproducing Experiment 2
- Ensure that the dataset is properly prepared and that the `.csv` files are correct, updating the necessary file paths in the `BenchmarkingNumDataItems.ipynb` notebook if necessary.
- On a device with CUDA available, run all the cells in the `BenchmarkingNumDataItems.ipynb` notebook. The currently set parameters are the ones which were used in the experiment.
	- This code will output tables containing the average epoch times for training/testing with a 70/30 train-test split with various dataset subset sizes [4096, 8192, 16384] across five iterations of 5 epochs each. (A 0-th iteration is performed as well to warm-up the machine, but its results are not kept or displayed at all.)
	- Additionally, graphs displaying the overall average epoch times across the various subset sizes are displayed.
	- All of the above is done between the two configurations of CUDA (GPU) and CPU for only the chord type CNN.


## Notes
- Ensure your system has the required dependencies installed before running the notebook.
- For GPU acceleration, ensure you have a compatible CUDA setup.

## References

### Dataset
- https://zenodo.org/records/4740877
- https://drive.google.com/file/d/1HsfatsMhJLSiQZRQ5mJgGFL-GYHxXwG5/view?usp=drive_link

### Programs and Libraries
- https://www.python.org/downloads/
- https://www.ffmpeg.org/documentation.html
- (It appears that Mixcraft 9 Recording Studio is no longer available, so this links to the newest Mixcraft Recording Studio):
https://store.acoustica.com/bundles/mixcraft-10-recording-studio
- https://jupyter.org/install
- https://pandas.pydata.org/docs/getting_started/install.html
- https://numpy.org/install/
- https://scikit-learn.org/1.5/install.html
- https://pytorch.org/get-started/locally/
- https://pypi.org/project/torch-summary/
- https://matplotlib.org/stable/install/index.html
- https://www.gradio.app/guides/quickstart
- https://pypi.org/project/PySoundFile/
- https://pypi.org/project/pydub/
- https://developer.nvidia.com/cuda-12-1-0-download-archive

### Coding Resources Consulted
- https://www.youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm
- https://www.youtube.com/watch?v=V_xro1bcAuA
- https://www.youtube.com/watch?v=dOG-HxpbMSw
- https://www.youtube.com/watch?v=gs0FNQR0njI
- https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
- https://stackoverflow.com/questions/78097861/how-to-solve-runtimeerror-couldnt-find-appropriate-backend-to-handle-uri-in-py
- https://discuss.pytorch.org/t/torch-operation-time-measurement/202796

### Helpful Resource For Learning About Convolutional Neural Networks
- https://poloclub.github.io/cnn-explainer/

## License
This project uses the [Zenodo Piano Chord Dataset] (https://zenodo.org/records/4740877). Please review the dataset license for more details.

