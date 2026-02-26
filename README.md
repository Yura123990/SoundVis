# The method of searching for acoustic signals at high noise levels
#### * Important note: the sndvis-0.1 folder is the latest stable version at the moment!
In the modern world, information is often hidden not only in visible data but also in weak, almost imperceptible signals. Due to the ongoing war in Ukraine, which affects not only my country but also neighboring states, FPV-drones frequently cross international borders, which causes a serious threat to the safety of the civilian population and infrastructure. Detecting such drones early, especially when they are still far beyond the horizon, requires advanced methods for identifying extremely weak and noisy signals. My algorithm offers a potential solution by enhancing periodic patterns hidden within noise, making it applicable to real-world defense and monitoring systems.

### The signal analysis process consists of several consecutive stages.
1. The first stage involves loading the audio and setting the sampling parameters—the user selects the file via the graphical interface. 
2. Next, fragmentation occurs — the signal is divided into short time segments (shells).
3. The next stage is the summation of shells, where all fragments are added to each other.
4. The result is analyzed using a neural network that recognizes repetitive frequency patterns. If the signal has a periodic structure, its components accumulate linearly, while noise accumulates stochastically, i.e., it decreases.
5. At the end, the result is visualized, and the user sees graphs where the moment of a stable signal appears becomes obvious.

## To train the AI model, use my Google Colab listed below
[Google Colab](https://colab.research.google.com/drive/1a15WNj1I2PczWDd3ZjYnwFfXFf78GcYk?usp=sharing)

#### Note: for dataset please use .wav files at 44100 sample rate!
[My dataset](https://drive.google.com/drive/folders/18Tr8uOFEd0IvkudYNmKNY6Fyin6IvWnX?usp=sharing) 
