# A Hierarchically Resistive Skin as Intelligent Multimodal On-Throat Wearable Biosensors

Motivation
==========
Our team report on a hierarchically resistive skin (HR-skin) which can overcome the aforementioned challenges, offering the ability of “one resistive signal - multiple biometrics”. By integrating resistive skins with different tiers into one single sensor, we could achieve an unconventional staircase-like resistive responses to tensile stains. This HR-skin sensor, in conjunction with Machine learning, offers the unprecedented ability of identifying five physical/psychological activities (speak, heartrate, breath rate, touch, and neck movement) from a single electrical signal.

Methods
=======

network
-------
Regarding the neural network, we initially trained two different 4-layer fully connected neural networks for classification. The two networks have the same numbers of hidden layers and neurons, but the different parameters, because one is based-on raw signals, named time-domain neural network, and the other employed FFT responses, called frequency-domain neural network. Next, based on two pretrained networks, we trained the third fully connected neural network, taking the input of concatenated outputs from the second-to-last layers from these two networks.

signal processing
-----------------
we used fft to extract the frequency features of raw signals, and used some filters for detecting two rates.

Experiment
==========
The network aims to classify 11 classes of throat activities, including speaking yes/no/one/two, nodding/shaking/stretch,
nodding+yes/nodding+no/shaking+yes/shaking+no. 

The classification accuracies of the time-domain model and frequency-domain model could reach up to **85.69% ±1.38%** and **71.38% ±1.76%** respectively. 

The final classification accuracy of the fused model could reach up to **88.93% ±0.28%** for 11 throat activities.

Addational
==========
The folder "live demo" is only a toy model for live demonstration.
The paper "*A Hierarchically Resistive Skin as Intelligent Multimodal On-Throat Wearable Biosensors*" is under reviewed by Nature Materials now.
