
# Speech Background Noise Suppression with Deep Learning

**Company / Org:** MathWorks  
**Challenge Advisor:** Neha Sardesai, nsardesa@mathworks.com.   
**AI Studio Coach:** Bhavya Gopal, bhavya.gopal@breakthroughtech.org   
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About MathWorks
MathWorks is the leading developer of mathematical computing software for engineers and scientists worldwide. Their flagship products, MATLAB and Simulink, empower innovators to accelerate the pace of engineering and scientific discovery across industries such as automotive, aerospace, and communications. The team objectives for this challenge focus on enhancing speech processing capabilities to improve real-world accessibility and communication clarity.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use labeled speech and background noise audio data (from the Microsoft DNS Challenge dataset) and deep learning techniques (CNNs or RNNs), together with signal processing and feature extraction, to build and validate a model that suppresses background noise from speech signals. Teams will choose either a MATLAB-based or a Python-based (e.g., TensorFlow/Keras or PyTorch, Librosa) implementation, using architectures lightweight enough to train within the resources available to your chosen environment (e.g., free-tier Google Colab for Python). To keep the scope well-defined, the project will focus on a specific, manageable subset of noise types from the DNS dataset rather than general-purpose noise suppression. This will help MathWorks advance speech enhancement tools that improve the quality of life for people with hearing impairments and noise suppression in online meeting environments.

### Success Criteria

_Objective Metrics:_
- PESQ (Perceptual Evaluation of Speech Quality) — standard ITU metric for quantifying speech quality improvement after denoising
- STOI (Short-Time Objective Intelligibility) — measures how intelligible the cleaned speech is

A successful outcome would show meaningful improvement in both scores compared to the noisy (unprocessed) baseline.

_Subjective Evaluation:_
- Informal listening tests where the denoised audio is compared against the noisy input — the cleaned speech should sound noticeably clearer with reduced background interference.

_A successful December outcome looks like:_   
- A trained deep learning model, built in either MATLAB or Python that (1) demonstrably reduces background noise across varied noise types from the DNS dataset, (2) achieves competitive PESQ/STOI scores relative to published benchmarks, and (3) is documented well enough that the approach and results can be clearly presented — ideally with a live or recorded demo.

### Stretch Goals

1. Deploy the trained model to process audio in real time using streaming audio capabilities — this is the most impactful stretch goal given the hearing aid application context.
2. Evaluate robustness on a broader range of noise types beyond the initial project scope — including types outside the DNS dataset, such as cafeteria noise, music, or wind — to test real-world applicability.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Foundation & Data Pipeline | Get the environment set up and data ready. Students should complete a literature review on deep learning-based noise suppression, download and explore the Microsoft DNS Challenge dataset, define the target subset of noise types for the project, decide on MATLAB or Python as their working environment and implement basic audio preprocessing and feature extraction (e.g., spectrograms). |
| October | Model Design & Training | Design and train the deep learning network. Students should select a lightweight architecture (CNN or RNN) suited to their chosen environment's available resources, train on the processed dataset, and iterate on hyperparameters. By end of month, the model should produce preliminary denoised audio outputs. |
| November | Evaluation & Delivery | Validate the trained model using both subjective (listening tests) and objective metrics (e.g., PESQ, STOI). Students should document results, reflect on limitations, and prepare a final demo or presentation showcasing the model's performance on real-world noisy speech samples. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** Microsoft DNS Challenge (Deep Noise Suppression)   
**Format:** Audio files (e.g, .mp3, .wav)   
**Size:** over 10gb  
**Location:** https://github.com/microsoft/DNS-Challenge

### Key Details
- **What's in the data:** Raw clean speech, noise clips, and room impulse responses (RIRs) — not pre-mixed pairs. Clean speech is drawn largely from Librivox (~500 hours, 2,150 speakers); noise clips come from AudioSet and Freesound (~150 classes, ~60,000 clips); 118,000+ real and synthetic RIRs support reverberation simulation.   
- **Preprocessing needed:** Noise classes are imbalanced and were sampled/speech-filtered by the organizers, but teams still need to run the repo's synthesizer script (noisyspeech_synthesizer_singleprocess.py) to mix clean speech + noise + RIRs into labeled training pairs at chosen SNR levels. Editions vary widely in size (16kHz vs. 48kHz tracks, up to ~1TB for the largest), so confirm which subset fits the ~10GB budget above.   
- **Documentation:** microsoft/DNS-Challenge repo; INTERSPEECH 2020 / ICASSP 2021 challenge papers.
  
---

## 🛠️ Suggested Approach

**ML Problem Type:** Regression, Time Series Analysis, Deep Learning / Neural Networks, Transfer Learning / Pre-trained Models

**Recommended Libraries / Toolboxes:**
***If working in Python:***
- TensorFlow/Keras or PyTorch — model building and training
- Librosa — audio loading, preprocessing, and feature extraction (e.g., spectrograms, MFCCs)
- NumPy / SciPy — signal processing

***If working in MATLAB:***
- Deep Learning Toolbox — model building and training
- Audio Toolbox — audio loading, preprocessing, and feature extraction
- Signal Processing Toolbox — signal processing

**Evaluation Metrics:**
- PESQ, STOI (see Success Criteria above)
  
---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [Speech Enhancement Using Deep Learning Methods: A Review](https://www.researchgate.net/publication/354252395_Speech_Enhancement_Using_Deep_Learning_Methods_A_Review) — accessible survey of DNN/CNN/GAN approaches to noise suppression
- [Deep Neural Networks for Speech Enhancement and Speech Recognition](https://www.sciencedirect.com/science/article/pii/S2090447925001467): A Systematic Review — overview of CNN/RNN/LSTM architectures used for denoising

**Technical Tutorials:**
- Python: [Librosa documentation](https://librosa.org/doc/latest/index.html) — audio loading, spectrograms, MFCCs, and other feature extraction
- Python: [pesq](https://pypi.org/project/pesq/) and [pystoi](https://github.com/mpariente/pystoi) packages — compute PESQ/STOI scores directly on numpy audio arrays
- MATLAB: [Denoise Speech Using Deep Learning Networks](https://www.mathworks.com/help/deeplearning/ug/denoise-speech-using-deep-learning-networks.html) — official example comparing fully connected vs. convolutional networks on the same denoising task

**Code Examples:**
- [vbelz/Speech-enhancement](https://github.com/vbelz/Speech-enhancement) (Python/Keras) — U-Net denoiser with a full data-creation → training → prediction pipeline
- [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (Python/Rust) — low-complexity, real-time full-band speech enhancement; useful reference for the real-time stretch goal

**Other:**
- [DNS Challenge Project Site](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2023/) — challenge tracks, rules, and links to each year's results
- [ICASSP 2021 DNS Challenge paper](https://arxiv.org/abs/2009.06122) — dataset construction and baseline model details

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* Email preferred nsardesa@mathworks.com; please copy your teammates and AI Studio Coach
* Your team's channel within Break Through Tech’s Discord space
* Request a team check-in on Zoom
* Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.


---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
