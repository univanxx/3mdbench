# 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark

![Preview](3mdbench.jpg)

## How to run estimation

### Data & dependencies preparing

* Install dependencies from ```requirements.txt```
* Download [images and complaints](https://drive.google.com/drive/folders/10j3bgase36w_IcEjGDgaErYFzVHiCjWZ?usp=sharing)

---
### Dialogues generation

* Go to the ```scripts``` folder;
* Run ```run_dialogue.sh```, choosing models from used in the paper or implementing custom in the ```agents/doctor_agent.py``` file;

---
### Dialogues assessment

* Run ```run_assessment.sh``` to estimate generated dialogue which will be contained in the ```results/assessments``` folder;
* Run ```run_diagnoses_obtaining.sh```to extract final diagnoses by Doctor Agent for each case;
* Explore ```benchmarking/count_metrics.ipynb``` to analyze model's metrics.
