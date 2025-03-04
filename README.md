# 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark

![Preview](3mdbench.jpg)

**How to run and estimate the chosen model**:

0. Install dependencies from ```requirements.txt``` 
1. Download [images and complaints](https://drive.google.com/drive/folders/10j3bgase36w_IcEjGDgaErYFzVHiCjWZ?usp=sharing);
2. Go to the ```scripts``` folder;
3. Run ```run_dialogue.sh```, choosing models from used in the paper or implementing custom in the ```agents/doctor_agent.py``` file;
4. Run ```run_assessment.sh``` to estimate generated dialogue which will be contained in the ```results/assessments``` folder;
5. Run ```run_diagnoses_obtaining.sh```to extract final diagnoses by Doctor Agent for each case;
6. Explore ```benchmarking/count_metrics.ipynb``` to analyze model's metrics.