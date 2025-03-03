# 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark

How to run and estimate the model:

1. Download [images and complaints](https://drive.google.com/drive/folders/10j3bgase36w_IcEjGDgaErYFzVHiCjWZ?usp=sharing);
2. Run ```scripts/run_dialogue.sh```, choosing models from used in the paper or implementing your own in the ```agents/doctor_agent.py``` file;
3. Run ```scripts/run_assessment.sh``` to estimate generated dialogue which will be contained in the ```results/assessments``` folder;
4. Run ```scripts/run_diagnoses_obtaining.sh```to extract final diagnoses by Doctor Agent for each case
5. Run ```benchmarking/count_metrics.ipynb``` to estimate the model

