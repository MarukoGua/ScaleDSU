## ScaleDSU ##

This repository provides demonstrations for reproducing the results presented in "Modeling Device Shifts via Scaled Uncertainty for Cross-Device Side-Channel Analysis".


### Datasets ###
To download the datasets evaluated in the paper:

1. XMEGA, XMEGA_EM and SAKURA_AES:
    Follow the instructions at: https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack.git

2. AES_PTv2: 
     Follow the instructions at: https://github.com/urioja/AESPTv2.git

3. CHES Challenge 2025 
    Follow the instructions at: https://pace-tl.gitbook.io/ches-challenge-2025


### Reproducing the Results ###
To reproduce the results from the paper, follow these steps:

1. Download the required datasets as described above.

2. Clone this repository and navigate to the code directory:
    cd ./ScaleDSU

3. Check the requirements.txt file to ensure you get all the dependent libraries. To Install all required dependencies:
    pip install -r requirements.txt

4. Run the evaluation scripts. You may need to modify the data_file argument in each script to point to your local dataset directory:

    bash ./scripts/run_${dataset}.sh

OR to simply run all scripts:

    bash ./scripts/run_all.sh


5. Review the outputs in the ./result/ directory (e.g., best checkpoint files, logs, attack results etc.).
