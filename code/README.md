# ML Security Practical Session

Setup: install the requirements from the `requirements.txt` file:

```shell
pip install -r requirements.txt
```

First, read the tutorial notebook. It can be found [here](../code/notebooks/adversarial_evasion_attacks.ipynb).

Then, we are going to write a pipeline for running a security evaluation experiment.

Complete the parts that are missing in the file `pipeline_for_robustness_evaluation.py`. 
They are marked with a `TODO` comment (automatically recognized by most IDEs).

For running the debugging script, issue the following commands in the terminal:

```shell
python -m pipeline_for_robustness_evaluation --help
```

First, debug the adversarial attack, find out if the attack is working well with the 
configuration you defined.

```shell
python -m pipeline_for_robustness_evaluation --model 0 --debug
```

Once everythin is fine, we can finally run the complete evaluation on more samples (recommended GPU, or limit the number 
of samples to a small value).

```shell
python -m pipeline_for_robustness_evaluation --model 0 --samples 5
```

