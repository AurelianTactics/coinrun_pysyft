# CoinRun with PySyft
* Need CoinRun (https://github.com/openai/coinrun) and PySyft (https://github.com/OpenMined/PySyft) installed per those repo's instructions.
* In this repo are some miscellaneous files for training, testing, and plotting results. Files can be run per CoinRun execution instructions
* Train an agent by running train_pysyft_coinrun.py (which loads a DQN agent depending on your options from the dqn_utils.py file), test the saved models with test_pysyft_coinrun_multi_level.py, and see plots of the testing results with the Jupyter Notebooks.
* Attempted files for running PySyft with secure aggregation and encrypted training are not included as I could not get those scripts to work.
