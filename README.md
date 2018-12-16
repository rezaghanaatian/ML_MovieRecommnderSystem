## Code Usage:

- Below, you can find out more about the codes in this repository and how you can use them:
 

1. In order to use the OOP framework, you can run the code using the script below (example):

	* `python  main.py --optimizer='als' --n_folds=5   --file_name='submission1'`


2. To see all the options and the parameters you can change please type:

	* `python main.py --help`


3. In order to use cross-validation for 'Blending' algorithm, all you need to do is ONLY run the command below (please use `--help` to see the list of available options and parameters you can set):

	* `python  main_blend.py  --file_name='blending_sub'`


4. The command in item 3 will give the 'best weights' for 'Blending'  (i.e. will not give out submission predictions and .csv file). To make the final 'recommendations' and the final submission file, update the weights in `main.py` file accordingly and run it using the command in item 1. 


