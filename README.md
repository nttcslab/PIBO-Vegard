
### Paper
Paper URL: https://doi.org/10.1038/s41524-025-01522-8

Cite the following article to refer to this work.
```BibTeX
@article{wo2023,
  title = {Physics-informed {Bayesian} optimization suitable for extrapolation of materials growth},
  author = {W. Kobayashi and Takuma Otsuka and Yuki K. Wakabayashi and G. Tei},
  journal = {npj Computational Materials},
  volume = {11},
  pages = {36},
  doi = {https://doi.org/10.1038/s41524-025-01522-8},
  year = {2025}
}
```

### How to run
Use `run_PIBO_1st.py` to reproduce Figure 4 in our paper. 
Similarly, `run_PIBO_2nd.py` produces Figure 7.

You can specify the target composition (*x, y*) by `--target` option. 
For example, `run_PIBO_1st.py --target 0.19,0.42`, which means the target values are *x*=0.19 and *y*=0.42.


### Software version
Codes are confirmed to run with the following libraries. Likely to be compatible with newer versions. 

* `python`: `3.11.5`
* `numpy`: `1.24.3`
* `scipy`: `1.11.1`
* `sklearn`: `1.3.0`
* `matplotlib`: `3.7.2`
* `seaborn`: `0.12.2`

### Files
* `README.md`: This file. 
* `LICENSE.md`: Document of agreement for using this sample code. Read this carefully before using the code. 
* `code`: Contains codes
  * `run_PIBO_1st.py`: Script to execute PIBO for the first experiment (Fig. 4 in our paper). 
  * `run_PIBO_2nd.py`: Script to execute PIBO for the second experiment (Fig. 7 in our paper). 
  * `BO_target.py`: Implements BO class. 
  * `utils.py`: Contains internal functions. 
  * `lhsmdu.py`: Latin hypercube sampling package for acquisition function. Repository: https://dx.doi.org/10.5281/zenodo.2578780  
* `data`: Contains data
  * `data_1st.csv`: Experimental data of 6 trials for `run_PIBO_1st.py`.
  * `data_2nd.csv`: Experimental data for `run_PIBO_2nd.py`. Former 7 points were collected through the first experiment. Latter 5 points were measured in the second phase of experiment. 
