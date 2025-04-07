This folder contains the all necessary components to run SHyNN code successfully.

"SHyNN_paper.py" is the source file of SHyNN algorithm under python environment v. 3.11, tensorflow v. 2.13. It can be run readily in compatible python platform with required inputs. The inputs include: 
a) size of net
b) The .mat file, which is supposed to at least include original Jacobian matrices and measured datasets, with respect to different wavelengths, SD matrices and datasets, nodes and SD-CSCG result (the detail is given in matlab_data_format.JPG).
c) number of iterations

"SHyNN_python_interface.m" is the interface code that runs "SHyNN_paper.py" in Matlab. First, the user should check the compatibility between python and Matlab version, see:https://www.mathworks.com/support/requirements/python-compatibility.html.
Next, The code should be run successfully once the environment path is changed accordingly and the necessary inputs are given as the same as "SHyNN_paper.py". The SHyNN reconstruction result is finally recorded in "python_result".

"matlab_data_format.JPG" shows an example of the inputted matlab file to SHyNN. 