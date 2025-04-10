# SHyNN
This repository includes Python code of Self-supervised hybrid neural network used for achieving quantitative bioluminescence tomography (QBLT) in BIRTlab. In SHyNN code, we use:

the conventional result as the initial value
neural network to approximate the light source distribution term S(x), with softplus activation function to gaurantee its nonnegativity 
a designed converging path by L2, L1, L0.05 regularizaions to guide the optimization

SHyNN code is written under the Python environment. However, a matlab interface file, "SHyNN_python_interface.m", is provided to call it through Matlab.
