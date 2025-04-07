% In this SHyNN (quantitative bioluminescence tomography) code, we do: 
% 1. import the data processed by QBLT_v2 in MATLAB
% 2. least square fitting with the SD-CSCG results
% 3. SHyNN training to optmize the reconstruction
% to reconstruct quantitative information of the target, like volume, positioning, and light power. 
% The mesh format follows Nirfast mesh format. 
%
% Beichuan Deng, UT Southwestern, Feb 2025.

%%
% Requires certain matlab version depending on your installed python ver.
% e.g. Python 3.11 requires R2024b.
pyenv("ExecutionMode","OutOfProcess")
pyenv('Version', 'C:\Users\S226397\.conda\envs\tf2\python.exe');   % change to the path where your python is installed.
folderPath = 'C:\Users\S226397\Documents\Python Scripts';         % change to the path where your .py source code is saved.
if count(py.sys.path, folderPath) == 0
    insert(py.sys.path,int32(0),folderPath);
end

myMLModule = py.importlib.import_module('SHYNN_paper');

net_size = int32([64,64, 96, 1]);                 % network size
p = [2.0, 1.0, 0.05];                             % ordered p values for Lp regularization
a = int32([10, 10]);                              % number of iterations for each p, adaptive learning rate if 2<=dim<=4 
numberwv = int32(3);                              % number of wavelengths
np = py.importlib.import_module('numpy');

model = myMLModule.SHyNN("blt_m8_simu_Srgt_0noise_g100", net_size,a,p, 'GB', numberwv);   % input the .mat file that restores nodes, SD results, all Jacobians and measurements
model.fitting_model()                             % apply least square fitting with SD result
model.train_shynn()                               % apply optimization
%%
python_result = model.model(model.pts).numpy();   % output final result

% Inspect the type of python_result
disp('Type of python_result:');
disp(class(python_result));

% Convert NumPy array to MATLAB array
if isa(python_result, 'py.numpy.ndarray')
    A = double(python_result);  % Convert to MATLAB array
else
    error('Unsupported type for python_result.');
end
