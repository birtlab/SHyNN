import scipy.io  # For loading and saving MATLAB files
import numpy as np  # For numerical operations
import tensorflow as tf  # Deep learning library
import matplotlib.pyplot as plt  # For plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting utilities
from scipy.io import savemat  # For saving data to MATLAB files

# Set the default data type for Keras (TensorFlow)
tf.keras.backend.set_floatx('float64')

class SHyNN:
    def __init__(self, matlab_file, net_size, a, ICtype, p, num_wavelengths=3):
        """
        Initialize the SHyNN model with adaptive regularization weights.

        Args:
            matlab_file (str): Path to MATLAB data file.
            net_size (list): Network architecture [layer1, layer2, ...].
            a (list): Training steps per phase.
            ICtype (str): Initial condition type ('GB' or 'SD').
            p (list or str): Regularization parameters. Use ["h1", ...] for hybrid regularization.
            num_wavelengths (int): Number of wavelengths (3 or 4).
        """
        # Store constructor arguments
        self.matlab_file = matlab_file
        self.net_size = net_size
        self.a = a              # List of training iteration counts per stage
        self.ICtype = ICtype
        self.num_wavelengths = num_wavelengths
        self.p = p              # Regularization parameters
        
        # Learning rates for different training phases
        self.lr = [0.001, 0.0005, 0.00025, 0.0001]
        # Interval (in steps) at which we append to X_hist for logging
        self.nnn = 5
        
        # Load data from file, process it, and build the neural network
        self.load_data()
        self.process_data()
        self.build_model()

    def load_data(self):
        """
        Load data from the specified MATLAB file and convert to TensorFlow tensors.
        """
        mat_data = scipy.io.loadmat(self.matlab_file)  # Load .mat file
        self.pts = mat_data['nodes']  # Spatial coordinate points

        # Casting the loaded data into TensorFlow float64 for consistency
        self.Jc1 = tf.cast(mat_data['Jc1'], dtype=tf.float64)
        self.Jc2 = tf.cast(mat_data['Jc2'], dtype=tf.float64)
        self.Jc3 = tf.cast(mat_data['Jc3'], dtype=tf.float64)
        self.phi1 = tf.cast(mat_data['Phi1'], dtype=tf.float64)
        self.phi2 = tf.cast(mat_data['Phi2'], dtype=tf.float64)
        self.phi3 = tf.cast(mat_data['Phi3'], dtype=tf.float64)

        # If we have a fourth wavelength, load its corresponding data
        if self.num_wavelengths == 4:
            self.Jc4 = tf.cast(mat_data['Jc4'], dtype=tf.float64)
            self.phi4 = tf.cast(mat_data['Phi4'], dtype=tf.float64)

        # These are presumably the reference solutions for different IC types
        self.Sr = mat_data['Sr_rec']
        self.Srsd = mat_data['Sr_rec_sd']

    def process_data(self):
        """
        Process the spatial data points to compute bounding values in each dimension.
        """
        pts = self.pts
        # bd will store the min and max of x, y, z coordinates
        self.bd = np.array([
            [np.min(pts[:, 0]), np.max(pts[:, 0])],
            [np.min(pts[:, 1]), np.max(pts[:, 1])],
            [np.min(pts[:, 2]), np.max(pts[:, 2])]
        ])

    def build_model(self):
        """
        Build the neural network model using TensorFlow Keras Sequential API.
        """
        # Normalization layer to map input coordinates to [-1, 1] range
        s_layer = tf.keras.layers.Lambda(
            lambda x: 2 * (x - self.bd[:, 0]) / (self.bd[:, 1] - self.bd[:, 0]) - 1.0
        )

        # Create a Sequential model with the specified architecture
        self.model = tf.keras.Sequential([
            s_layer,  # First layer does spatial scaling
            tf.keras.layers.Dense(self.net_size[0], activation='tanh'),
            tf.keras.layers.Dense(self.net_size[1], activation='tanh'),
            tf.keras.layers.Dense(self.net_size[2], activation='tanh'),
            tf.keras.layers.Dense(self.net_size[3], activation='softplus')
        ])

    def fitting_model(self, weight_file=None):
        """
        Initial fitting of the model. Either load weights from file or train from scratch.

        Args:
            weight_file (str, optional): Path to the pretrained weights file. Defaults to None.
        """
        # If a weight file is provided, load it
        if weight_file:
            Sr_m = self.model(self.pts)  # Forward pass (not actually used here)
            self.model.load_weights(weight_file)  # Load existing weights
            print(f"Weights loaded from {weight_file}")
        else:
            # Otherwise, do a simple training pass with two learning rates and epochs
            learning_rates = [0.001, 0.0002]
            epochs = [350, 100]

            # Choose target based on the initial condition type
            target = self.Sr if self.ICtype == 'GB' else self.Srsd

            # Perform training for each phase of learning rate and epoch
            for lr, epoch in zip(learning_rates, epochs):
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='mean_squared_error'
                )
                self.model.fit(self.pts, target, epochs=epoch, batch_size=32)

            # Uncomment if you want an interactive save prompt after initial fitting
            # self.save_weights_interactive()

    def compute_bl_residual(self):
        """
        Compute the boundary loss (bl) by comparing model outputs
        with boundary conditions (JcX - phi).

        Returns:
            float: The boundary loss value.
        """
        # Compute the predicted solution at all points
        X_pos = self.model(self.pts)
        X_pos_out = self.model(self.pts_out)

        # Start boundary loss at 0
        bl = 0
        # Add squared error between Jc1*X and phi1
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc1, X_pos) - self.phi1))
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc2, X_pos) - self.phi2))
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc3, X_pos) - self.phi3))

        # If we have a fourth wavelength, include its boundary loss
        if self.num_wavelengths == 4:
            bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc4, X_pos) - self.phi4))

        # Return boundary loss as a Python float
        return bl.numpy()

    def compute_reg_term(self, reg_type, p=None):
        """
        Compute current regularization term for both Lp and H1.

        Args:
            reg_type (str): 'h1' or 'lp'
            p (float, optional): The Lp exponent, if reg_type is 'lp'.

        Returns:
            float: The computed regularization term.
        """
        # H1 regularization uses the gradient norm
        if reg_type == 'h1':
            pts_tensor = tf.convert_to_tensor(self.pts_in, dtype=tf.float64)
            with tf.GradientTape() as tape:
                tape.watch(pts_tensor)
                X_pos_grad = self.model(pts_tensor)
            grad_X_pos = tape.gradient(X_pos_grad, pts_tensor)
            L2_norm_grad = tf.reduce_mean(tf.square(grad_X_pos))
            # Return sqrt of the mean of squared gradients
            return tf.sqrt(L2_norm_grad).numpy()

        # Lp regularization depends on the exponent p
        elif reg_type == 'lp':
            if p == 0:
                return 0.0  # No regularization if p = 0
            X_pos = self.model(self.pts)
            return (tf.reduce_mean(X_pos**p) ** (1/p)).numpy()
        else:
            # Invalid regularization type
            raise ValueError(f"Unknown regularization type: {reg_type}")

    def train_shynn(self):
        """
        Enhanced training routine with adaptive Cr for both Lp and H1 regularizations.
        This function manages the multi-phase training strategy.
        """
        # Identify the region of interest
        self.find_ROI(2.5)

        # Get initial predictions
        Sr_m = self.model(self.pts)
        # Initialize X_hist with the model's current predictions (flattened)
        self.X_hist = [Sr_m.numpy().reshape(-1)]
        
        # Parse the regularization parameters to get the training phases
        reg_sequence = self._parse_regularization()
        
        # Loop over each phase of regularization
        for reg_type, p_val, _ in reg_sequence:
            print(f"\n=== Starting {reg_type.upper()} phase ===")
            
            # Initialize Cr (regularization coefficient) if it's an Lp with p!=0 or H1
            if reg_type == 'lp' and p_val == 0:
                # If p == 0, there's no regularization term
                cr = 0.0
                adjust_cr = False
            else:
                # Compute boundary loss and current reg to initialize Cr
                current_bl = self.compute_bl_residual()
                current_reg = self.compute_reg_term(reg_type, p_val)
                # A small factor to avoid dividing by zero, typically 0.25 here
                cr = (current_bl / (current_reg + 1e-6)) * 0.25
                adjust_cr = True  # We'll adjust Cr dynamically in training
                next_adjust = 200  # Next step count at which we adjust Cr
                adjust_count = 1   # Keep track of how many times we've adjusted Cr so far

            phase_steps = 0  # Counter for steps in the current phase

            # We have potentially multiple sub-phases determined by self.a
            for j in range(len(self.a)):
                # Set up an optimizer with the j-th learning rate
                optimizer = tf.optimizers.Adam(learning_rate=self.lr[j])
                print(f"LR Phase {j+1}: {self.lr[j]:.5f}, Initial Cr: {cr:.2f}, p: {p_val}")
                
                # Run gradient updates for a[j] steps
                for step in range(self.a[j]):
                    with tf.GradientTape() as tape:
                        # Choose which loss function to use (H1 or Lp)
                        if reg_type == 'h1':
                            loss = self.loss_function_h1(cr)
                        else:
                            loss = self.loss_function(p_val, cr)
                    
                    # Compute gradients and update the model
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Log progress at intervals
                    if step % self.nnn == 0:
                        self.X_hist.append(self.model(self.pts).numpy().reshape(-1))
                        print(f"Step {phase_steps}: Loss = {loss.numpy():.4f}, Cr = {cr:.2f}")
                    
                    # Increment the overall phase step counter
                    phase_steps += 1
                    
                    # Check if we should adjust Cr dynamically at this step
                    if adjust_cr and phase_steps >= next_adjust:
                        current_bl = self.compute_bl_residual()
                        current_reg = self.compute_reg_term(reg_type, p_val)
                        factor = 0.75 if adjust_count > 0 else 0.25
                        cr = (current_bl / (current_reg + 1e-6)) * factor
                        
                        print(f"Cr updated to {cr:.2f} at step {phase_steps}")
                        next_adjust += 500  # Next time to adjust Cr
                        adjust_count += 1

        # Optionally save the training history
        # self.save_hist_interactive()

    def _parse_regularization(self):
        """
        Parse the user-supplied regularization parameters (p).
        Determines if we have an H1 phase, multiple Lp phases, or a default sequence.

        Returns:
            list: A list of tuples (reg_type, p_val, None) describing each phase.
        """
        reg_sequence = []
        
        # If p is a list and not empty
        if isinstance(self.p, (list, tuple)) and len(self.p) > 0:
            # If the first element is "h1", we do an H1 phase first
            if self.p[0] == "h1":
                reg_sequence.append(('h1', None, None))  # H1 with dynamic Cr
                # Then for each following value, do Lp if it's numeric
                for val in self.p[1:]:
                    if isinstance(val, (int, float)):
                        if val == 0:
                            reg_sequence.append(('lp', 0.0, 0.0))  # Lp with p=0
                        else:
                            reg_sequence.append(('lp', val, None))  # Lp with p=val
            else:
                # No H1 phase, just interpret each value as an Lp exponent
                for val in self.p:
                    if isinstance(val, (int, float)):
                        if val == 0:
                            reg_sequence.append(('lp', 0.0, 0.0))  # p=0
                        else:
                            reg_sequence.append(('lp', val, None))
        else:
            # If p isn't specified, use a default list of Lp exponents
            default_p = [2.0, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01, 0.006, 0.003, 0.001]
            for val in default_p:
                reg_sequence.append(('lp', val, None))
                
        return reg_sequence

    def loss_function(self, p, cr):
        """
        Lp regularization loss function (boundary term + Cr * Lp term).

        Args:
            p (float): The Lp exponent.
            cr (float): The current regularization coefficient.

        Returns:
            tf.Tensor: The computed loss (scalar).
        """
        # Predictions inside and outside the ROI
        X_pos = self.model(self.pts)
        X_pos_out = self.model(self.pts_out)
        
        # Start boundary loss with a factor on max outside the ROI
        bl = 1000 * tf.reduce_max(X_pos_out)
        # Add squared errors for boundary constraints
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc1, X_pos) - self.phi1))
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc2, X_pos) - self.phi2))
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc3, X_pos) - self.phi3))

        # If a fourth wavelength is present, include that boundary term
        if self.num_wavelengths == 4:
            bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc4, X_pos) - self.phi4))
            
        # Lp regularization
        reg = 0.0 if p == 0 else (tf.reduce_mean(X_pos**p) ** (1/p))

        # Return total loss = boundary loss + cr * regularization
        return bl + cr * reg

    def loss_function_h1(self, cr):
        """
        H1 regularization loss function (boundary term + Cr * H1 term).

        Args:
            cr (float): The current regularization coefficient.

        Returns:
            tf.Tensor: The computed loss (scalar).
        """
        # Predictions inside and outside the ROI
        X_pos = self.model(self.pts)
        X_pos_out = self.model(self.pts_out)
        pts_tensor = tf.convert_to_tensor(self.pts_in, dtype=tf.float64)

        # Calculate gradient with respect to pts_in
        with tf.GradientTape() as tape:
            tape.watch(pts_tensor)
            X_pos_grad = self.model(pts_tensor)
        grad_X_pos = tape.gradient(X_pos_grad, pts_tensor)

        # Compute average of squared gradient (L2 norm)
        L2_norm_grad = tf.reduce_mean(tf.square(grad_X_pos))

        # Start boundary loss with a factor on max outside the ROI
        bl = 1000 * tf.reduce_max(X_pos_out)
        # Add boundary condition constraints (squared differences)
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc1, X_pos) - self.phi1))
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc2, X_pos) - self.phi2))
        bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc3, X_pos) - self.phi3))
        if self.num_wavelengths == 4:
            bl += tf.reduce_mean(tf.square(tf.matmul(self.Jc4, X_pos) - self.phi4))
            
        # Return total loss = boundary loss + cr * sqrt(mean of squared gradients)
        return bl + cr * tf.sqrt(L2_norm_grad)

    def find_ROI(self, croi):
        """
        Identify region of interest (ROI) by thresholding the predictions
        and checking distances from the center of the region.

        Args:
            croi (float): A multiplier that defines the cutoff for ROI vs out-of-ROI.
        """
        # Identify points where Sr is above 30% of its maximum
        ind1 = self.Sr > np.max(self.Sr) * 0.3
        ind1 = tf.reshape(ind1, [-1])
        pts1 = self.pts[ind1]
        
        # Compute the "center" of the region by taking the mean of the selected points
        Sr_center = np.mean(pts1, axis=0)
        
        # Determine the maximum distance from this center
        max_dist = np.max(np.sqrt(np.sum((pts1 - Sr_center)**2, axis=1)))
        
        # Compute distances for all points in the domain
        distances = np.sqrt(np.sum((self.pts - Sr_center)**2, axis=1))
        
        # Separate points into "in ROI" and "out of ROI"
        self.ind2 = np.where(distances > croi * max_dist)[0]  # Out of ROI
        self.ind3 = np.where(distances <= croi * max_dist)[0] # In ROI
        self.pts_in = self.pts[self.ind3]
        self.pts_out = self.pts[self.ind2]

    def save_weights_interactive(self, filename=None):
        """
        Interactive function to save the model weights. 
        Asks the user for confirmation and filename if not provided.

        Args:
            filename (str, optional): File name to save the weights. Defaults to None.
        """
        while True:
            if not filename:
                save_flag = input("\nSave current weights? [y/n]: ").lower()
                if save_flag != 'y':
                    print("Weight saving cancelled.")
                    return
                
                filename = input("Enter weight file name: ").strip()
                if not filename:
                    print("Invalid empty name! Try again.")
                    continue

            if not filename.endswith('.h5'):
                filename += '.h5'
            
            try:
                self.model.save_weights(filename)
                print(f"Weights saved to {filename}")
                return
            except Exception as e:
                print(f"Save error: {str(e)}")
                retry = input("Retry? [y/n]: ").lower()
                if retry != 'y':
                    return
                filename = None

    def save_hist_interactive(self, filename=None):
        """
        Save training history (model outputs over training steps).
        Interactively prompts the user, unless a filename is provided.

        Args:
            filename (str, optional): File name for the .mat file. Defaults to None.
        """
        while True:
            if not filename:
                save_flag = input("\nSave training history? [y/n]: ").lower()
                if save_flag != 'y':
                    print("History saving cancelled.")
                    return
                
                filename = input("Enter history file name: ").strip()
                if not filename:
                    print("Invalid empty name! Try again.")
                    continue

            if not filename.endswith('.mat'):
                filename += '.mat'
            
            try:
                X_hist_np = np.stack(self.X_hist)
                savemat(filename, {
                    'X_hist': X_hist_np,
                    'nodes': self.pts
                })
                print(f"Training history saved to {filename}")
                return
            except Exception as e:
                print(f"Save error: {str(e)}")
                retry = input("Retry? [y/n]: ").lower()
                if retry != 'y':
                    return
                filename = None

    def plot_results(self, mode, croi, el, az):
        """
        Visualize model predictions in 3D. 
        Optionally plots only within the region of interest (ROI).

        Args:
            mode (str): 'roi' or 'full' to plot. 
            croi (float): Radius multiplier for ROI.
            el (float): Elevation for 3D plot.
            az (float): Azimuth angle for 3D plot.
        """
        if mode == "roi":
            # Find ROI and plot only those points
            self.find_ROI(croi)
            pts = self.pts_in
            Sr_m = self.model(pts)
            title = "ROI Results"
        else:
            # Plot the entire domain
            pts = self.pts
            Sr_m = self.model(pts)
            title = "Full Domain Results"

        # Set up a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the points, colored by their predicted values
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=Sr_m, cmap='hot', s=50, alpha=0.6)
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(title)
        
        # Add a color bar
        fig.colorbar(sc, shrink=0.5, aspect=5)
        plt.show()

    def plot_fittingerror(self, mode, croi, el, az):
        """
        Plot fitting error (difference between model and reference) in 3D.

        Args:
            mode (str): 'roi' or 'whole' to plot.
            croi (float): ROI radius multiplier.
            el (float): Elevation for 3D plot.
            az (float): Azimuth angle for 3D plot.

        Raises:
            ValueError: If mode is invalid.
        """
        if mode == "roi":
            # Plot error only inside ROI
            self.find_ROI(croi)
            Sr_m = self.model(self.pts_in)
            error = Sr_m - self.Sr[self.ind3]
            title = "ROI Fitting Error"
            pts = self.pts_in
        elif mode == "whole":
            # Plot error over the entire domain
            Sr_m = self.model(self.pts)
            error = Sr_m - self.Sr
            title = "Full Domain Fitting Error"
            pts = self.pts
        else:
            # Invalid mode argument
            raise ValueError("Invalid mode. Use 'roi' or 'whole'.")

        # Create a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter the points colored by the error
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=error, cmap='coolwarm', s=50, alpha=0.6)
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(title)
        
        # Add a color bar
        fig.colorbar(sc, shrink=0.5, aspect=5)
        plt.show()

    def save_output(self, filename='results.mat'):
        """
        Save final results including the model's final predictions, nodes, parameters, and training history.

        Args:
            filename (str, optional): Name of the .mat file. Defaults to 'results.mat'.
        """
        # Model predictions over all points
        Sr_net = self.model(self.pts)
        # Save data to a .mat file
        savemat(filename, {
            'prediction': Sr_net.numpy(),
            'nodes': self.pts,
            'parameters': {'a': self.a},
            'X_hist': np.stack(self.X_hist)
        })

# Check if a GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU acceleration enabled")
else:
    print("Running on CPU")
