import numpy as np
import matplotlib.pyplot as plt
import random
import os

class Simulator:
    def __init__(self):
        self.gw = 12500/256  # gate width in ps
        # try:
        #     # self.noise = np.load('.data/noise_micro.npy')
        #     self.pIRF = np.load('.data/hbifli/microscopy/irf_micro.npy')
        # except FileNotFoundError:
        #     try:
        #         self.noise = np.load('noise_micro.npy')
        #     except FileNotFoundError:
        #         self.noise = None
        #     self.pIRF = np.load('irf_micro.npy')
        self.pIRF = np.load('data/hbifli/microscopy/irf_micro.npy') # the shape of IRF is (x,y,t)
        self.n_time_points = self.pIRF.shape[2]
        self.img_size_full = (self.pIRF.shape[0], self.pIRF.shape[1])
        self.intens = np.random.randint(low=0, high=245, size=(self.pIRF.shape[0], self.pIRF.shape[1]))
        # self.intens = np.random.randint(low=0, high=245, size=(self.pIRF.shape[0], self.pIRF.shape[1])) + 10

    def __call__(self, params):
        # Convert parameters to numpy arrays.
        batch_tau_L, batch_tau_L_2, batch_A_L = params['tau_L'], params['tau_L_2'], params['A_L']

        sims = []
        for tau_L, tau_L_2, A_L in zip(batch_tau_L, batch_tau_L_2, batch_A_L):
            F_dec_conv = self.decay_gen_single(tau_L=tau_L, tau_L_2=tau_L_2, A_L=A_L)
            sims.append(F_dec_conv)

        F_dec_conv = np.stack(sims)
        return dict(observable=F_dec_conv)

    # def _sample_noise(self, data, scale=1):
    #     if self.noise is None:
    #         # sample intensity
    #         img = np.random.randint(0, high=25)
    #         noisy_data = np.round(np.random.poisson(data * img))
    #     else:
    #         # use recorded noise
    #         i = np.random.choice(self.noise.shape[0])
    #         j = np.random.choice(self.noise.shape[1])
    #         noisy_data = data + scale * self.noise[i, j]
    #     return noisy_data
    
    

    def decay_gen_single(self, tau_L, tau_L_2, A_L):
        intensity  = np.squeeze(self._random_crop(self.intens, crop_size=(1, 1))); # generating the pixel intensity to avoid the zero intensity
        cropped_pIRF = self._random_crop(self.pIRF, crop_size=(1, 1))
        # cropped_pIRF = cropped_pIRF / np.sum(cropped_pIRF)
        a1, b1, c1 = np.shape(cropped_pIRF)
        t = np.linspace(0, c1 * (self.gw * (10 **-3)), c1)
        A = A_L * np.exp(-t / tau_L)
        B = (1-A_L) * np.exp(-t / tau_L_2)
        dec = A + B
        irf_out = cropped_pIRF[0,0]
        irf_out = irf_out / np.sum(irf_out)
        dec_conv = self._conv_dec(dec, irf_out) * intensity

        # add noise
        dec_conv = [round(x) for x in dec_conv]
        dec_conv = np.random.poisson(lam=dec_conv)
        # dec_conv = self._sample_noise(dec_conv)

        # truncated from below
        dec_conv = self._jitter(dec_conv)
        # dec_conv = np.maximum(dec_conv, 0)

        # scale output to 1
        # dec_conv = self._norm1D(dec_conv)
        return dec_conv

    @staticmethod
    def _random_crop(array, crop_size):
        if array.ndim not in [2, 3]:
            raise ValueError("Input array must be 2D or 3D")

        A, B = array.shape[:2]
        a, b = crop_size

        if a > A or b > B:
            raise ValueError("Crop size must be smaller than or equal to the original array size")

        # Randomly select the top-left corner of the crop
        top = np.random.randint(0, A - a + 1)
        left = np.random.randint(0, B - b + 1)

        # Crop the subarray
        if array.ndim == 2:
            return array[top:top + a, left:left + b]
        else:  # 3D
            return array[top:top + a, left:left + b, :]

    @staticmethod
    def _norm1D(fn):
        if np.amax(fn) == 0:
            nfn = fn
        else:
            nfn = fn / np.amax(fn)
        return nfn

    @staticmethod
    def _conv_dec(dec, irf):
        conv = np.convolve(dec, irf)
        conv = conv[:len(dec)]
        return conv
    
    @staticmethod
    def _jitter(decay):
        num_gate = len(decay)
        r = np.random.rand()
        gate_shift = int(np.random.rand() * 3)
        
        if r > 0.75:
            # No shift
            modified_decay = decay
        elif r < 0.25:
            # Shift right by padding with zeros on the left
            modified_decay = np.concatenate((np.zeros(gate_shift), decay[:num_gate - gate_shift]))
        else:
            # Shift left by removing elements from the left
            modified_decay = decay[gate_shift:]
            # Pad with zeros at the end to maintain the original length
            modified_decay = np.concatenate((modified_decay, np.zeros(gate_shift)))
            
        return modified_decay



# Main testing script
if __name__ == '__main__':
    print(os.getcwd())
    # 1. Instantiate the Simulator class
    try:
        sim = Simulator()
        print("Simulator instance created successfully.")
    except Exception as e:
        print(f"Error creating Simulator instance: {e}")
        exit()

    # 2. Generate random input parameters with the specified shapes and ranges
    # tau_L: [0.1, 0.6], shape [512, 512]
    tau_L = np.random.uniform(0.1, 0.6, size=(10))

    # tau_L_2: [0.6, 1.5], shape [512, 512]
    tau_L_2 = np.random.uniform(0.6, 1.5, size=(10))

    # A_L: [0, 1], shape [512, 512]
    A_L = np.random.uniform(0, 1, size=(10))

    # 3. Create the params dictionary
    params = {
        'tau_L': tau_L,
        'tau_L_2': tau_L_2,
        'A_L': A_L
    }

    print("\nGenerated input parameters with specified ranges and shapes:")
    print(f"tau_L shape: {params['tau_L'].shape}, min: {params['tau_L'].min():.2f}, max: {params['tau_L'].max():.2f}")
    print(f"tau_L_2 shape: {params['tau_L_2'].shape}, min: {params['tau_L_2'].min():.2f}, max: {params['tau_L_2'].max():.2f}")
    print(f"A_L shape: {params['A_L'].shape}, min: {params['A_L'].min():.2f}, max: {params['A_L'].max():.2f}")

    # 4. Call the simulator with the generated parameters
    try:
        print("\nRunning the simulator...")
        result = sim(params)
        print("\nSimulator ran without errors.")
        print(f"\nResult keys: {list(result.keys())}")

        # 5. Check the output
        observable_data = result['observable']
        print("\nSimulator ran successfully.")
        print(f"Output 'observable' shape: {observable_data.shape}")

        # The output of decay_gen_single is a 1D array of length self.n_time_points.
        # The __call__ method stacks these, so the final shape should be
        # (512*512, number_of_time_points).
        expected_shape = (512 * 512, sim.n_time_points)
        print(f"Expected output shape: {expected_shape}")

        if observable_data.shape == expected_shape:
            print("✅ Test Passed: The output shape is correct.")
        else:
            print(f"❌ Test Failed: Expected shape {expected_shape}, but got {observable_data.shape}.")

        # Additional check for data validity (e.g., values are between 0 and 1)
        if np.all(observable_data >= 0) and np.all(observable_data <= 1):
            print("✅ Test Passed: All values are within the [0, 1] range as expected from _norm1D.")
        else:
            print("❌ Test Failed: Some output values are outside the [0, 1] range.")

    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

    # 1. Access the simulated data
    num_pixels, n_time_points = observable_data.shape

    # 2. Randomly select 4 indices (pixels)
    # Use random.sample to select unique indices
    random_indices = random.sample(range(num_pixels), 4)

    # 3. Create a time axis for plotting
    # This assumes the time axis logic from your Simulator's decay_gen_single method.
    # Since the time points are not explicitly returned, we'll re-create a representative one.
    # The simulator uses gw = 12500/256 and n_time_points from the IRF.
    gw = 12500 / 256  # Gate width in ps
    t = np.linspace(0, (n_time_points - 1) * (gw * (10 ** -3)), n_time_points)


    # 4. Plot the results
    plt.figure(figsize=(10, 6))
    for i, index in enumerate(random_indices):
        # Plot the decay curve for the randomly selected pixel
        plt.plot(t, observable_data[index, :], label=f'Pixel {index + 1}')

    plt.title('Simulated Decay Curves for 4 Randomly Selected Pixels')
    plt.xlabel('Time (ns)')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle('Simulated Decay Curves for 4 Randomly Selected Pixels', fontsize=16)

    for ax, index in zip(axes.flat, random_indices):
        ax.plot(t, observable_data[index, :])
        ax.set_title(f'Pixel {index + 1}')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Normalized Intensity')
        ax.grid(True)

    plt.show()
