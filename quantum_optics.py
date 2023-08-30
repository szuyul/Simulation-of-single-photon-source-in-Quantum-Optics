import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# reference to Alain Aspect's paper: https://www.researchgate.net/publication/337945267_The_First_Single_Photon_Sources_and_Single_Photon_Interference_Experiments


def w2f_converter(wavelength):
    c = 3e8
    omega = 2*math.pi*(c/wavelength)
    return omega

def photon_counting_exp(light_source, duration, N_realizations):
    photon_counts = np.zeros((N_realizations,))
    for i in range(N_realizations):
        _, sequence = light_source.emission_trace(duration)
        photon_counts[i] = photodetector.photon_counting(sequence)

    return photon_counts, np.mean(photon_counts), np.std(photon_counts)

def anti_correlation_exp(light_source, duration):
    BS = beamsplitter(T = 0.5, R = 0.5)
    vacuum_source = vacuum()
    _, sequence = light_source.emission_trace(duration)
    _, sequence_vacuum = vacuum_source.emission_trace(duration, light_source.RR)
    N_pulse = duration * light_source.RR # number of excitation pulses
    photon_number_stat = [ (i,sequence.count(i)) for i in set(sequence) ] # statistics of emitted photons
    photon_number_stat.sort(key=lambda x: x[0])

    output_1, output_2 = BS.split_light(input_1=sequence, input_2=sequence_vacuum)
    coincidence_sequence = np.array([output_1[i]*output_2[i] for i in range(len(output_1))])
    N_output_1 = photodetector.photon_counting(output_1)
    N_output_2 = photodetector.photon_counting(output_2)
    N_c = photodetector.photon_counting(coincidence_sequence)

    P_output_1 = N_output_1/N_pulse
    P_output_2 = N_output_2/N_pulse
    P_c = N_c/N_pulse

    print('(N_pulse, N_output_1, N_output_2, N_coincidence) = (' + str(N_pulse) + ', '  + str(N_output_1) + ', ' + str(N_output_2) + ', ' + str(N_c) + ')')
    print('P_c / (P_o1 x P_o2) = ' + str(P_c/(P_output_1*P_output_2)))
    return (N_pulse, photon_number_stat, N_output_1, N_output_1, N_c)

def single_photon_interference_exp(light_source, duration, phase_delays):
    MZ = MZ_interferometer(T = 0.5, R = 0.5)
    _, sequence = light_source.emission_trace(duration)
    
    N_output_1 = []
    N_output_2 = []
    for i in range(len(phase_delays)):
        output_1, output_2 = MZ.interference(sequence, phase_delays[i])
        N_output_1.append( photodetector.photon_counting(output_1) )
        N_output_2.append( photodetector.photon_counting(output_2) )

    return (N_output_1, N_output_2)


class light_source:
    # generic pulsed light source
    @staticmethod
    def single_photon_energy(w):
        h = 6.626e-34 # Planck's constant, unit: m^2 kg s^-1
        return h*(w/(2*math.pi))

    @staticmethod
    def single_wavepacket_profile(w_center, bw, S_window, T_window, N=1000, spectrum_shape='Lorentzian', to_plot=False):
        # bw is fwhm of the spectrum
        # N = int(S_window*T_window/(2*math.pi))

        if spectrum_shape == 'Lorentzian':
            gamma = bw
            w = np.linspace(w_center - S_window/2, w_center + S_window/2, N)
            Spectrum = 1. / (np.power(w - w_center, 2) + (gamma/2)**2)

            t = np.linspace(-0.5*T_window, 0.5*T_window, N)
            Wavepacket = np.heaviside(t, 1)*np.exp(-gamma*t)
        else: # Gaussian lineshape
            sigma = bw/2.355
            w = np.linspace(w_center - S_window/2, w_center + S_window/2, N)
            Spectrum = np.exp(-np.power( (w - w_center)/(np.sqrt(2)*sigma), 2))

            t = np.linspace(-0.5*T_window, 0.5*T_window, N)
            Wavepacket = np.exp(-np.power( np.pi*np.sqrt(2)*sigma*t, 2))

        if to_plot:
            _, ax = plt.subplots(1, 2)
            ax[0].set_title(spectrum_shape + ' Spectrum')
            ax[0].plot(((w-w_center)*1e-9)/(2*math.pi), Spectrum/np.max(Spectrum))
            ax[0].set_xlabel('offset freq. (GHz)')
            ax[0].set_ylabel('normalized intensity')
            ax[1].set_title('Wavepacket profile')
            ax[1].plot(t*1e9, Wavepacket)
            ax[1].set_xlabel('time (ns)')
        return (w, Spectrum, t, Wavepacket)

    def emission_trace(RR, duration, emit_generator):
        N_pulse = int(RR*duration) # total number of excitation pulses within the duration

        sequence = list(emit_generator(N_pulse)) # the sequence of emitted photons per pulse
        return (N_pulse, sequence)

class single_photon_light(light_source):
    # only one |n> quantum state, assume spontaneous parametric down-conversion: ground, excited, and intermediate levels
    def __init__(self, t_e, t_i, w_h, w_o, N_mean, RR, spectrum_shape):
        self.t_e = t_e # lifetime of excited state
        self.t_i = t_i # lifetime of intermediate state
        self.w_h = w_h # freq. of heralding photon
        self.w = w_o # freq. of heralded photon
        self.RR = RR
        self.spectrum_shape = spectrum_shape

        # quantum properties per pulse:
        self.N_mean = N_mean # < N >

    @property
    def bw(self):
        return 1/self.t_i

    @property
    def pulse_energy(self):
        return self.single_photon_energy(self.w) # average pulse energy
        
    @property # quantum properties per pulse
    def N_square(self):
        return self.N_mean^2 # < N^2 >
    
    @property # quantum properties per pulse
    def N_var(self):
        return 0 # Var[N] = < N^2 > - < N >^2 = 0

    def emit(self, N):
        for i in range(N):
            yield self.N_mean

    def emission_trace(self, duration):
        return light_source.emission_trace(self.RR, duration, self.emit)
       
class classical_light(light_source):
    # sum of many |n> quantum states, e.g., LED, laser, light bulb...
    def __init__(self, tp, w, I, RR, spectrum_shape):
        self.tp = tp
        self.w = w
        self.I = I
        self.RR = RR
        self.spectrum_shape = spectrum_shape

    @property
    def bw(self):
        return 0.441/self.tp

    @property
    def pulse_energy(self):
        return self.I/self.RR # average pulse energy

    @property # quantum properties per pulse
    def N_mean(self):
        return self.pulse_energy/self.single_photon_energy(self.w) # < N >
        
    @property # quantum properties per pulse
    def N_square(self):
        return self.N_mean + self.N_mean**2 # < N^2 >
    
    @property # quantum properties per pulse
    def N_var(self):
        return self.N_mean # Var[N] = < N^2 > - < N >^2

    def emit(self, N):
        for i in range(N):
            yield int(np.random.poisson(self.N_mean, 1)) # Poisson distribution (discrete values: 0, 1, 2, ...)

    def emission_trace(self, duration):
        return light_source.emission_trace(self.RR, duration, self.emit)

class vacuum(light_source):
    def __init__(self):
    # quantum properties per pulse:
        self.N_mean = 0 # < N >
        self.N_square = self.N_mean^2 # < N^2 >
        self.N_var = 0 # Var[N] = < N^2 > - < N >^2 = 0

    def emit(self, N):
        for i in range(N):
            yield self.N_mean

    def emission_trace(self, duration, RR):
        return light_source.emission_trace(RR, duration, self.emit)

class photodetector:
    @staticmethod
    def photon_counting(sequence):
        return int(sum(sequence))
    
    def __init__(self, duration, sampling_rate, eta):
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.eta = eta # detection efficiency

    def detect(self, time_lapse, light_source, detection_type='direct'):
        t_steps = int(self.duration * self.sampling_rate)
        _, _, _, wp = light_source.single_wavepacket_profile(light_source.w, light_source.bw, 10*light_source.bw, 
                                                             self.duration, N=t_steps, 
                                                             spectrum_shape=light_source.spectrum_shape)
        detection_window = np.zeros((t_steps,))

        if detection_type == 'direct':
            iter = int(self.eta * time_lapse / self.duration)
            for i in range(iter):
                detection_window = detection_window + self.direct_detection(light_source, wp) # integrate over times
        else: # heralded detection
            iter = int(self.eta * time_lapse * light_source.RR)
            for i in range(iter):
                detection_window = detection_window + self.heralded_detection(light_source, wp) # integrate over times

        return detection_window

    def direct_detection(self, light_source, wp):
        if type(light_source).__name__ == 'single_emitter':
            N_pulse = np.floor(self.duration * light_source.RR) + np.random.binomial(1, light_source.RR*(self.duration % (1/light_source.RR)), 1) # random number of excitation pulses
            N_emit = N_pulse * light_source.N_mean
        elif type(light_source).__name__ == 'classical_light':
            N_emit = np.random.poisson(self.duration * light_source.RR * light_source.N_mean, 1) # how many photons within this observation window, self.T_window
        delays = self.duration*(np.random.rand( int(N_emit) )-0.5)

        t_steps = len(wp)
        detect_trace = np.zeros(wp.shape)
        for delay in delays:
            detect_trace += np.roll(wp, int(delay*t_steps/self.duration)) # superposition of all randomly delayed photons 
        return detect_trace

    def heralded_detection(self, light_source, wp):
        # heralded photon delays
        delay_heralded = [light_source.t_i]

        # scattered photon delays
        # stray w_o photon from another atom than the heralding atom. See Eq. 1.30 N_tau_gate in Alain's paper (N_stray = N_tau_gate).
        N_stray = int(np.random.poisson(2e1 * light_source.RR * self.duration, 1))
        delays_stray = self.duration * (np.random.rand( N_stray )-0.5)

        delays = delay_heralded + [i for i in delays_stray]
        t_steps = len(wp)
        detect_trace = np.zeros(wp.shape)
        for delay in delays:
            detect_trace += np.roll(wp, int(delay*t_steps/self.duration))
        return detect_trace

class beamsplitter:
    def __init__(self, T=0.5, R=0.5):
        # 3 = r*1 + t*2
        # 4 = t*1 - r*2
        self.T = T
        self.R = R
        self.t = math.sqrt(self.T)
        self.r = math.sqrt(self.R)

    def split_light(self, input_1, input_2, phase_delay=0):
        # input_1 and input_2 are lists of photon number sequence in the same mode (k, w)
        # theta is the phase delay in rad of input_2 relative to input_1
        N_t = len(input_1) # number of input time steps
        # extract only non-zero photon inputs
        input_nonzero = [(input_1[i], input_2[i]) for i in range(N_t) if (input_1[i] + input_2[i]) > 0]
        input_1_nonzero = [in_1 for in_1, in_2 in input_nonzero]
        input_2_nonzero = [in_2 for in_1, in_2 in input_nonzero]
        N_t_reduced = len(input_nonzero) # number of time steps where input is not zero
        
        # probability to output 3 considering interference = ( N1*R + N2*T + sqrt(N1*N2)*r*t*cos(theta) )/( N1 + N2 )
        output_1 = [np.random.binomial(input_1_nonzero[i] + input_2_nonzero[i], 
                    (input_1_nonzero[i]*self.R + input_2_nonzero[i]*self.T + math.sqrt(input_1_nonzero[i]*input_2_nonzero[i])*self.r*self.t*math.cos(phase_delay))/(input_1_nonzero[i] + input_2_nonzero[i]), 1) 
                    for i in range(N_t_reduced)]
        
        # assume a lossless beamsplitter, output 4 = all input - output 3
        output_2 = [input_1_nonzero[i] + input_2_nonzero[i] - output_1[i] for i in range(N_t_reduced)] 

        return (output_1, output_2)

class MZ_interferometer:
    def __init__(self, T=0.5, R=0.5):
        # out_1 = r*3 + t*4 = r*(r*1 + t*2) + t*exp(jphi)*(t*1 - r*2)
        # out_2 = t*3 - r*4 = t*(r*1 + t*2) - r*exp(jphi)*(t*1 - r*2)
        self.T = T
        self.R = R
        self.t = math.sqrt(self.T)
        self.r = math.sqrt(self.R)

    def interference(self, input, phase_delay=0):
        # consider only one input: 
        # out_1 = r*3 + t*4 = r*r*1 + t*exp(jphi)*t*1 = (R + T*exp(jphi))*1
        # out_2 = t*3 - r*4 = t*r*1 - r*exp(jphi)*t*1 = t*r(1 - exp(jphi))*1
        N_t = len(input)
        output_1 = [np.random.binomial(input[i], self.R**2 + self.T**2 + 2*self.R*self.T*math.cos(phase_delay), 1) for i in range(N_t)]
        output_2 = [input[i] - output_1[i] for i in range(N_t)]

        return output_1, output_2


#%% set up the light sources
# radiative cascade in calcium atoms
t_e = 7.3e-9 # 4p2 state lifetime, unit: s, ref: https://agjsr.agu.edu.bh/uploads/images/papers/pdfs/9cb25c3a693d9bd1a6644bc69e5e0d29_60db0e30d2fb5.pdf
t_i = 4.7e-9 # 4s4p1 state lifetime, unit: s
lambda_h = 5.513e-7 # heralding wavelength, unit: m
lambda_o = 4.227e-7
w_h = w2f_converter(lambda_h)
w_o = w2f_converter(lambda_o)
RR_s = 2e4 # repetition rate, unit: Hz
Ca_atom = single_photon_light(t_e, t_i, w_h, w_o, 1, RR_s, 'Lorentzian')
Ca_atom.single_wavepacket_profile(Ca_atom.w, Ca_atom.bw, 10*(t_i**(-1)), 20*t_i, N=1000, spectrum_shape=Ca_atom.spectrum_shape, to_plot=True)


# attenuated pulses from classical light source
lambda_c = 5.32e-7 # unit: m
tp_c = 5e-9 # pulse duration, unit: s
w_c = w2f_converter(lambda_c)
I_c = RR_s*light_source.single_photon_energy(w_c) # intensity, unit: W
RR_c = 1e3 # repetition rate, unit: Hz
led = classical_light(tp_c, w_c, I_c, RR_c, 'Gaussian')
led.single_wavepacket_profile(led.w, led.bw, 10*(tp_c**(-1)), 20*tp_c, N=1000, spectrum_shape=led.spectrum_shape, to_plot=True)


#%% show the emission traces
duration = 5e-2 # unit: s

N_pulse_Ca_atom, emit_sequence_Ca_atom = Ca_atom.emission_trace(duration)
N_pulse_led, emit_sequence_led = led.emission_trace(duration)

_, ax1 = plt.subplots(1, 2)
ax1[0].plot(np.linspace(0, duration, len(emit_sequence_Ca_atom)), emit_sequence_Ca_atom)
ax1[0].set_title('Calcium single emitter')
ax1[0].set_xlabel('time (s)')
ax1[0].set_ylim((0, max(emit_sequence_led)))
ax1[1].plot(np.linspace(0, duration, len(emit_sequence_led)), emit_sequence_led)
ax1[1].set_title('LED')
ax1[1].set_xlabel('time (s)')
ax1[1].set_ylim((0, max(emit_sequence_led)))

# statistics on repeated photon counting
N_realizations = 100
Ca_atom_photon_counts, _, _ = photon_counting_exp(Ca_atom, duration, N_realizations)
led_photon_counts, _, _ = photon_counting_exp(led, duration, N_realizations)

_, ax2 = plt.subplots(1, 2)
ax2[0].hist(Ca_atom_photon_counts)
ax2[0].set_title('Calcium single emitter')
ax2[0].set_xlabel('counts')
ax2[0].set_ylabel('realizations (total ' + str(N_realizations) + ')')
ax2[1].hist(led_photon_counts)
ax2[1].set_title('LED')
ax2[1].set_xlabel('counts')


#%% SNR of photon numbers
duration = 1e-1 # unit: s
I_cs = [i*RR_s*light_source.single_photon_energy(w_c) for i in np.power(10, np.linspace(0, 3, 10))]
iter = len(I_cs)
N_mean = np.zeros((iter,))
N_std = np.zeros((iter,))
SNR = np.zeros((iter,))

for i in range(iter):
    led.I = I_cs[i] # increasing the led power
    _, N_mean[i], N_std[i] = photon_counting_exp(led, duration, N_realizations=100)
    SNR[i] = N_mean[i]/N_std[i]

_, ax3 = plt.subplots(1,)
ax3.plot(N_mean, SNR)
ax3.set_title('photon count SNR')
ax3.set_xlabel('average photon number')
ax3.set_ylabel('SNR')
ax3.grid()


#%% single-photon pulse detection
duration = 1e3*t_i # tau_gate
sampling_rate = 1e9
t_steps = int(duration * sampling_rate)
eta = 1e-4 # detection efficiency
led.I = RR_s*light_source.single_photon_energy(w_c) # intensity, unit: W
detector = photodetector(duration, sampling_rate, eta)

time_lapse = 1200 # observation time, unit: s
detection_window_Ca_atom = detector.detect(time_lapse, Ca_atom, detection_type='heralded')
detection_window_led = detector.detect(time_lapse, led, detection_type='direct')

_, ax4 = plt.subplots(1, 2)
ax4[0].plot(np.linspace(0, duration, t_steps), detection_window_Ca_atom)
ax4[0].set_title('Calcium single emitter (heralded detection)')
ax4[0].set_xlabel('t (s)')
ax4[0].set_ylabel('intensity (a.u.)')
ax4[1].plot(np.linspace(0, duration, t_steps), detection_window_led)
ax4[1].set_title('Classical light (direct detection)')
ax4[1].set_xlabel('t (s)')

#%% coincidence detection
duration = 10 # unit: s

anti_correlation_exp(Ca_atom, duration)

led.I = 5e-3*RR_s*light_source.single_photon_energy(w_c) # intensity, unit: W
anti_correlation_exp(led, duration)
# WHY often smaller than 1????

#%% single photon interference
durations = [5*i for i in [1e-5, 1e-4, 1e-3, 1e-2]] # unit: s
phase_delays = [i for i in np.linspace(-5*math.pi, 5*math.pi, 100)]

_, ax5 = plt.subplots(len(durations), 2)
for i in range(len(durations)):
    output_1_per_delay, output_2_per_delay = single_photon_interference_exp(Ca_atom, durations[i], phase_delays)

    ax5[i, 0].scatter(phase_delays, output_1_per_delay)
    ax5[i, 0].set_ylabel('photon counts')
    ax5[i, 1].scatter(phase_delays, output_2_per_delay)
ax5[0, 0].set_title('output 1')
ax5[0, 1].set_title('output 2')
ax5[-1, 0].set_xlabel('phase delay (rad)')
ax5[-1, 1].set_xlabel('phase delay (rad)')

i= 1
