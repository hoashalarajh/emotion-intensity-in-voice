import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import webrtcvad
import pandas as pd
import pickle

# importing standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# storing the prediction scores
prediction_scores = []
prediction_probabs = []

neutral = []
onset = []
offset = []
apex = []

target_names = ["Neutral", "Onset", "Offset", "Apex"]

# defining constants
FRAME_SIZE = 1440
HOP_LENGTH = 720

feature_array = np.array([])

data_path = "E://#ICIIS/#human_studies/"
dir_list = os.listdir(data_path)

print (f"The total number of files is {len(dir_list)}")

model_path = "E://#ICIIS/EmoIntenseRFC_Model#200#_mfcc + fund_freq.sav"

#loading the trained model
def load_model(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

# reading signal
def read_signal(path):
    sp_sig, sample_rate = librosa.load(path, sr = 48000)
    return (sp_sig, sample_rate)

# signal smoothening
def moving_avg(signal, window_size):
    filter = []
    for i in range(((len(signal) - (window_size - 1)))):
        summation = 0
        k = i
        for j in range(window_size):
            summation = summation + signal[k]
            k = k + 1
        avg_val = summation / window_size
        filter.append(avg_val)
    return np.array(filter)


# convert float to pcm audio
def float_to_pcm16(audio):
    import numpy

    ints = (audio * 32767).astype(numpy.int16)
    little_endian = ints.astype('<u2')
    pcm_data = little_endian.tobytes()
    return pcm_data

# voice activity detection and segmenting voiced and unvoiced data
def voice_detect(filtered_speech, sample_rate, pcm_data):
    vad = webrtcvad.Vad()
    vad.set_mode(3)

    k = 0
    voice_data = []
    unvoice_data = []
    temp_list = []
    while k < ((len (filtered_speech) / ((len(pcm_data))/2))- 1):
        #print (k)
        pcm_data = float_to_pcm16(filtered_speech[(k * 1440) : (k+1) * 1440])
        #print (f"Contains speech: {(vad.is_speech(pcm_data, sample_rate))}")
        #print (f"Length of pcm data : {len(pcm_data)}")
        #print (f"Length of filtered_speech data : {len(filtered_speech)}")
        
        if ((vad.is_speech(pcm_data, sample_rate))):
            temp_list = ((filtered_speech[(k * 1440) : (k+1) * 1440]).tolist())
            for i in temp_list:
                voice_data.append(i)
        else:
            temp_list = ((filtered_speech[(k * 1440) : (k+1) * 1440]).tolist())
            for i in temp_list:
                unvoice_data.append(i)

        k = k + 1
    return voice_data, unvoice_data


# calculating mfccs
def mfcc (signal, FRAME_SIZE, HOP_LENGTH, sample_rate):
    mfcc_val = librosa.feature.mfcc(np.array(signal), n_mfcc = 13, sr = sample_rate, win_length = FRAME_SIZE, hop_length = HOP_LENGTH)
    return mfcc_val

# getting fundamental frequency
def fundamental_freq(signal, FRAME_SIZE, HOP_LENGTH):
    fund_frq = librosa.yin(np.array(signal), fmin = 65, fmax = 2093, frame_length = FRAME_SIZE, hop_length = HOP_LENGTH)
    return fund_frq

# normalizing the frequencies
def freq_normalization(frequency):
    norm_frq = []
    for i in range (0, len(frequency)):
        current_norm_freq = (frequency[i] - np.min(frequency)) / (np.max(frequency) - np.min(frequency))
        norm_frq.append(current_norm_freq)
    return np.array(norm_frq)

# getting amplitude envelope
# calculate Amplitude Envelope for each frame
def calc_amplitude_envelope(signal, FRAME_SIZE, HOP_LENGTH):
    amplitude_env = []
    # calcualte maximum amplitude of each frame
    for i in range(0, len(signal), HOP_LENGTH):
        max_amp_fr = np.max(signal[i : FRAME_SIZE + i])
        amplitude_env.append(max_amp_fr)
    
    amplitude_env.append(amplitude_env[-1])
    # returning the amplitude envelope
    return np.array(amplitude_env)

# getting rms energy for voice data
def rms_voice (signal, FRAME_SIZE, HOP_LENGTH):
    rms_voice = librosa.feature.rms(np.array(signal), frame_length= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
    return rms_voice

# getting zero-crossing-rate
def zero_cross (signal, FRAME_SIZE, HOP_LENGTH):
    zero_cross_rate = librosa.feature.zero_crossing_rate(np.array(signal), frame_length = FRAME_SIZE, hop_length = HOP_LENGTH)[0]
    return zero_cross_rate

# data manipulation and writing feature table
# creting a pandas dataframe for making a feature table
data_frame = pd.DataFrame({"fundamental_frequency" : [],
"mfcc1" : [],
"mfcc2" : [],
"mfcc3" : [],
"mfcc4" : [],
"mfcc5" : [],
"mfcc6" : [],
"mfcc7" : [],
"mfcc8" : [],
"mfcc9" : [],
"mfcc10" : [],
"mfcc11" : [],
"mfcc12" : [],
"mfcc13" : []})

# function to assign emotional intensity
def assign_intense_val(user_inp):
    if user_inp == "Neutral":
        return 0
    elif user_inp == "Onset":
        return 1
    elif user_inp == "Offset":
        return 3
    elif user_inp == "Apex":
        return 7
    
# function to count intensity
def count_intense(prediction):
    print (f"prediction is : {prediction}")
    if prediction == "Neutral":
        neutral.append("Neutral")
    elif prediction == "Onset":
        onset.append("Onset")
    elif prediction == "Offset":
        offset.append("Offset")
    elif prediction == "Apex":
        apex.append("Apex")
    

# function to display max intense
def disp_max_intense(neutral, onset, offset, apex, target_names):
    intense_list = np.array([neutral, onset, offset, apex])
    print (f"Intense list is : {intense_list}")
    idx = intense_list.argmax()
    print (f"Index is : {idx}")
    return target_names[idx]


    
# function to estimate intensity per second per sample
def estimate_intense(df):
    x = df
    x = x[:].values
    x = scaler.fit_transform(x)
    print (x.shape)
    count = 0
    for i in range (0, x.shape[0], 40):
        y = x[count*40 : (count + 1) * 40]
        count = count + 1

        feature1 = y[:,0]
        ft1_mean = np.mean(feature1)
        ft1_std = np.std(feature1)
        feature2 = y[:,1]
        ft2_mean = np.mean(feature2)
        ft2_std = str(np.std(feature2))
        feature3 = y[:,2]
        ft3_mean = np.mean(feature3)
        ft3_std = np.std(feature3)
        feature4 = y[:,3]
        ft4_mean = np.mean(feature4)
        ft4_std = np.std(feature4)
        feature5 = y[:,4]
        ft5_mean = np.mean(feature5)
        ft5_std = np.std(feature5)
        feature6 = y[:,5]
        ft6_mean = np.mean(feature6)
        ft6_std = np.std(feature6)
        feature7 = y[:,6]
        ft7_mean = np.mean(feature7)
        ft7_std = np.std(feature7)
        feature8 = y[:,7]
        ft8_mean = np.mean(feature8)
        ft8_std = np.std(feature8)
        feature9 = y[:,8]
        ft9_mean = np.mean(feature9)
        ft9_std = np.std(feature9)
        feature10 = y[:,9]
        ft10_mean = np.mean(feature10)
        ft10_std = np.std(feature10)
        feature11 = y[:,10]
        ft11_mean = np.mean(feature11)
        ft11_std = np.std(feature11)
        feature12 = y[:,11]
        ft12_mean = np.mean(feature12)
        ft12_std = np.std(feature12)
        feature13 = y[:,12]
        ft13_mean = np.mean(feature13)
        ft13_std = np.std(feature13)
        feature14 = y[:,13]
        ft14_mean = np.mean(feature14)
        ft14_std = np.std(feature14)
        feature15 = y[:,14]
        ft15_mean = np.mean(feature15)
        ft15_std = np.std(feature15)
        feature16 = y[:,15]
        ft16_mean = np.mean(feature16)
        ft16_std = np.std(feature16)
        feature17 = y[:,16]
        ft17_mean = np.mean(feature17)
        ft17_std = np.std(feature17)
        

        input_example = np.array([ft1_mean, ft1_std, ft2_mean , ft2_std , ft3_mean , ft3_std , ft4_mean , ft4_std , ft5_mean , ft5_std, ft6_mean , ft6_std , ft7_mean , ft7_std, ft8_mean , ft8_std , ft9_mean , ft9_std , ft10_mean , ft10_std , ft11_mean , ft11_std , ft12_mean , ft12_std , ft13_mean , ft13_std , ft14_mean , ft14_std,ft15_mean, ft15_std, ft16_mean, ft16_std, ft17_mean, ft17_std])
        input_example = input_example.reshape(1, -1)
        print (input_example)
        print(input_example.shape)
        model = load_model(model_path)
        prediction = model.predict(input_example)
        count_intense(prediction[0])
        pred_score = assign_intense_val(prediction)
        prediction_scores.append(pred_score)
    return prediction_scores





for i in dir_list:
    print (i)
    new_path1 = data_path + i + '/'
    dir_list1 = os.listdir(new_path1)
    print (dir_list1[0])
    # reading signal
    signal_read, sample_rate = read_signal(new_path1 + dir_list1[0] )
    print (sample_rate)
    # filtering signal
    filtered_speech_sig = moving_avg(signal_read, 33)
    # converting from float to pcm data
    pcm_data = float_to_pcm16(filtered_speech_sig[0 : FRAME_SIZE])
    
    # extracting speech segments
    voice_data, unvoice_data = voice_detect(filtered_speech_sig, sample_rate, pcm_data)

    # calculating amplitude envelope
    ampl_env = calc_amplitude_envelope(voice_data, FRAME_SIZE, HOP_LENGTH)
    print(f"Length of amplitude envelope is : {len(ampl_env)}")
    print(f"Length of amplitude envelope is : {type(ampl_env)}")

    # calculate rms energy
    rms_energy = rms_voice(voice_data, FRAME_SIZE, HOP_LENGTH)
    print (f"Length of rms_energy is : {len(rms_energy)}")
    print (f"Length of rms_energy is : {type(rms_energy)}")
    # getting zero-cross-rate
    zero_cross_rate = zero_cross(voice_data, FRAME_SIZE, HOP_LENGTH)
    print (f"Length of zero cross rate is : {len(zero_cross_rate)}")
    print (f"Length of zero cross rate is : {type(zero_cross_rate)}")

    # getting fundamental frequency
    fundamental_frequency = fundamental_freq(voice_data, FRAME_SIZE, HOP_LENGTH)
    fundamental_frequency_normallized = freq_normalization(fundamental_frequency)
    print (f"Length of fundamental_frequency_normalized  is : {len(fundamental_frequency_normallized)}")
    print (f"Length of fundamental_frequency_normalized  is : {type(fundamental_frequency_normallized)}")

    # getting mfcc values
    mfcc_values = mfcc(voice_data, FRAME_SIZE, HOP_LENGTH,sample_rate)
    print (f"Shape of MFCC is : {mfcc_values.shape}")
    mfcc_val1 = mfcc_values[0]
    print (f"Length of mfcc1 is : {len(mfcc_val1)}")
    print (f"Length of mfcc1 is : {type(mfcc_val1)}")
    mfcc_val2 = mfcc_values[1]
    print (f"Length of mfcc2 is : {len(mfcc_val2)}")
    print (f"Length of mfcc2 is : {type(mfcc_val2)}")
    mfcc_val3 = mfcc_values[2]
    print (f"Length of mfcc3 is : {len(mfcc_val3)}")
    print (f"Length of mfcc3 is : {type(mfcc_val3)}")
    mfcc_val4 = mfcc_values[3]
    print (f"Length of mfcc4 is : {len(mfcc_val4)}")
    print (f"Length of mfcc4 is : {type(mfcc_val4)}")
    mfcc_val5 = mfcc_values[4]
    print (f"Length of mfcc5 is : {len(mfcc_val5)}")
    print (f"Length of mfcc5 is : {type(mfcc_val5)}")
    mfcc_val6 = mfcc_values[5]
    print (f"Length of mfcc6 is : {len(mfcc_val6)}")
    print (f"Length of mfcc6 is : {type(mfcc_val6)}")
    mfcc_val7 = mfcc_values[6]
    print (f"Length of mfcc7 is : {len(mfcc_val7)}")
    print (f"Length of mfcc7 is : {type(mfcc_val7)}")
    mfcc_val8 = mfcc_values[7]
    print (f"Length of mfcc8 is : {len(mfcc_val8)}")
    print (f"Length of mfcc8 is : {type(mfcc_val8)}")
    mfcc_val9 = mfcc_values[8]
    print (f"Length of mfcc9 is : {len(mfcc_val9)}")
    print (f"Length of mfcc9 is : {type(mfcc_val9)}")
    mfcc_val10 = mfcc_values[9]
    print (f"Length of mfcc10 is : {len(mfcc_val10)}")
    print (f"Length of mfcc10 is : {type(mfcc_val10)}")
    mfcc_val11 = mfcc_values[10]
    print (f"Length of mfcc11 is : {len(mfcc_val11)}")
    print (f"Length of mfcc11 is : {type(mfcc_val11)}")
    mfcc_val12 = mfcc_values[11]
    print (f"Length of mfcc12 is : {len(mfcc_val12)}")
    print (f"Length of mfcc12 is : {type(mfcc_val12)}")
    mfcc_val13 = mfcc_values[12]
    print (f"Length of mfcc13 is : {len(mfcc_val13)}")
    print (f"Length of mfcc13 is : {type(mfcc_val13)}")


    # appending pandas dataframe
    current_df =  pd.DataFrame({"amplitude_envelope" : np.transpose(ampl_env),
                                "rms_energy" : np.transpose(rms_energy),
                                "zero_cross_rate" : np.transpose(zero_cross_rate),
                                    "fundamental_frequency" : np.transpose(fundamental_frequency_normallized),
                                    "mfcc1" : np.transpose(mfcc_val1),
                                    "mfcc2" : np.transpose(mfcc_val2),
                                    "mfcc3" : np.transpose(mfcc_val3),
                                    "mfcc4" : np.transpose(mfcc_val4),
                                    "mfcc5" : np.transpose(mfcc_val5),
                                    "mfcc6" : np.transpose(mfcc_val6),
                                    "mfcc7" : np.transpose(mfcc_val7),
                                    "mfcc8" : np.transpose(mfcc_val8),
                                    "mfcc9" : np.transpose(mfcc_val9),
                                    "mfcc10" : np.transpose(mfcc_val10),
                                    "mfcc11" : np.transpose(mfcc_val11),
                                    "mfcc12" : np.transpose(mfcc_val12),
                                    "mfcc13" : np.transpose(mfcc_val13),})
    data_frame = pd.concat([data_frame, current_df], axis = 0)
    data_frame.to_csv("Hello" + str(i))

    prediction_scores = estimate_intense(data_frame)
    print (f"Length of the considered clip is : {len(prediction_scores)}")


import matplotlib.pyplot as plt

x_axis = range(len(prediction_scores))
y_axis = prediction_scores
print (prediction_scores)
plt.bar(x_axis, y_axis, color ='blue', width = 0.5)
plt.plot(x_axis, y_axis, color = "red")
plt.title('Intensity of emotion vs Time')
plt.xlabel('Time period')
plt.ylabel('Emotion intensity')
plt.show()

# printing the predicted maximum emotion intensity as user's final emotional intensity
print ("=" * 100)
print (f"The Emotion Intensity of the user throughout the period  is Concluded as : {disp_max_intense(len(neutral), len(onset), len(offset), len(apex), target_names)}")
print ("=" * 100)