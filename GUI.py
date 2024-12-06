from customtkinter import *
import numpy as np
import sounddevice as sd
import threading

import joblib

loaded_scaler = joblib.load('scaler.pkl')
loaded_scaler2 = joblib.load('scaler2.pkl')

loaded_model = joblib.load('logistic_regression_model.pkl')
loaded_model2 = joblib.load('svm_model.pkl')
loaded_model3 = joblib.load('linear_regression_model.pkl')

freq_lst = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
current_freq_index = 0

app = CTk()
app.title("Aspect Ratio 9:16")

initial_width = 360
initial_height = 640
app.geometry(f"{initial_width}x{initial_height}")
app.minsize(initial_width, initial_height)
set_appearance_mode("dark")

app.wm_aspect(9, 16, 9, 16)

panel = CTkFrame(app, width=200, height=100, corner_radius=10)
panel.place(relx=0.5, rely=0.4, anchor="center")

panel_label = CTkLabel(panel, text=f"{freq_lst[current_freq_index]} Hz", font=("Arial", 24, "bold"))
panel_label.place(relx=0.5, rely=0.5, anchor="center")

playing = False
volume = 5

HTL = []
placed_labels=[]

def play_binaural_tone(frequency=250, duration=5, sample_rate=44100):
    global playing
    playing = True

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    left_tone = np.sin(2 * np.pi * frequency * t)
    right_tone = np.sin(2 * np.pi * (frequency + 3) * t)

    stereo_signal = np.stack((left_tone, right_tone), axis=1)/50

    def adjust_volume():
        while playing:
            adjusted_signal = stereo_signal * (volume / 10)
            sd.play(adjusted_signal, sample_rate)
            sd.wait()
            if not playing:
                break

    threading.Thread(target=adjust_volume, daemon=True).start()

def stop_playback():
    global playing
    playing = False
    sd.stop()

def play_button_action():
    global playing
    if not playing:
        play_binaural_tone(frequency=freq_lst[current_freq_index], duration=5)

def increase_volume():
    global volume
    if volume < 10:
        volume += 1
        update_volume_label()
        stop_playback()
        play_binaural_tone(frequency=freq_lst[current_freq_index], duration=5)

def decrease_volume():
    global volume
    if volume > 0:
        volume -= 1
        update_volume_label()
        stop_playback()
        play_binaural_tone(frequency=freq_lst[current_freq_index], duration=5)

def update_frequency():
    panel_label.configure(text=f"{freq_lst[current_freq_index]} Hz")

def next_frequency():
    global current_freq_index, volume

    HTL.append(volume)

    current_freq_index = (current_freq_index + 1) % len(freq_lst)
    update_frequency()

    if freq_lst[current_freq_index] > 1000:
        volume = 3
    else:
        volume = 4

    update_volume_label()

    stop_playback()
    play_binaural_tone(frequency=freq_lst[current_freq_index], duration=5)

    if current_freq_index == len(freq_lst) - 1:
        next_button.configure(text="Finish Test", fg_color="red", hover_color="darkred", command=finish_test)

def restart_test():
    restart_button.place_forget()

    global current_freq_index, volume

    current_freq_index = 0
    volume = 5

    for label in placed_labels:
        label.place_forget()

    placed_labels.clear()
    HTL.clear()

    panel.place(relx=0.5, rely=0.4, anchor="center")
    play_button.place(relx=0.35, rely=0.85, anchor="center")
    stop_button.place(relx=0.65, rely=0.85, anchor="center")
    next_button.place(relx=0.5, rely=0.95, anchor="center")
    plus_button.place(relx=0.35, rely=0.7, anchor="center")
    minus_button.place(relx=0.65, rely=0.7, anchor="center")
    volume_label.place(relx=0.5, rely=0.75, anchor="center")

    update_frequency()
    update_volume_label()

restart_button = CTkButton(app, text="Restart Test", command=restart_test, width=200, fg_color="lightblue", hover_color="deepskyblue")

def finish_test():
    HTL.append(volume)
    stop_playback()
    next_button.configure(text="Next", fg_color="mediumpurple", hover_color="orchid", command=next_frequency)

    panel.place_forget()
    play_button.place_forget()
    stop_button.place_forget()
    next_button.place_forget()
    plus_button.place_forget()
    minus_button.place_forget()
    volume_label.place_forget()

    test_data = [HTL]

    scaled_data = loaded_scaler.transform(test_data)
    scaled_data2 = loaded_scaler2.transform(test_data)

    logistic_prediction = loaded_model.predict(scaled_data)[0]
    svm_prediction = loaded_model2.predict(scaled_data)[0]
    linear_prediction = loaded_model3.predict(scaled_data2)[0]


    y_offset = 0.3
    results = [
        f"Logistic Regression Prediction: {logistic_prediction}",
        f"SVM Prediction: {svm_prediction}",
        f"Linear Regression Result: {linear_prediction:.2f}/5"
    ]

    for result in results:
        label = CTkLabel(
            app,
            text=result,
            font=("Arial", 16, "bold"),
            text_color="white"
        )
        placed_labels.append(label)
        label.place(relx=0.5, rely=y_offset, anchor="center")
        y_offset += 0.1


    HTL.clear()

    restart_button.place(relx=0.5, rely=0.95, anchor="center")

def update_volume_label():
    global volume_label, volume

    volume_label.configure(text=f"Volume: {volume} ")

plus_button = CTkButton(app, text="+", command=increase_volume, width=90, fg_color="green", hover_color="lightgreen")
plus_button.place(relx=0.35, rely=0.7, anchor="center")

minus_button = CTkButton(app, text="-", command=decrease_volume, width=90, fg_color="red", hover_color="lightcoral")
minus_button.place(relx=0.65, rely=0.7, anchor="center")

volume_label = CTkLabel(app, text=f"Volume: {volume}", font=("Arial", 16, "bold"))
volume_label.place(relx=0.5, rely=0.75, anchor="center")

play_button = CTkButton(app, text="Play", command=play_button_action, width=90, fg_color="seagreen", hover_color="forestgreen")
play_button.place(relx=0.35, rely=0.85, anchor="center")

stop_button = CTkButton(app, text="Stop Playing", command=stop_playback, width=90, fg_color="indianred", hover_color="firebrick")
stop_button.place(relx=0.65, rely=0.85, anchor="center")

next_button = CTkButton(app, text="Next", command=next_frequency, width=200, fg_color="mediumpurple", hover_color="orchid")
next_button.place(relx=0.5, rely=0.95, anchor="center")

app.mainloop()