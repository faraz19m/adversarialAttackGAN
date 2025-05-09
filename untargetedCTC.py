from random import random

import numpy as np
import torch
import torchaudio
import torchaudio.pipelines as pipelines
import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity
import scipy.io.wavfile as wav
from scipy.io import wavfile
import librosa
from keras.layers import (Conv1D, BatchNormalization, LeakyReLU, Add, Input,
                          GlobalAveragePooling1D, Dense, UpSampling1D, Concatenate, Lambda,
                          Activation, LSTM, GRU, Reshape)
from keras.metrics import binary_accuracy
import Levenshtein
import nltk
import enchant

from spellchecker import SpellChecker

from nltk.corpus import words
import wave
import datetime
from tensorflow.keras.optimizers import Adam, SGD
from keras import layers, Model
from matplotlib import pyplot as plt
import keras.backend as K
from tensorflow.python.eager import tape
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

if not tf.executing_eagerly():
    tf.config.experimental_run_functions_eagerly(True)

class SpeechGAN_Wav2Vec:
    def __init__(self, clean_wav, target_transcription, alpha=1.0, epsilon=0.5, vocab_size=27, embedding_dim=256):
        self.alpha = alpha
        self.epsilon = epsilon
        self.clean_wav = clean_wav
        self.target_transcription = target_transcription

        nltk.download('words')  # Run this once
        self.word_list = set(words.words())

        # self.english_dict = enchant.Dict("en_US")

        self.spell = SpellChecker()

        # Load Wav2Vec 2.0 model
        model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
        self.model_x = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        clean_audio, sampling_rate = torchaudio.load(clean_wav)
        print("TYPE OF AUDIO: ", type(clean_audio))
        print("SHAPE OF AUDIO: ", clean_audio.shape)


        # Convert target transcription to integer sequence
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9,
                      "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19,
                      "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29,
                      "Q": 30, "Z": 31}

        # Convert target transcription to integer sequence for embedding
        self.target_sequence = [self.vocab[c] for c in target_transcription if c in self.vocab]
        self.target_padded = np.pad(self.target_sequence, (0, 50 - len(self.target_sequence)), mode='constant')
        self.target_padded = np.array(self.target_padded)


        # Optimizers
        self.optimizer_g = Adam(0.0001)
        self.optimizer_d = SGD(0.01, momentum=0.9)
        # Build Generator and Discriminator
        input_shape = (clean_audio.shape[1], 1)
        # print("XXXXXINPUTSHAPE: ", input_shape)
        inputs = Input(shape=input_shape)
        generator = self.build_generator(inputs)
        self.G = Model(inputs, generator, name='Generator')
        self.G.summary()

        discriminator = self.build_discriminator(self.G(inputs))
        self.D = Model(inputs, discriminator, name='Discriminator')

        self.D.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1),
            optimizer=self.optimizer_d,
            metrics=[self.custom_acc]
        )
        self.D.summary()

        self.gan = Model(inputs=inputs,
                         outputs=[self.G(inputs), self.D(inputs), self.G(inputs)])

        self.gan.compile(
            loss=[
                self.hinge_loss_generator,
                tf.keras.losses.binary_crossentropy,
                self.ctc_loss
            ],
            loss_weights=[0.01, 0.5, 1.0],
            optimizer=self.optimizer_g,
            run_eagerly=True
        )
        self.gan.summary()

    def custom_acc(self, y_true, y_pred):
        return binary_accuracy(K.round(y_pred), K.round(y_true))

    def preprocess_audio(self, audio):
        audio_flatten = tf.reshape(audio, [-1])
        audio_clipped = tf.clip_by_value(audio_flatten, -1.0, 1.0) * 32767
        return audio_clipped.numpy().astype(np.int16)

    def transcribe_audio(self, audio):
        with torch.no_grad():
            logits = self.model_x(audio).logits
        # Decode Transcription and Probabilities
        predicted_ids = torch.argmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        # print("Transcription:", transcription)
        return logits, predicted_ids, probs, transcription

    def ctc_loss(self,  target_text, adversarial_audio):
        # print("Target text: ", target_text)
        # adversarial_audio = self.build_generator(adversarial_audio)
        target_text = self.target_transcription
        # print("Adversarial audio shape: ", adversarial_audio.shape, adversarial_audio)
        adversarial_audio = torch.tensor(adversarial_audio.numpy().squeeze(), dtype=torch.float32)

        # convert numpy array to PyTorch tensor:
        audio_tensor = torch.tensor(adversarial_audio, dtype=torch.float32)
        audio_tensor = audio_tensor.unsqueeze((0))
        #Process audio input
        inputs = self.processor(audio_tensor, sampling_rate=16000, return_tensors='pt', padding=True)
        input_values = inputs.input_values.view(1, 1, -1).squeeze(0)
        # print(f"input_values shape: {input_values.shape}")  # Should be (batch, time)
        # Get logits from the Wav2Vec2 model
        with torch.no_grad():
            logits = self.model_x(input_values).logits  # Shape: (batch, time, vocab_size)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Convert target text to token IDs
        target_encoded = self.processor.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True).input_ids

        # Define sequence lengths
        input_lengths = torch.full(size=(log_probs.shape[0],), fill_value=log_probs.shape[1], dtype=torch.long)
        target_lengths = torch.full(size=(target_encoded.shape[0],), fill_value=target_encoded.shape[1], dtype=torch.long)

        # Compute CTC Loss
        ctc_loss_fn = torch.nn.CTCLoss(blank=self.processor.tokenizer.pad_token_id, reduction='mean')
        loss = ctc_loss_fn(log_probs.permute(1, 0, 2), target_encoded, input_lengths, target_lengths)
        # print(loss.item())
        return loss.item()  # Lower loss means a better adversarial attack

    def cosine_similarity_loss(self, y_targ, y_pred):
        # audio = clean_audio.reshape(-1, 1)
        # audio = np.expand_dims(audio, axis=0)
        y_pred = self.G(clean_audio)

        _, _, _, y_pred_transcription = self.transcribe_audio(y_pred)

        y_pred_sequence = [self.vocab.get(c, 26) for c in y_pred_transcription]
        y_pred_padded = np.pad(y_pred_sequence, (0, 50 - len(y_pred_sequence)), mode='constant')

        y_pred_padded = tf.convert_to_tensor(y_pred_padded, dtype=tf.float32)

        y_targ = tf.cast(y_targ, tf.float32)
        y_targ = tf.nn.l2_normalize(y_targ, axis=-1)
        y_pred_padded = tf.nn.l2_normalize(y_pred_padded, axis=-1)

        cosine_similarity = tf.reduce_sum(y_targ * y_pred_padded, axis=-1)
        loss = 1 - cosine_similarity

        return loss * 2

    def mse_loss(self, y_targ, y_pred):
        pred_ids,y_pred_transcription = tf.numpy_function(self.get_pred_ids_flattened_fn, [y_pred], tf.float32)
        pred_ids = tf.reshape(pred_ids, [-1])
        pred_ids = tf.cast(pred_ids, tf.float32)

        # Flatten and cast target IDs
        y_targ_flat = tf.reshape(tf.cast(y_targ, tf.float32), [-1])

        # Compute Mean Squared Error
        print("pred_ids.shape",pred_ids.shape)
        print("y_targ_flat.shape",y_targ_flat.shape)
        loss = tf.reduce_mean(tf.square(pred_ids - y_targ_flat))

        return loss

    def hinge_loss_generator(self, y_true, y_pred):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.1, 0), axis=-1)

    def get_pred_ids_flattened_fn(self, adv_audio_np):
        adv_audio = torch.tensor(adv_audio_np.squeeze(), dtype=torch.float32)
        adv_audio = self.processor(adv_audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        adv_audio = adv_audio.reshape(1, -1)
        logits, y_pred_prediction_ids, probs, y_pred_transcription = self.transcribe_audio(adv_audio)
        return y_pred_prediction_ids.detach().cpu().numpy().flatten(), y_pred_transcription

    def build_generator(self, inputs):
        x = Conv1D(32, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(1, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Add()([x * 0.25, inputs])
        return x

    def build_discriminator(self, generator_output):
        x = Conv1D(16, kernel_size=3, strides=2, padding='same')(generator_output)
        x = LeakyReLU()(x)
        x = Conv1D(32, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(8, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return x

    def train_discriminator(self, x_, Gx_):
        self.D.trainable = True
        real_labels = np.ones((1, 1)) * 0.9  # Real data labels
        real_loss = self.D.train_on_batch(x_, real_labels)

        # Train Discriminator on fake data (generated by the generator)
        fake_labels = np.zeros((1, 1)) + 0.1  # Fake data labels
        fake_loss = self.D.train_on_batch(Gx_, fake_labels)

        # Total discriminator loss
        d_loss = 0.25 * np.add(real_loss, fake_loss)
        # print("D_LOSS: ", d_loss)
        return d_loss

    def train_generator(self, x_):
        self.D.trainable = False
        N = x_.shape[0]  # Get the batch size
        real_label = np.ones((1, 1))  # discriminator target for generator's output
        target_text = self.target_transcription

        target_encoded = self.processor.tokenizer(target_text, return_tensors="pt", padding=True,
                                                  truncation=True).input_ids
        target_encoded = tf.convert_to_tensor(target_encoded, dtype=tf.int32)  # Convert to TensorFlow tensor
        target_encoded_batch = tf.reshape(target_encoded, (N, -1))

        print("target_encoded_batch.shape", target_encoded_batch.shape)
        print("target_encoded_batch", target_encoded_batch)

        # print("Adversarial audio input shape: ", x_.shape)

        g_loss = self.gan.train_on_batch(x_, [x_, real_label, target_encoded_batch])

        return g_loss

    def save_audio(self, filename, audio_data, sample_rate=16000):
        wav.write(filename, sample_rate, audio_data)  # Save as 16-bit PCM

    def plot_audio(self, epoch, original_audio, adversarial_audio, sample_rate=16000):
        # Ensure audio arrays are flattened to shape (16000,)
        # original_audio = original_audio.flatten()
        adversarial_audio = adversarial_audio.flatten()
        # noise = noise.flatten()

        # Create a time axis that corresponds to the audio length
        time = np.linspace(0, len(original_audio) / sample_rate, num=len(original_audio))
        plt.figure(figsize=(10, 4))

        # Plot the original audio
        plt.plot(time, original_audio, label="Original Audio", alpha=0.7)


        # Plot the adversarial audio
        plt.plot(time, adversarial_audio, label="Adversarial Audio", alpha=0.7)

        # Plot the noise (adversarial audio - original audio)
        # plt.plot(time, noise, label="Noise", alpha=0.7)

        # Add labels and legend
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Original, Adversarial Audio, and Noise")
        plt.legend(loc="upper right")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # plt.savefig(f"HPC_yes_{self.target_transcription}_epoch_{epoch}_time_{timestamp}.png")
        plt.show()

    # def is_semantically_valid(self, transcription):
    #     words_in_transcription = transcription.lower().split()
    #     return all(word in self.word_list for word in words_in_transcription)

    # def is_semantically_valid(self, transcription):
    #     words_in_transcription = transcription.lower().split()
    #     return all(self.english_dict.check(word) for word in words_in_transcription)

    def is_semantically_valid(self, transcription):
        words_in_transcription = transcription.lower().split()
        misspelled = self.spell.unknown(words_in_transcription)
        return len(misspelled) == 0

    def train(self, epochs=1000):

        print("target transcription", self.target_transcription)
        _, original_prediction_ids, _, original_transcription = self.transcribe_audio(clean_audio)
        print("Original transcription:", original_transcription)
        original_length = len(original_transcription)
        threshold = original_length
        print("threshold", threshold)

        tgt_ids = [self.vocab[c] for c in self.target_transcription]

        # print("ORINAL GRALSIFJ: ", original_transcription)
        audio = clean_audio.reshape(-1, 1)  # Reshape raw audio for model input (batch size 1)
        X_batch = np.expand_dims(audio, axis=0)  # Expand to match the batch size (1, ...)
        saved_untargeted = 1
        for epoch in range(epochs):
            print("==========================================================================================")
            print("==========================================================================================")
            print("epoch", epoch)

            Gx = self.G.predict(X_batch)

            # adv_audio =  Gx# + np.expand_dims(audio, axis=0)
            adv_audio = np.clip(Gx, -1.0, 1.0)


            print("Original_max, Original_min", max(audio), min(audio))
            # adv_audio = Gx
            print("Adv_audio_max, Adv_audio_min", max(adv_audio[0]), min(adv_audio[0]))
            d_loss = self.train_discriminator(X_batch, adv_audio)
            losses  = self.train_generator(X_batch)
            # Print individual losses
            total_loss = losses[0]
            generator_loss = losses[1]
            discriminator_loss = losses[2]
            ctc_loss = losses[3]

            ### CHECKING NOISE MAGNITUDES::

            audio_0_numpy = audio.cpu().detach().numpy()
            # noise = adv_audio[0]  - audio_0_numpy
            # print("noise_max, noise_min", max(noise), min(noise))

            Adv_torch = torch.tensor(adv_audio, dtype=torch.float32)
            Adv_torch = Adv_torch.squeeze(-1)

            Adv_torch = torch.tensor(Adv_torch.numpy(), dtype=torch.float32)
            Adv_torch = self.processor(Adv_torch, sampling_rate=16000, return_tensors="pt", padding=True).input_values
            Adv_torch = Adv_torch.reshape(1, -1)

            with torch.no_grad():
                logits = self.model_x(Adv_torch).logits
            # Decode Transcription and Probabilities
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            diff_count = sum(1 for a, b in zip(original_transcription, transcription) if a != b)
            diff_count += abs(len(original_transcription) - len(transcription))  # Account for extra/missing chars
            print(f"Character difference: {diff_count} / Threshold: {threshold}")

            lev_distance = Levenshtein.distance(original_transcription, transcription)
            print(f"Levenshtein distance: {lev_distance} / Threshold: {threshold}")

            if lev_distance != diff_count:
                print("DIFFERENCE")

            print(f"EPOCH {epoch}:  Adversarial Audio: {transcription}      "
                  f"Tot_loss: {total_loss},     Generator_loss: {generator_loss},   "
                  f"Discriminator_LOSS: {discriminator_loss},  CTC_LOSS: {ctc_loss}")

            if (epoch % 20 == 0 and epoch != 0) or epoch == epochs - 1:
                Adv_numpy = np.array(adv_audio[0]).flatten()
                self.plot_audio(epoch, audio_0_numpy, Adv_numpy)

            if lev_distance >= threshold:
                print("Threshold exceeded — untargeted attack successful.")
                if self.is_semantically_valid(transcription):
                    print("Adversarial Transcription: ", transcription, "; Original Transcription: ",
                          original_transcription)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save_audio(f"untargeted_epoch_{epoch}_{timestamp}.wav", adv_audio[0], sample_rate=16000)

                    audio_0_numpy = audio.cpu().detach().numpy()
                    Adv_numpy = np.array(adv_audio[0]).flatten()
                    self.plot_audio(epoch, audio_0_numpy, Adv_numpy)
                    break
                else:
                    print("Semantically invalid transcription — skipping.")
                    print("Adversarial Transcription: ", transcription, "; Original Transcription: ",
                          original_transcription)


# Initialize parameters
clean_wav_path = "data/yes/c120e80e_nohash_4.wav"  # Your clean .wav file
adversarial_wav_path = "adversarial_audio.wav"  # Path to save the generated adversarial audio
target_transcription = ""  # The transcription you want DeepSpeech to wrongly predict

clean_audio, sampling_rate = torchaudio.load(clean_wav_path)
print("TypeCleanAudio: ", type(clean_audio))
print("ShapeCleanAudio: ", clean_audio.shape)

# Instantiate SpeechGAN
speech_gan = SpeechGAN_Wav2Vec(
    clean_wav=clean_wav_path,
    target_transcription=target_transcription
)

# Train the model
speech_gan.train(epochs=10000)