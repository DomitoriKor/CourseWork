#!/usr/bin/python3
import os
import sys
import shutil
from datetime import datetime
from gtts import gTTS
from playwav import playsound

import pyaudio  
import wave

sys.path += ['D:\Projects\ScoreDraft-master\python_test']
temp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

import ScoreDraft
from ScoreDraftNotes import *
import VCCVEnglishConverter

def produce_speech(sentence, output_path = "", playback = True, by_sentence = False, sentence_number = '1'):
    try:
        return_message = ""
        if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
        filename = "temp" + sentence_number + ".wav"
        temp_filename = os.path.join(temp_dir, filename)

        doc = ScoreDraft.Document()
        doc.setTempo(120)

        sn = 24 #Short note
        ln = 48 #Long note

        singer = ScoreDraft.Yami_UTAU()
        ScoreDraft.UtauDraftSetLyricConverter(singer, VCCVEnglishConverter.VCCVEnglishConverter)
        singer.tune('CZMode')

        sequence = []
        for word in sentence:
            subsequence = ()
            for syllable in word:
                if syllable[0] == 'rest':
                    if len(subsequence) != 0:
                        sequence.append(subsequence)
                        subsequence = ()
                    if syllable[1] == -3: #Long pause
                        sequence.append(BL(ln))
                    elif syllable[1] == -2: #Short pause
                        sequence.append(BL(sn))
                elif syllable[0] != '':
                    if syllable[1] == 0: #Unstressed syllable
                        subsequence += (syllable[0], mi(5, sn))
                    elif syllable[1] == 2: #Secondary stressed syllable
                        subsequence += (syllable[0], fa(5, sn))
                    elif syllable[1] == 1: #Primary stressed syllable
                        subsequence += (syllable[0], so(5, sn))
            if len(subsequence) != 0:
                sequence.append(subsequence)

        doc.sing(sequence, singer)
        doc.mixDown(temp_filename)
        rendered_filename = temp_filename

        if output_path != "":
            try:
                if not "." in os.path.basename(output_path) or os.path.basename(output_path).startswith("."):
                    if not os.path.exists(os.path.abspath(output_path)):
                        os.mkdir(os.path.abspath(output_path))
                    filename = 'sentence' + sentence_number + '-on-{}.wav'.format(str(datetime.now()).replace(':', '-'))
                    output_filename = os.path.join(output_path, filename)
                else:
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.mkdir(os.path.dirname(output_path))
                    output_filename = output_path
                    if not output_path.endswith(".wav"):
                        output_filename += sentence_number*by_sentence + ".wav"
                shutil.copy(temp_filename, output_filename)
                rendered_filename = output_filename
                return_message += "----Rendered .wav file is available at " + rendered_filename + "\n"
            except:
                pass
                return_message += "!!!-An error occured during saving rendered file at output path\n"
        
    except:
        return_message += "!!!-An error occured during rendering process\n"
    try:
        if playback:
            chunk = 1024  
            file = wave.open(rendered_filename, "rb")  
            player = pyaudio.PyAudio()
            stream = player.open(format = player.get_format_from_width(file.getsampwidth()),  
                channels = file.getnchannels(),  
                rate = file.getframerate(),  
                output = True)

            data = file.readframes(chunk) 
            while len(data) > 0:
                stream.write(data)  
                data = file.readframes(chunk)

            stream.stop_stream()  
            stream.close()

            player.terminate()
#            playsound(rendered_filename)
            pass
    except:
        pass
    shutil.rmtree(temp_dir, ignore_errors = True)
    return return_message


def render_speech(sentence, output_path = "", playback = True, by_sentence = False, sentence_number = '0'):
    try:
        return_message = ""
        if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
        filename = "temp" + sentence_number + ".mp3"
        temp_filename = os.path.join(temp_dir, filename)
        comm = 'gtts-cli "' + sentence + '" --output ' + temp_filename
        os.system(comm)
        rendered_filename = temp_filename
        if output_path != "":
            try:
                if not "." in os.path.basename(output_path) or os.path.basename(output_path).startswith("."):
                    if not os.path.exists(os.path.abspath(output_path)):
                        os.mkdir(os.path.abspath(output_path))
                    filename = 'sentence' + sentence_number + '-on-{}.mp3'.format(str(datetime.now()).replace(':', '-'))
                    output_filename = os.path.join(output_path, filename)
                else:
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.mkdir(os.path.dirname(output_path))
                    output_filename = output_path
                    if not output_path.endswith(".mp3"):
                        output_filename += sentence_number*by_sentence + ".mp3"
                shutil.copy(temp_filename, output_filename)
                rendered_filename = output_filename
                return_message += "----Rendered .mp3 file is available at " + rendered_filename + "\n"
            except:
                pass
                return_message += "!!!-An error occured during saving rendered file at output path\n"
        
    except:
        return_message += "!!!-An error occured during rendering process\n"
    try:
        if playback:
            playsound(rendered_filename)
    except:
        pass
    shutil.rmtree(temp_dir, ignore_errors = True)
    return return_message







