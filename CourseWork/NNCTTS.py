import os

from six import text_type
import NLPModule

def NNCTTS(input_file = "", output_path = "", playback = True, by_sentence = False, logging = True):
    model = NLPModule.startpoint(model_dir = 'D:\Projects\g2p-models\g2p-seq2seq-model-6.2-cmudict-nostress')
    model.interactive(input_file = input_file, output_path = output_path, playback = playback, by_sentence = by_sentence, logging = logging)

if __name__ == "__main__":
    NNCTTS()
    pass