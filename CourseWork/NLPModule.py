# -*- coding: utf-8 -*-
##########################
#  IMPORTS
##########################
#####
#  app.py
#####
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from gtts import gTTS

import os
import shutil
import tensorflow as tf
import DSPModule
#import g2p_seq2seq.g2p_trainer_utils as g2p_trainer_utils
#from g2p_seq2seq.g2p import G2PModel
#from g2p_seq2seq.params import Params

#####
#  g2p.py
#####
import contextlib
import re
import numpy as np
import six

from tensor2tensor.data_generators.problem import problem_hparams_to_features
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.framework import graph_util

# Dependency imports

from tensor2tensor import models # pylint: disable=unused-import

#from g2p_seq2seq import g2p_problem
#from g2p_seq2seq import g2p_trainer_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from tensor2tensor.data_generators import text_encoder
from six.moves import input
from six import text_type

EOS = text_encoder.EOS

#####
#  g2p_encoder.py
#####
PAD = text_encoder.PAD

#####
#  g2p_problem.py
#####
import random
from collections import OrderedDict

from tensorflow.python.data.ops import dataset_ops as dataset_ops
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems

#####
#  g2p_trainer_utils.py
#####
import json

from tensorflow.contrib.learn.python.learn import learn_runner

from tensor2tensor.utils import devices

#####
#  transcribe.py
#####
from os.path import join, abspath, dirname
import sqlite3
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from num2words import num2words

temp_dir = join(os.path.dirname(__file__), 'tmp')
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)


######################
#  CODE
######################
#####
#  app.py
#####
def startpoint(model_dir = None, interactive = False, evaluate  = "", decode = "", output = "", train = "",
               valid = "", test = "", reinit = False, freeze = False,
               
               batch_size = 4096, min_length_bucket = 6, max_length = 30, length_bucket_step = 1.5, num_layers = 3,
               size = 256, filter_size = 512, num_heads = 4, max_epochs = 0, cleanup = False, save_checkpoints_steps = None,
               
               return_beams = False, beam_size = 1, alpha = 0.6):

    tf.flags.DEFINE_string("model_dir", model_dir, "Training directory.")
    tf.flags.DEFINE_boolean("interactive", interactive,
                            "Set to True for interactive decoding.")
    tf.flags.DEFINE_string("evaluate", evaluate, "Count word error rate for file.")
    tf.flags.DEFINE_string("decode", decode, "Decode file.")
    tf.flags.DEFINE_string("output", output, "Decoding result file.")
    tf.flags.DEFINE_string("train", train, "Train dictionary.")
    tf.flags.DEFINE_string("valid", valid, "Development dictionary.")
    tf.flags.DEFINE_string("test", test, "Test dictionary.")
    tf.flags.DEFINE_boolean("reinit", reinit,
                            "Set to True for training from scratch.")
    tf.flags.DEFINE_boolean("freeze", freeze,
                            "Set to True for freeze the graph.")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", batch_size,
                            "Batch size to use during training.")
    tf.flags.DEFINE_integer("min_length_bucket", min_length_bucket,
                            "Set the size of the minimal bucket.")
    tf.flags.DEFINE_integer("max_length", max_length,
                            "Set the size of the maximal bucket.")
    tf.flags.DEFINE_float("length_bucket_step", length_bucket_step,
        """This flag controls the number of length buckets in the data reader.
        The buckets have maximum lengths from min_bucket_length to max_length,
        increasing (approximately) by factors
        of length_bucket_step.""")
    tf.flags.DEFINE_integer("num_layers", num_layers, "Number of hidden layers.")
    tf.flags.DEFINE_integer("size", size,
                            "The number of neurons in the hidden layer.")
    tf.flags.DEFINE_integer("filter_size", filter_size,
                            "The size of the filter in a convolutional layer.")
    tf.flags.DEFINE_integer("num_heads", num_heads,
                            "Number of applied heads in Multi-attention mechanism.")
    tf.flags.DEFINE_integer("max_epochs", max_epochs,
                            "How many epochs to train the model."
                            " (0: no limit).")
    tf.flags.DEFINE_boolean("cleanup", cleanup,
                            "Set to True for cleanup dictionary from stress and "
                            "comments (after hash or inside braces).")
    tf.flags.DEFINE_integer("save_checkpoints_steps", save_checkpoints_steps,
    """Save checkpoints every this many steps. Default=None means let
    tensorflow.contrib.learn.python.learn decide, which saves checkpoints
    every 600 seconds.""")
    
    # Decoding parameters
    tf.flags.DEFINE_boolean("return_beams", return_beams,
                            "Set to true for beams decoding.")
    tf.flags.DEFINE_integer("beam_size", beam_size, "Number of decoding beams.")
    tf.flags.DEFINE_float("alpha", alpha,
        """Float that controls the length penalty. Larger the alpha, stronger the
        preference for longer sequences.""")

    FLAGS = tf.app.flags.FLAGS

    tf.logging.set_verbosity(tf.logging.ERROR)
    file_path = FLAGS.train or FLAGS.decode or FLAGS.evaluate
    test_path = FLAGS.decode or FLAGS.evaluate or FLAGS.test
    if not FLAGS.model_dir:
        raise RuntimeError("Model directory not specified.")

    if FLAGS.reinit and os.path.exists(FLAGS.model_dir):
        shutil.rmtree(FLAGS.model_dir)

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    params = Params(FLAGS.model_dir, file_path, flags=FLAGS)

    if FLAGS.train:
        save_params(FLAGS.model_dir, params.hparams)
        g2p_model = G2PModel(params, train_path=FLAGS.train, dev_path=FLAGS.valid,
                            test_path=test_path, cleanup=FLAGS.cleanup)
        #g2p_model.train()

    else:
        params.hparams = load_params(FLAGS.model_dir)
        g2p_model = G2PModel(params, test_path=test_path)

        #if FLAGS.freeze:
            #g2p_model.freeze()

        #elif FLAGS.interactive:
            #g2p_model.interactive()

        #elif FLAGS.decode:
            #g2p_model.decode(output_file_path=FLAGS.output)

        #elif FLAGS.evaluate:
            #g2p_model.evaluate()
    return g2p_model
        
#####
#  g2p.py
#####
class G2PModel(object):
  """Grapheme-to-Phoneme translation model class.
  """
  def __init__(self, params, train_path="", dev_path="", test_path="",
               cleanup=False):
    # Point out the current directory with t2t problem specified for g2p task.
    usr_dir.import_usr_dir(os.path.dirname(os.path.abspath(__file__)))
    self.params = params
    self.test_path = test_path
    if not os.path.exists(self.params.model_dir):
      os.makedirs(self.params.model_dir)

    # Register g2p problem.
    self.problem = registry._PROBLEMS[self.params.problem_name](
        self.params.model_dir, train_path=train_path, dev_path=dev_path,
        test_path=test_path, cleanup=cleanup)

    self.frozen_graph_filename = join(self.params.model_dir,
                                              "frozen_model.pb")
    self.frozen_loaded = False
    self.inputs, self.features, self.input_fn = None, None, None
    self.mon_sess, self.estimator_spec, self.g2p_gt_map = None, None, None
    self.first_ex = False
    if train_path:
      self.train_preprocess_file_path, self.dev_preprocess_file_path =\
          None, None
      self.estimator, self.decode_hp, self.hparams =\
          self.__prepare_model()
      self.train_preprocess_file_path, self.dev_preprocess_file_path =\
          self.problem.generate_preprocess_data()

    elif os.path.exists(self.frozen_graph_filename):
      self.estimator, self.decode_hp, self.hparams =\
          self.__prepare_model()
      self.__load_graph()
      self.checkpoint_path = tf.train.latest_checkpoint(self.params.model_dir)

    else:
      self.estimator, self.decode_hp, self.hparams =\
          self.__prepare_model()

    self.res_iter = list.__iter__

  def __prepare_model(self):
    """Prepare utilities for decoding."""
    hparams = trainer_lib.create_hparams(
        hparams_set=self.params.hparams_set,
        hparams_overrides_str=self.params.hparams)
    trainer_run_config = create_run_config(hparams,
        self.params)
    exp_fn = create_experiment_fn(self.params, self.problem)
    self.exp = exp_fn(trainer_run_config, hparams)

    decode_hp = decoding.decode_hparams(self.params.decode_hparams)
    decode_hp.add_hparam("shards", self.params.decode_shards)
    decode_hp.add_hparam("shard_id", self.params.worker_id)
    estimator = trainer_lib.create_estimator(
        self.params.model_name,
        hparams,
        trainer_run_config,
        decode_hparams=decode_hp,
        use_tpu=False)

    return estimator, decode_hp, hparams

  def __prepare_interactive_model(self):
    """Create monitored session and generator that reads from the terminal and
    yields "interactive inputs".

    Due to temporary limitations in tf.learn, if we don't want to reload the
    whole graph, then we are stuck encoding all of the input as one fixed-size
    numpy array.

    We yield int32 arrays with shape [const_array_size].  The format is:
    [num_samples, decode_length, len(input ids), <input ids>, <padding>]

    Raises:
      ValueError: Could not find a trained model in model_dir.
      ValueError: if batch length of predictions are not same.
    """

    def input_fn():
      """Input function returning features which is a dictionary of
        string feature name to `Tensor` or `SparseTensor`. If it returns a
        tuple, first item is extracted as features. Prediction continues until
        `input_fn` raises an end-of-input exception (`OutOfRangeError` or
        `StopIteration`)."""
      gen_fn = decoding.make_input_fn_from_generator(
          self.__interactive_input_fn())
      example = gen_fn()
      example = decoding._interactive_input_tensor_to_features_dict(
          example, self.hparams)
      return example

    self.res_iter = self.estimator.predict(input_fn)

    if os.path.exists(self.frozen_graph_filename):
      return

    # List of `SessionRunHook` subclass instances. Used for callbacks inside
    # the prediction call.
    hooks = estimator_lib._check_hooks_type(None)

    # Check that model has been trained.
    # Path of a specific checkpoint to predict. The latest checkpoint
    # in `model_dir` is used
    checkpoint_path = estimator_lib.saver.latest_checkpoint(
        self.params.model_dir)
    if not checkpoint_path:
      raise ValueError('Could not find trained model in model_dir: {}.'
                       .format(self.params.model_dir))

    with estimator_lib.ops.Graph().as_default() as graph:

      estimator_lib.random_seed.set_random_seed(
          self.estimator._config.tf_random_seed)
      self.estimator._create_and_assert_global_step(graph)

      self.features, input_hooks = self.estimator._get_features_from_input_fn(
          input_fn, estimator_lib.model_fn_lib.ModeKeys.PREDICT)
      self.estimator_spec = self.estimator._call_model_fn(
          self.features, None, estimator_lib.model_fn_lib.ModeKeys.PREDICT,
          self.estimator.config)
      self.mon_sess = estimator_lib.training.MonitoredSession(
          session_creator=estimator_lib.training.ChiefSessionCreator(
              checkpoint_filename_with_path=checkpoint_path,
              scaffold=self.estimator_spec.scaffold,
              config=self.estimator._session_config),
          hooks=hooks)

  def decode_word(self, word):
    """Decode word.

    Args:
      word: word for decoding.

    Returns:
      pronunciation: a decoded phonemes sequence for input word.
    """
    num_samples = 1
    decode_length = 100
    vocabulary = self.problem.source_vocab
    # This should be longer than the longest input.
    const_array_size = 50

    input_ids = vocabulary.encode(word)
    input_ids.append(text_encoder.EOS_ID)
    self.inputs = [num_samples, decode_length, len(input_ids)] + input_ids
    assert len(self.inputs) < const_array_size
    self.inputs += [0] * (const_array_size - len(self.inputs))

    result = next(self.res_iter)
    pronunciations = ""
    if self.decode_hp.return_beams:
      beams = np.split(result["outputs"], self.decode_hp.beam_size, axis=0)
      for k, beam in enumerate(beams):
        tf.logging.info("BEAM %d:" % k)
        beam_string = self.problem.target_vocab.decode(
            decoding._save_until_eos(beam, is_image=False))
        pronunciations += beam_string + ' '
        tf.logging.info(beam_string)
    else:
      if self.decode_hp.identity_output:
        tf.logging.info(" ".join(map(str, result["outputs"].flatten())))
      else:
        res = result["outputs"].flatten()
        if text_encoder.EOS_ID in res:
          index = list(res).index(text_encoder.EOS_ID)
          res = res[0:index]
        pronunciations += self.problem.target_vocab.decode(res)
    return pronunciations

  def __interactive_input_fn(self):
    num_samples = self.decode_hp.num_samples if self.decode_hp.num_samples > 0\
        else 1
    decode_length = self.decode_hp.extra_length
    input_type = "text"
    problem_id = 0
    p_hparams = self.hparams.problems[problem_id]
    has_input = "inputs" in p_hparams.input_modality
    vocabulary = p_hparams.vocabulary["inputs" if has_input else "targets"]
    # This should be longer than the longest input.
    const_array_size = 10000
    # Import readline if available for command line editing and recall.
    try:
      import readline  # pylint: disable=g-import-not-at-top,unused-variable
    except ImportError:
      pass
    while True:
      features = {
          "inputs": np.array(self.inputs).astype(np.int32),
      }
      for k, v in six.iteritems(problem_hparams_to_features(p_hparams)):
        features[k] = np.array(v).astype(np.int32)
      yield features

  def __run_op(self, sess, decode_op, feed_input):
    """Run tensorflow operation for decoding."""
    results = sess.run(decode_op,
                       feed_dict={"inp_decode:0" : [feed_input]})
    return results

  def train(self):
    """Run training."""
    execute_schedule(self.exp, self.params)

##### INTERACTIVE MODE #####
  def interactive(self, input_file = "", output_path = "", playback = True, by_sentence = False, logging = True):
        """Interactive decoding."""
        self.inputs = []
        word = "warmup"
        prev_word = ""
        text = []
        punct_marks = '!"#$%&\'()*+,-./:;<=>/?@[\\]^_`{|}~«» '
        self.__prepare_interactive_model()
        if False:
            pass
        #if os.path.exists(self.frozen_graph_filename):
        #    with tf.Session(graph=self.graph) as sess:
        #        saver = tf.train.import_meta_graph(self.checkpoint_path + ".meta",
        #                                       import_scope=None,
        #                                       clear_devices=True)
        #        saver.restore(sess, self.checkpoint_path)
        #        inp = tf.placeholder(tf.string, name="inp_decode")[0]
        #        decode_op = tf.py_func(self.decode_word, [inp], tf.string)
        #        self.__run_op(sess, decode_op, word)
        #        self.frozen_loaded = True
        #        while word != None:
        #            text = get_text()
        #            for sentence in text:
        #                for word in word_tokenize(sentence):
        #                    if word.isalpha():
        #                        #####convert(text, stress_marks='place')
        #                        self.get_word_pron(word)
        #                        pass
        #                    if word.isdigit():
        #                        #number_words = number_to_words(word)
        #                        #for w in number_words:
        #                        #self.get_word_pron(w)
        #                        pass
        #            #result = self.__run_op(sess, decode_op, word).decode("utf-8")
        #            #print("Output: {}".format(result))
        else:
            self.decode_word(word)
            self.frozen_loaded = False
            os.system("cls")
            sentence_number = 1
            while not self.mon_sess.should_stop():
                text = get_text()
                sentence_words = ""
                sentence_syllables = []
                for sentence in text:
                    sentence_transcription = ""
                    if by_sentence:
                        sentence_words = ""
                        sentence_syllables = []

                    if sentence == [None]:
                        break
                    end_punct = False
                    for word in word_tokenize(sentence):
                        pronunciation = []
                        normalized_word = ""
                        add_pause = False
                        saved_chunk = ""
                        for chunk in chartype_tokenize(word):
                            is_converted = False
                            if chunk.isdigit():
                                chunk = num2words(int(chunk)).replace('-', ' ')
                                is_converted = True
                            if not (chunk.isalpha() or is_converted):
                                if word in ['.','!','?', '...', '?!', '!?']:
                                    add_pause = True
                                    pause_len = -3 # Long pause for ending punctuation
                                elif word in [',', ';', ':', '-', '--']:
                                    add_pause = True
                                    pause_len = -2 # Short pause for middle punctuation
                                saved_chunk = chunk
                                chunk = ''

                            normalized_word += chunk + ' '
                        if add_pause:
                            sentence_words = sentence_words.rstrip() + saved_chunk + ' '
                            if len(sentence_syllables) != 0:
                                pronunciation = [{'transcription' : [''], 'phones_map' : [('rest', pause_len)]}]
                            pass
                        if not normalized_word.isspace() and len(normalized_word) != 0:
                            pronunciation = self.get_word_pron(normalized_word)
                            sentence_words += normalized_word
                        for word_pron in pronunciation:
                            if word_pron['transcription'][0] != '':
                                sentence_transcription += word_pron['transcription'][0] + " "
                            sentence_syllables.append([])
                            for syllable in word_pron['phones_map']:
                                if syllable[0] != '':
                                    sentence_syllables[-1].append(syllable)
                                end_punct = syllable[0] == 'rest'
                        pass
                    
                    if len(sentence_syllables) != 0 and not end_punct:
                        sentence_syllables.append([('rest', -3)])
                    #print(sentence, sentence_words, sentence_transcription, sentence_syllables, sep = '\n', end = '\n\n')
                    if logging:
                        print("--Sentence {}:".format(sentence_number), sentence)
                        print("----Normalized form: " + sentence_words)
                        print("----Trancsription: " + sentence_transcription)
                        print("----Syllable map: ")
                        print(sentence_syllables)
                        print()
                    if by_sentence:
#                        render_message = DSPModule.produce_speech(sentence_words, output_path = output_path, playback = playback, by_sentence=by_sentence, sentence_number = str(sentence_number))
                        render_message = DSPModule.produce_speech(sentence_syllables, output_path = output_path, playback = playback, by_sentence=by_sentence, sentence_number = str(sentence_number))
                        if logging:
                            print(render_message)
                    sentence_number += 1
                if not by_sentence:
#                    render_message = DSPModule.produce_speech(sentence_words, output_path = output_path, playback = playback, by_sentence=by_sentence, sentence_number = str(sentence_number))
                    render_message = DSPModule.produce_speech(sentence_syllables, output_path = output_path, playback = playback, by_sentence = by_sentence, sentence_number = str(sentence_number))
                    if logging:
                        print(render_message)

                if sentence == [None]:
                    break

#        os.system("pause")

  ##### CMU TO XSAMPA #####    
  def get_cmu(self, tokens_in):
        """query the SQL database for the words and return the phonemes in the order of user_in"""
        result = fetch_words(tokens_in)
        ordered = []
        for word in tokens_in:
            this_word = [[i[1] for i in result if i[0] == word]][0]
            if this_word:
                ordered.append(this_word[0])
            else:
                if self.frozen_loaded:
                    result = self.__run_op(sess, decode_op, word).decode("utf-8")
                else:
                    result = self.decode_word(word) 
                ordered.append([result.lower()])
        return ordered

  def ipa_list(self, words_in, stress_marks='place'):
        """Returns a list of all the discovered ipa transcriptions for each word."""
        if type(words_in) == str:
            words = [preserve_punc(w.lower())[0] for w in words_in.split()]
        else:
            words = [preserve_punc(w.lower())[0] for w in words_in]
        cmu = self.get_cmu([w[1] for w in words])
        ipa = cmu_to_ipa(cmu, stress_marking=stress_marks)
        #if keep_punct:
            #ipa = _punct_replace_word(words, ipa)
        return ipa

  def get_word_pron(self, text):
      ipa = self.ipa_list(words_in = text)
      return ipa

  def decode(self, output_file_path = ""):
    """Run decoding mode."""
    if os.path.exists(self.frozen_graph_filename):
      with tf.Session(graph=self.graph) as sess:
        inp = tf.placeholder(tf.string, name="inp_decode")[0]
        decode_op = tf.py_func(self.__decode_from_file, [inp], [tf.string, tf.string])
        [inputs, decodes] = self.__run_op(sess, decode_op, self.test_path)
    else:
      inputs, decodes = self.__decode_from_file(self.test_path)

    # If path to the output file pointed out, dump decoding results to the file
    if output_file_path:
      tf.logging.info("Writing decodes into %s" % output_file_path)
      outfile = tf.gfile.Open(output_file_path, "w")
      if self.decode_hp.return_beams:
        for index in range(len(inputs)):
          outfile.write("%s%s" % ("\t".join(decodes[index]), self.decode_hp.delimiter))
          outfile.write("\n")
      else:
        for index in range(len(inputs)):
          outfile.write("%s%s" % (decodes[index], self.decode_hp.delimiter))
    else:
        ret = ""
        if self.decode_hp.return_beams:
          for index in range(len(inputs)):
            ret += ("%s%s" % ("\t".join(decodes[index]), self.decode_hp.delimiter))
            ret += "\n"
        else:
          for index in range(len(inputs)):
            ret += ("%s%s" % (decodes[index], self.decode_hp.delimiter))
        return ret

  def evaluate(self):
    """Run evaluation mode."""
    words, pronunciations = [], []
    for case in self.problem.generator(self.test_path,
                                       self.problem.source_vocab,
                                       self.problem.target_vocab):
      word = self.problem.source_vocab.decode(case["inputs"]).replace(
          EOS, "").strip()
      pronunciation = self.problem.target_vocab.decode(case["targets"]).replace(
          EOS, "").strip()
      words.append(word)
      pronunciations.append(pronunciation)

    self.g2p_gt_map = create_g2p_gt_map(words, pronunciations)

    if os.path.exists(self.frozen_graph_filename):
      with tf.Session(graph=self.graph) as sess:
        inp = tf.placeholder(tf.string, name="inp_decode")[0]
        decode_op = tf.py_func(self.calc_errors, [inp], [tf.int64, tf.int64])
        [correct, errors] = self.__run_op(sess, decode_op, self.test_path)

    else:
      correct, errors = self.calc_errors(self.test_path)

    print("Words: %d" % (correct+errors))
    print("Errors: %d" % errors)
    print("WER: %.3f" % (float(errors)/(correct+errors)))
    print("Accuracy: %.3f" % float(1.-(float(errors)/(correct+errors))))

  def freeze(self):
    """Freeze pre-trained model."""
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(self.params.model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    #output_graph = absolute_model_folder + "/frozen_model.pb"
    output_graph = self.frozen_graph_filename
    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what
    # part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = ["transformer/parallel_0_5/transformer/body/decoder/"
        "layer_0/self_attention/multihead_attention/dot_product_attention/"
        "Softmax",
                         "transformer/parallel_0_5/transformer/body/encoder/"
        "layer_0/self_attention/multihead_attention/dot_product_attention/"
        "Softmax",
                         "transformer/parallel_0_5/transformer/body/encoder/"
        "layer_1/self_attention/multihead_attention/dot_product_attention/"
        "Softmax",
                         "transformer/parallel_0_5/transformer/body/encoder/"
        "layer_2/self_attention/multihead_attention/dot_product_attention/"
        "Softmax",
                         "transformer/parallel_0_5/transformer/body/decoder/"
        "layer_0/encdec_attention/multihead_attention/dot_product_attention/"
        "Softmax",
                         "transformer/parallel_0_5/transformer/body/decoder/"
        "layer_1/self_attention/multihead_attention/dot_product_attention/"
        "Softmax",
                         "transformer/parallel_0_5/transformer/body/decoder/"
        "layer_1/encdec_attention/multihead_attention/dot_product_attention/"
        "Softmax",
                         "transformer/parallel_0_5/transformer/body/decoder/"
        "layer_2/self_attention/multihead_attention/dot_product_attention/"
        "Softmax",
                         "transformer/parallel_0_5/transformer/body/decoder/"
        "layer_2/encdec_attention/multihead_attention/dot_product_attention/"
        "Softmax"]

    # We clear devices to allow TensorFlow to control on which device it will
    # load operations
    clear_devices = True
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
      saver.restore(sess, input_checkpoint)

      # We use a built-in TF helper to export variables to constants
      output_graph_def = graph_util.convert_variables_to_constants(
          sess, # The session is used to retrieve the weights
          input_graph_def, # The graph_def is used to retrieve the nodes
          output_node_names, # The output node names are used to select the
                             #usefull nodes
          variable_names_blacklist=['global_step'])

      # Finally we serialize and dump the output graph to the filesystem
      with tf.gfile.GFile(output_graph, "wb") as output_graph_file:
        output_graph_file.write(output_graph_def.SerializeToString())
      print("%d ops in the final graph." % len(output_graph_def.node))

  def __load_graph(self):
    """Load freezed graph."""
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(self.frozen_graph_filename, "rb") as frozen_graph_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(frozen_graph_file.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as self.graph:
      # The name var will prefix every op/nodes in your graph
      # Since we load everything in a new graph, this is not needed
      tf.import_graph_def(graph_def, name="import")

  def __decode_from_file(self, filename):
    """Compute predictions on entries in filename and write them out."""

    if not self.decode_hp.batch_size:
      self.decode_hp.batch_size = 32
      tf.logging.info("decode_hp.batch_size not specified; default=%d" %
                      self.decode_hp.batch_size)

    problem_id = self.decode_hp.problem_idx
    # Inputs vocabulary is set to targets if there are no inputs in the problem,
    # e.g., for language models where the inputs are just a prefix of targets.
    inputs_vocab = self.hparams.problems[problem_id].vocabulary["inputs"]
    targets_vocab = self.hparams.problems[problem_id].vocabulary["targets"]
    problem_name = "grapheme_to_phoneme_problem"
    tf.logging.info("Performing decoding from a file.")
    inputs = _get_inputs(filename, '\n')
    num_decode_batches = (len(inputs) - 1) // self.decode_hp.batch_size + 1

    def input_fn():
      """Function for inputs generator."""
      input_gen = _decode_batch_input_fn(
          num_decode_batches, inputs, inputs_vocab,
          self.decode_hp.batch_size, self.decode_hp.max_input_size)
      gen_fn = decoding.make_input_fn_from_generator(input_gen)
      example = gen_fn()
      return decoding._decode_input_tensor_to_features_dict(example,
                                                            self.hparams)

    decodes = []
    result_iter = self.estimator.predict(input_fn)
    for result in result_iter:
      if self.decode_hp.return_beams:
        beam_decodes = []
        output_beams = np.split(result["outputs"], self.decode_hp.beam_size,
                                axis=0)
        for k, beam in enumerate(output_beams):
          tf.logging.info("BEAM %d:" % k)
          _, decoded_outputs, _ = decoding.log_decode_results(
              result["inputs"],
              beam,
              problem_name,
              None,
              inputs_vocab,
              targets_vocab)
          beam_decodes.append(decoded_outputs)
        decodes.append(beam_decodes)
      else:
        _, decoded_outputs, _ = decoding.log_decode_results(
            result["inputs"],
            result["outputs"],
            problem_name,
            None,
            inputs_vocab,
            targets_vocab)
        decodes.append(decoded_outputs)

    return [inputs, decodes]

  def calc_errors(self, decode_file_path):
    """Calculate a number of prediction errors."""
    inputs, decodes = self.__decode_from_file(decode_file_path)

    correct, errors = 0, 0
    for index, word in enumerate(inputs):
      if self.decode_hp.return_beams:
        beam_correct_found = False
        for beam_decode in decodes[index]:
          if beam_decode in self.g2p_gt_map[word]:
            beam_correct_found = True
            break
        if beam_correct_found:
          correct += 1
        else:
          errors += 1
      else:
        if decodes[index] in self.g2p_gt_map[word]:
          correct += 1
        else:
          errors += 1

    return correct, errors

##### G2P #####
def convert(text, retrieve_all=False, keep_punct=True, stress_marks='both'):
    """takes either a string or list of English words and converts them to ipa"""
    ipa = ipa_list(
                    words_in=text,
                    keep_punct=keep_punct,
                    stress_marks=stress_marks
                    )
    if retrieve_all:
        return get_all(ipa)
    return get_top(ipa)

def isin_cmu(word):
    """checks if a word is in the CMU dictionary. Doesn't strip punctuation.
    If given more than one word, returns True only if all words are present."""
    if type(word) == str:
        word = [preprocess(w) for w in word.split()]
    results = fetch_words(word)
    as_set = list(set(t[0] for t in results))
    return len(as_set) == len(set(word))

def ipa_list_from_outside(cmu_in, stress_marks='place'):
    """Returns a list of all the undiscovered ipa transcriptions for each word."""
    if type(cmu_in) == str:
        words = [[cmu_in.lower()]]
    else:
        for word in cmu_in:
            words.append(word.lower())
    ipa = cmu_to_ipa(words, stress_marking=stress_marks)
    return ipa

def get_all(ipa_list):
    """utilizes an algorithm to discover and return all possible combinations of ipa transcriptions"""
    final_size = 1
    for word_list in ipa_list:
        final_size *= len(word_list)
    list_all = ["" for s in range(final_size)]
    for i in range(len(ipa_list)):
        if i == 0:
            swtich_rate = final_size / len(ipa_list[i])
        else:
            swtich_rate /= len(ipa_list[i])
        k = 0
        for j in range(final_size):
            if (j+1) % int(swtich_rate) == 0:
                k += 1
            if k == len(ipa_list[i]):
                k = 0
            list_all[j] = list_all[j] + ipa_list[i][k] + " "
    return sorted([sent[:-1] for sent in list_all])

def get_top(ipa_list):
    """Returns only the one result for a query. If multiple entries for words are found, only the first is used."""
    return ' '.join([word_list[-1] for word_list in ipa_list])

def cmu_to_ipa(cmu_list, mark=True, stress_marking='both'):
    """converts the CMU word lists into ipa transcriptions"""
    #cmu_ipa_dict = {"ax": "ʌ", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
    #               "aw": "aʊ", "ay": "aɪ", "ch": "tʃ", "dh": "ð", "eh": "ɛ", "er": "ər",
    #               "hh": "h", "ih": "ɪ", "jh": "dʒ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
    #               "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j",
    #               "dx": "ɾ", "ix": "ɨ", "ux": "ʉ", "el": "əl", "em" : "əm",
    #               "en": "ən", "nx": "ɾ̃", "q": "ʔ", "wh": "ʍ"}

    #symbols = {"ax": "@", "ey": "e", "aa": "a", "ae": "{", "ah": "V", "ao": "O",
    #            "aw": "aU", "ay": "aI", "ch": "tS", "dh": "D", "eh": "E", "er": "@r",
    #            "hh": "h", "ih": "I", "jh": "dZ", "ng": "N",  "ow": "oU", "oy": "OI",
    #            "sh": "S", "th": "T", "uh": "U", "uw": "u", "zh": "Z", "iy": "i", "y": "j",
    #            "dx": "4", "ix": "1", "ux": "}", "b": "b", "d": "d", "el": "@l", "em" : "@m",
    #            "en": "@n", "f": "f", "g": "g", "k": "k", "l": "l", "m": "m", "n": "n",
    #            "nx": "r~", "p": "p", "q": "?", "r": "r", "s": "s", "t": "t", "v": "v",
    #            "w": "w", "wh": "W", "z": "z"}

    #cmu_xsampa_dict = {"ax": "V", "ey": "e", "aa": "a", "ae": "{", "ah": "@", "ao": "O",
    #            "aw": "aU", "ay": "aI", "ch": "tS", "dh": "D", "eh": "e", "er": "@r",
    #            "hh": "h", "ih": "I", "jh": "dZ", "ng": "N",  "ow": "oU", "oy": "OI",
    #            "sh": "S", "th": "T", "uh": "U", "uw": "u", "zh": "Z", "iy": "i", "y": "j",
    #            "dx": "4", "ix": "1", "ux": "}", "el": "@l", "em" : "@m",
    #            "en": "@n", "nx": "r~", "q": "?", "wh": "W"}

    cmu_ipa_dict = json.load(open(join(os.path.abspath(os.path.dirname(__file__)), 'resources', 'cmu_ipa.dict'), "r"))
    cmu_cz_dict = json.load(open(join(os.path.abspath(os.path.dirname(__file__)), 'resources', 'cmu_cz.dict'), "r"))

    ipa_list = []  # the final list of ipa tokens to be returned
    for word_list in cmu_list:
        ipa_word_list = {'transcription' : [], 'phones_map' : [[]]}  # the word list for each word
        for word in word_list[:1]:
            if stress_marking:
                if stress_marking == 'place':
                    word = place_stress(word)
                mapped_word = map_phones(word)
                word = find_stress(word, type=stress_marking)
            else:
                if re.sub("\d*", "", word.replace("__IGNORE__", "")) == "":
                    pass  # do not delete token if it's all numbers
                else:
                    word = re.sub("[0-9]", "", word)
            ipa_form = ''
            if word.startswith("__IGNORE__"):
                ipa_form = word.replace("__IGNORE__", "")
                # mark words we couldn't transliterate with an asterisk:

                if mark:
                    if not re.sub("\d*", "", ipa_form) == "":
                        ipa_form += "*"
            else:
                for i in range(len((mapped_word))):
                    try:
                        if mapped_word[i][0] in cmu_cz_dict:
                            mapped_word[i] = (cmu_cz_dict[mapped_word[i][0]], mapped_word[i][1])
                    except:
                        pass

                syllables = []
                i = 0
                syl_phones = ""
                syl_stress = -1
                for piece in word.split(" "):
                    marked = False
                    unmarked = piece
                    try:
                        if piece[0] in ["|", "ˈ", "ˌ"]:
                            if syl_phones != "":
                                syllables.append((syl_phones, syl_stress))
                                syl_phones = ""
                                syl_stress = -1
                            marked = True
                            stress_mark = piece[0]
                            unmarked = piece[1:]
                        if unmarked in cmu_ipa_dict:
                            if marked:
                                syl_phones += mapped_word[i][0]
                                syl_stress = max(syl_stress, mapped_word[i][1])
                                ipa_form += stress_mark + cmu_ipa_dict[unmarked]
                            else:
                                syl_phones += mapped_word[i][0]
                                syl_stress = max(syl_stress, mapped_word[i][1])
                                ipa_form += cmu_ipa_dict[unmarked]
                        else:
                            syl_phones += mapped_word[i][0]
                            syl_stress = max(syl_stress, mapped_word[i][1])
                            ipa_form += piece
                    except:
                        pass
                    i += 1
                if syl_phones != "":
                    syllables.append((syl_phones, syl_stress))

                ipa_word_list['phones_map'] = syllables
                
            swap_list = [["ˈər", "əˈr"], ["ˈie", "iˈe"]]
            for sym in swap_list:
                if not ipa_form.startswith(sym[0]):
                    ipa_form = ipa_form.replace(sym[0], sym[1])
            ipa_word_list['transcription'].append(ipa_form.replace('|',''))
        ipa_list.append(ipa_word_list)
    return ipa_list

def fetch_words(words_in):
    conn = sqlite3.connect(join(abspath(dirname(__file__)), "./resources/CMU_dict.db"))
    c = conn.cursor()

    """fetches a list of words from the database"""
    quest = "?, " * len(words_in)
    c.execute(f"SELECT word, phonemes FROM dictionary WHERE word IN ({quest[:-2]})", words_in)
    result = c.fetchall()
    d = defaultdict(list)
    for k, v in result:
        d[k].append(v)
    return list(d.items())

def preprocess(words):
    """Returns a string of words stripped of punctuation"""
    punct_str = '!"#$%&\'()*+,-./:;<=>/?@[\\]^_`{|}~«» '
    return ' '.join([w.strip(punct_str).lower() for w in words.split()])

def preserve_punc(words):
    """converts words to ipa and finds punctuation before and after the word."""
    words_preserved = []
    for w in words.split():
        punct_list = ["", preprocess(w), ""]
        before = re.search("^([^A-Za-z0-9]+)[A-Za-z]", w)
        after = re.search("[A-Za-z]([^A-Za-z0-9]+)$", w)
        if before:
            punct_list[0] = str(before.group(1))
        if after:
            punct_list[2] = str(after.group(1))
        words_preserved.append(punct_list)
    return words_preserved
    
def apply_punct(triple, as_str=False):
    """places surrounding punctuation back on center on a list of preserve_punc triples"""
    if type(triple[0]) == list:
        for i, t in enumerate(triple):
            triple[i] = str(''.join(triple[i]))
        if as_str:
            return ' '.join(triple)
        return triple
    if as_str:
        return str(''.join(t for t in triple))
    return [''.join(t for t in triple)]

def _punct_replace_word(original, transcription):
    """Get the ipa transcription of word with the original punctuation marks"""
    for i, trans_list in enumerate(transcription):
        for j, item in enumerate(trans_list):
            triple = [original[i][0]] + [item] + [original[i][2]]
            transcription[i][j] = apply_punct(triple, as_str=True)
    return transcription


def get_text(input_textfile_path = ""):
    """Get text from file or console in the interactive mode."""
    text = []
    if input_textfile_path != "":
        if os.path.exists(input_textfile_path):
            try:
                with open(os.path(input_textfile_path), 'r') as input_textfile:
                    while True:
                        line = oto_file.readline()
                        if not line:
                            text.append([None])
                            break
                        for sentence in sent_tokenize(line):
                            text.append(sentence)
            except EOFError:
               text.append([None])
    else:
        try:
            line = input("> ")
            for sentence in sent_tokenize(line):
                text.append(sentence)
        except KeyboardInterrupt:
            text.append([None])
    return text

def create_g2p_gt_map(words, pronunciations):
    """Create grapheme-to-phoneme ground true mapping."""
    g2p_gt_map = {}
    for word, pronunciation in zip(words, pronunciations):
        if word in g2p_gt_map:
            g2p_gt_map[word].append(pronunciation)
        else:
            g2p_gt_map[word] = [pronunciation]
    return g2p_gt_map

def _get_inputs(filename, delimiters="\t "):
    """Returning inputs.

    Args:
    filename: path to file with inputs, 1 per line.
    delimiters: str, delimits records in the file.

    Returns:
    a list of inputs

    """
    tf.logging.info("Getting inputs")
    delimiters_regex = re.compile("[" + delimiters + "]+")

    inputs = []
    with tf.gfile.Open(filename) as input_file:
        try:
            while True:
                line = f.readline()
                if not line:
                    break
                if set("[" + delimiters + "]+$").intersection(line):
                    items = re.split(delimiters_regex, line.strip(), maxsplit=1)
                    inputs.append(items[0])
                else:
                    inputs.append(line.strip())
        except:
            pass
    return inputs

def _decode_batch_input_fn(num_decode_batches, inputs, vocabulary, batch_size, max_input_size):
  """Decode batch"""
  tf.logging.info(" batch %d" % num_decode_batches)
  for batch_idx in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % batch_idx)
    batch_length = 0
    batch_inputs = []
    for _inputs in inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size]:
      input_ids = vocabulary.encode(_inputs)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      input_ids.append(text_encoder.EOS_ID)
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      encoded_input = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(encoded_input)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
        "problem_choice": np.array(0).astype(np.int32),
    }

def execute_schedule(exp, params):
  if not hasattr(exp, params.schedule):
    raise ValueError(
            "Experiment has no method %s, from --schedule" % params.schedule)
  with profile_context(params):
    getattr(exp, params.schedule)()

@contextlib.contextmanager
def profile_context(params):
  if params.profile:
    with tf.contrib.tfprof.ProfileContext("t2tprof",
            trace_steps=range(100),
            dump_steps=range(100)) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling("op", opts, range(100))
      yield
  else:
    yield


#####
#  g2p_encoder.py
#####

class GraphemePhonemeEncoder(text_encoder.TextEncoder):
  """Encodes each grapheme or phoneme to an id. For 8-bit strings only."""

  def __init__(self,
               vocab_filename=None,
               vocab_list=None,
               separator="",
               num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS):
    """Initialize from a file or list, one token per line.

    Handling of reserved tokens works as follows:
    - When initializing from a list, we add reserved tokens to the vocab.
    - When initializing from a file, we do not add reserved tokens to the vocab.
    - When saving vocab files, we save reserved tokens to the file.

    Args:
      vocab_filename: If not None, the full filename to read vocab from. If this
         is not None, then vocab_list should be None.
      vocab_list: If not None, a list of elements of the vocabulary. If this is
         not None, then vocab_filename should be None.
      separator: separator between symbols in original file.
      num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
    """
    super(GraphemePhonemeEncoder, self).__init__(
        num_reserved_ids=num_reserved_ids)
    if vocab_filename and os.path.exists(vocab_filename):
      self._init_vocab_from_file(vocab_filename)
    else:
      assert vocab_list is not None
      self._init_vocab_from_list(vocab_list)
    self._separator = separator

  def encode(self, symbols_line):
      #unicode -> bytes
    if isinstance(symbols_line, bytes):
      symbols_line = symbols_line.decode("utf-8")
    if self._separator:
      symbols_list = symbols_line.strip().split(self._separator)
    else:
      symbols_list = list(symbols_line.strip())
    ids_list = []
    for sym in symbols_list:
      if sym in self._sym_to_id:
        ids_list.append(self._sym_to_id[sym])
      else:
        tf.logging.warning("Invalid symbol:{}".format(sym))
    return ids_list

  def decode(self, ids):
    return self._separator.join(self.decode_list(ids))

  def decode_list(self, ids):
    return [self._id_to_sym[id_] for id_ in ids]

  @property
  def vocab_size(self):
    return len(self._id_to_sym)

  def _init_vocab_from_file(self, filename):
    """Load vocab from a file.

    Args:
      filename: The file to load vocabulary from.
    """
    def sym_gen():
      """Symbols generator for vocab initializer from file."""
      with tf.gfile.Open(filename) as vocab_file:
        for line in vocab_file:
          sym = line.strip()
          yield sym

    self._init_vocab(sym_gen(), add_reserved_symbols=False)

  def _init_vocab_from_list(self, vocab_list):
    """Initialize symbols from a list of symbols.

    It is ok if reserved symbols appear in the vocab list. They will be
    removed. The set of symbols in vocab_list should be unique.

    Args:
      vocab_list: A list of symbols.
    """
    def sym_gen():
      """Symbols generator for vocab initializer from list."""
      for sym in vocab_list:
        if sym not in text_encoder.RESERVED_TOKENS:
          yield sym

    self._init_vocab(sym_gen())

  def _init_vocab(self, sym_generator, add_reserved_symbols=True):
    """Initialize vocabulary with sym from sym_generator."""

    self._id_to_sym = {}
    non_reserved_start_index = 0

    if add_reserved_symbols:
      self._id_to_sym.update(enumerate(text_encoder.RESERVED_TOKENS))
      non_reserved_start_index = len(text_encoder.RESERVED_TOKENS)

    self._id_to_sym.update(
        enumerate(sym_generator, start=non_reserved_start_index))

    # _sym_to_id is the reverse of _id_to_sym
    self._sym_to_id = dict((v, k) for k, v in six.iteritems(self._id_to_sym))

  def store_to_file(self, filename):
    """Write vocab file to disk.

    Vocab files have one symbol per line. The file ends in a newline. Reserved
    symbols are written to the vocab file as well.

    Args:
      filename: Full path of the file to store the vocab to.
    """
    with tf.gfile.Open(filename, "w") as vocab_file:
      #xrange->range
      for i in range(len(self._id_to_sym)):
        vocab_file.write(self._id_to_sym[i] + "\n")

def build_vocab_list(data_path, init_vocab_list=[]):
  """Reads a file to build a vocabulary with letters and phonemes.

    Args:
      data_path: data file to read list of words from.

    Returns:
      vocab_list: vocabulary list with both graphemes and phonemes."""
  vocab = {item:1 for item in init_vocab_list}
  with tf.gfile.GFile(data_path, "r") as data_file:
    for line in data_file:
      items = line.strip().split()
      vocab.update({char:1 for char in list(items[0])})
      vocab.update({phoneme:1 for phoneme in items[1:]})
    vocab_list = [PAD, EOS]
    for key in sorted(vocab.keys()):
      vocab_list.append(key)
  return vocab_list

def load_create_vocabs(vocab_filename, train_path=None, dev_path=None, test_path=None):
  """Load/create vocabularies."""
  vocab = None
  if os.path.exists(vocab_filename):
    source_vocab = GraphemePhonemeEncoder(vocab_filename=vocab_filename)
    target_vocab = GraphemePhonemeEncoder(vocab_filename=vocab_filename,
        separator=" ")
  else:
    vocab_list = []
    for data_path in [train_path, dev_path, test_path]:
      vocab_list = build_vocab_list(data_path, vocab_list)
    source_vocab = GraphemePhonemeEncoder(vocab_list=vocab_list)
    target_vocab = GraphemePhonemeEncoder(vocab_list=vocab_list,
        separator=" ")
    source_vocab.store_to_file(vocab_filename)

  return source_vocab, target_vocab


#####
#  g2p_problem.py
#####

EOS = text_encoder.EOS_ID

@registry.register_problem
class GraphemeToPhonemeProblem(text_problems.Text2TextProblem):
  """Problem spec for cmudict PRONALSYL Grapheme-to-Phoneme translation."""

  def __init__(self, model_dir, train_path=None, dev_path=None, test_path=None,
               cleanup=False):
    """Create a Problem.

    Args:
      was_reversed: bool, whether to reverse inputs and targets.
      was_copy: bool, whether to copy inputs to targets. Can be composed with
        was_reversed so that if both are true, the targets become the inputs,
        which are then copied to targets so that the task is targets->targets.
    """
    super(GraphemeToPhonemeProblem, self).__init__()
    self._encoders = None
    self._hparams = None
    self._feature_info = None
    self._model_dir = model_dir
    self.train_path, self.dev_path, self.test_path = train_path, dev_path,\
        test_path
    vocab_filename = join(self._model_dir, "vocab.g2p")
    if train_path:
      self.train_path, self.dev_path, self.test_path = create_data_files(
          init_train_path=train_path, init_dev_path=dev_path,
          init_test_path=test_path,cleanup=cleanup)
      self.source_vocab, self.target_vocab = load_create_vocabs(
          vocab_filename, train_path=self.train_path, dev_path=self.dev_path,
          test_path=self.test_path)
    elif not os.path.exists(join(self._model_dir, "checkpoint")):
      raise StandardError("Model not found in {}".format(self._model_dir))
    else:
      self.source_vocab, self.target_vocab = load_create_vocabs(
          vocab_filename)

  def generator(self, data_path, source_vocab, target_vocab):
    """Generator for the training and evaluation data.
    Generate source and target data from a single file.

    Args:
      data_path: The path to data file.
      source_vocab: the object of GraphemePhonemeEncoder class with encode and
        decode functions for symbols from source file.
      target_vocab: the object of GraphemePhonemeEncoder class with encode and
        decode functions for symbols from target file.

    Yields:
      dicts with keys "inputs" and "targets", with values being lists of token
      ids.
    """
    return self.tabbed_generator(data_path, source_vocab, target_vocab, EOS)

  def filepattern(self, data_dir, dataset_split, shard=None):
    if not (".preprocessed" in dataset_split):
      return join(self._model_dir, dataset_split + ".preprocessed")
    return join(data_dir, dataset_split)

  @property
  def input_space_id(self):
    return 0

  @property
  def target_space_id(self):
    return 0

  @property
  def num_shards(self):
    return 1

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def is_character_level(self):
    return False

  @property
  def targeted_vocab_size(self):
    return None

  @property
  def vocab_name(self):
    return None

  def generate_preprocess_data(self):
    """Generate and save preprocessed data as TFRecord files.

    Args:
      train_path: the path to the train data file.
      eval_path: the path to the evaluation data file.

    Returns:
      train_preprocess_path: the path where the preprocessed train data
          was saved.
      eval_preprocess_path: the path where the preprocessed evaluation data
          was saved.
    """
    train_preprocess_path = join(self._model_dir, "train.preprocessed")
    eval_preprocess_path = join(self._model_dir, "eval.preprocessed")
    train_gen = self.generator(self.train_path, self.source_vocab,
                               self.target_vocab)
    eval_gen = self.generator(self.dev_path, self.source_vocab,
                              self.target_vocab)

    generate_preprocess_files(train_gen, eval_gen, train_preprocess_path,
                              eval_preprocess_path)
    return train_preprocess_path, eval_preprocess_path

  def get_feature_encoders(self, data_dir=None):
    if self._encoders is None:
      self._encoders = self.feature_encoders()
    return self._encoders

  def feature_encoders(self):
    targets_encoder = self.target_vocab
    if self.has_inputs:
      inputs_encoder = self.source_vocab
      return {"inputs": inputs_encoder, "targets": targets_encoder}
    return {"targets": targets_encoder}

  def tabbed_generator(self, source_path, source_vocab, target_vocab, eos=None):
    r"""Generator for sequence-to-sequence tasks using tabbed files.

    Tokens are derived from text files where each line contains both
    a source and a target string. The two strings are separated by a tab
    character ('\t'). It yields dictionaries of "inputs" and "targets" where
    inputs are characters from the source lines converted to integers, and
    targets are characters from the target lines, also converted to integers.

    Args:
      source_path: path to the file with source and target sentences.
      source_vocab: a SubwordTextEncoder to encode the source string.
      target_vocab: a SubwordTextEncoder to encode the target string.
      eos: integer to append at the end of each sequence (default: None).
    Yields:
      A dictionary {"inputs": source-line, "targets": target-line} where
      the lines are integer lists converted from characters in the file lines.
    """
    eos_list = [] if eos is None else [eos]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
      for line_idx, line in enumerate(source_file):
        if line:
          source, target = split_graphemes_phonemes(line)
          if not (source and target):
            tf.logging.warning("Invalid data format in line {} in {}:\n"
                "{}\nGraphemes and phonemes should be separated by white space."
                .format(line_idx, source_path, line))
            continue
          source_ints = source_vocab.encode(source) + eos_list
          target_ints = target_vocab.encode(target) + eos_list
          yield {"inputs": source_ints, "targets": target_ints}


  def dataset(self,
              mode,
              data_dir=None,
              num_threads=None,
              output_buffer_size=None,
              shuffle_files=None,
              hparams=None,
              preprocess=True,
              dataset_split=None,
              shard=None,
              partition_id=0,
              num_partitions=1):
    """Build a Dataset for this problem.

    Args:
      mode: tf.estimator.ModeKeys; determines which files to read from.
      data_dir: directory that contains data files.
      num_threads: int, number of threads to use for decode and preprocess
        Dataset.map calls.
      output_buffer_size: int, how many elements to prefetch in Dataset.map
        calls.
      shuffle_files: whether to shuffle input files. Default behavior (i.e. when
        shuffle_files=None) is to shuffle if mode == TRAIN.
      hparams: tf.contrib.training.HParams; hparams to be passed to
        Problem.preprocess_example and Problem.hparams. If None, will use a
        default set that is a no-op.
      preprocess: bool, whether to map the Dataset through
        Problem.preprocess_example.
      dataset_split: tf.estimator.ModeKeys + ["test"], which split to read data
        from (TRAIN:"-train", EVAL:"-dev", "test":"-test"). Defaults to mode.
      shard: int, if provided, will only read data from the specified shard.

    Returns:
      Dataset containing dict<feature name, Tensor>.
    """
    if dataset_split or (mode in ["train", "eval"]):
      # In case when pathes to preprocessed files pointed out or if train mode
      # launched, we save preprocessed data first, and then create dataset from
      # that files.
      dataset_split = dataset_split or mode
      assert data_dir

      if not hasattr(hparams, "data_dir"):
        hparams.add_hparam("data_dir", data_dir)
      if not hparams.data_dir:
        hparams.data_dir = data_dir
      # Construct the Problem's hparams so that items within it are accessible
      _ = self.get_hparams(hparams)

      data_fields, data_items_to_decoders = self.example_reading_spec()
      if data_items_to_decoders is None:
        data_items_to_decoders = {
            field: tf.contrib.slim.tfexample_decoder.Tensor(field)
            for field in data_fields}

      is_training = mode == tf.estimator.ModeKeys.TRAIN
      data_filepattern = self.filepattern(data_dir, dataset_split, shard=shard)
      tf.logging.info("Reading data files from %s", data_filepattern)
      data_files = tf.contrib.slim.parallel_reader.get_data_files(
          data_filepattern)
      if shuffle_files or shuffle_files is None and is_training:
        random.shuffle(data_files)

    else:
      # In case when pathes to preprocessed files not pointed out, we create
      # dataset from generator object.
      eos_list = [] if EOS is None else [EOS]
      data_list = []
      with tf.gfile.GFile(self.test_path, mode="r") as source_file:
        for line in source_file:
          if line:
            if "\t" in line:
              parts = line.split("\t", 1)
              source, target = parts[0].strip(), parts[1].strip()
              source_ints = self.source_vocab.encode(source) + eos_list
              target_ints = self.target_vocab.encode(target) + eos_list
              data_list.append({"inputs":source_ints, "targets":target_ints})
            else:
              source_ints = self.source_vocab.encode(line) + eos_list
              data_list.append(generator_utils.to_example(
                  {"inputs":source_ints}))

      gen = Gen(self.generator(self.test_path, self.source_vocab,
                               self.target_vocab))
      dataset = dataset_ops.Dataset.from_generator(gen, tf.string)

      preprocess = False

    def decode_record(record):
      """Serialized Example to dict of <feature name, Tensor>."""
      decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
          data_fields, data_items_to_decoders)

      decode_items = list(data_items_to_decoders)
      decoded = decoder.decode(record, items=decode_items)
      return dict(zip(decode_items, decoded))

    def _preprocess(example):
      """Whether preprocess data into required format."""
      example = self.preprocess_example(example, mode, hparams)
      self.maybe_reverse_features(example)
      self.maybe_copy_features(example)
      return example

    dataset = (tf.data.Dataset.from_tensor_slices(data_files)
               .interleave(lambda x:
                   tf.data.TFRecordDataset(x).map(decode_record,
                                                  num_parallel_calls=4),
                   cycle_length=4, block_length=16))

    if preprocess:
      dataset = dataset.map(_preprocess, num_parallel_calls=4)

    return dataset

class Gen:
  """Generator class for dataset creation.
  Function dataset_ops.Dataset.from_generator() required callable generator
  object."""

  def __init__(self, gen):
    """ Initialize generator."""
    self._gen = gen

  def __call__(self):
    for case in self._gen:
      source_ints = case["inputs"]
      target_ints = case["targets"]
      yield generator_utils.to_example({"inputs":source_ints,
                                        "targets":target_ints})

def generate_preprocess_files(train_gen, dev_gen, train_preprocess_path, dev_preprocess_path):
  """Generate cases from a generators and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    train_gen: a generator yielding (string -> int/float/str list) train data.
    dev_gen: a generator yielding development data.
    train_preprocess_path: path to the file where preprocessed train data
        will be saved.
    dev_preprocess_path: path to the file where preprocessed development data
        will be saved.
  """
  if dev_gen:
    gen_file(train_gen, train_preprocess_path)
    gen_file(dev_gen, dev_preprocess_path)
  else:
    # In case when development generator was not given, we create development
    # preprocess file from train generator.
    train_writer = tf.python_io.TFRecordWriter(train_preprocess_path)
    dev_writer = tf.python_io.TFRecordWriter(dev_preprocess_path)
    line_counter = 1
    for case in train_gen:
      sequence_example = generator_utils.to_example(case)
      if line_counter % 20 == 0:
        dev_writer.write(sequence_example.SerializeToString())
      else:
        train_writer.write(sequence_example.SerializeToString())
      line_counter += 1
    train_writer.close()
    dev_writer.close()

def gen_file(generator, output_file_path):
  """Generate cases from generator and save as TFRecord file.

  Args:
    generator: a generator yielding (string -> int/float/str list) data.
    output_file_path: path to the file where preprocessed data will be saved.
  """
  writer = tf.python_io.TFRecordWriter(output_file_path)
  for case in generator:
    sequence_example = generator_utils.to_example(case)
    writer.write(sequence_example.SerializeToString())
  writer.close()

def create_data_files(init_train_path, init_dev_path, init_test_path, cleanup=False):
  """Create train, development and test data files from initial data files
  in case when not provided development or test data files or active cleanup
  flag.

  Args:
    init_train_path: path to the train data file.
    init_dev_path: path to the development data file.
    init_test_path: path to the test data file.
    cleanup: flag indicating whether to cleanup datasets from stress and
             comments.

  Returns:
    train_path: path to the new train data file generated from initially
      provided data.
    dev_path: path to the new development data file generated from initially
      provided data.
    test_path: path to the new test data file generated from initially
      provided data.
  """
  train_path, dev_path, test_path = init_train_path, init_dev_path,\
      init_test_path

  if (init_dev_path and init_test_path and os.path.exists(init_dev_path) and
      os.path.exists(init_test_path)):
    if not cleanup:
      return init_train_path, init_dev_path, init_test_path

  else:
    train_path = init_train_path + ".part.train"
    if init_dev_path:
      if not os.path.exists(init_dev_path):
        raise IOError("File {} not found.".format(init_dev_path))
    else:
      dev_path = init_train_path + ".part.dev"

    if init_test_path:
      if not os.path.exists(init_test_path):
        raise IOError("File {} not found.".format(init_test_path))
    else:
      test_path = init_train_path + ".part.test"

  if cleanup:
    train_path += ".cleanup"
    dev_path += ".cleanup"
    test_path += ".cleanup"

  train_dic, dev_dic, test_dic = OrderedDict(), OrderedDict(), OrderedDict()

  source_dic = collect_pronunciations(source_path=init_train_path,
                                      cleanup=cleanup)
  if init_dev_path:
    dev_dic = collect_pronunciations(source_path=init_dev_path,
                                     cleanup=cleanup)
  if init_test_path:
    test_dic = collect_pronunciations(source_path=init_test_path,
                                      cleanup=cleanup)

  #Split dictionary to train, validation and test (if not assigned).
  for word_counter, (word, pronunciations) in enumerate(source_dic.items()):
    if word_counter % 20 == 19 and not init_dev_path:
      dev_dic[word] = pronunciations
    elif ((word_counter % 20 == 18 or word_counter % 20 == 17) and
          not init_test_path):
      test_dic[word] = pronunciations
    else:
      train_dic[word] = pronunciations

  save_dic(train_dic, train_path)
  if not init_dev_path or cleanup:
    save_dic(dev_dic, dev_path)
  if not init_test_path or cleanup:
    save_dic(test_dic, test_path)
  return train_path, dev_path, test_path

def collect_pronunciations(source_path, cleanup=False):
  """Create dictionary mapping word to its different pronounciations.

  Args:
    source_path: path to the data file;
    cleanup: flag indicating whether to cleanup datasets from stress and
             comments.

  Returns:
    dic: dictionary mapping word to its pronunciations.
  """
  dic = OrderedDict()
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    word_counter = 0
    for line in source_file:
      if line:
        source, target = split_graphemes_phonemes(line, cleanup=cleanup)
        if not (source, target):
          tf.logging.warning("Invalid data format in line {} in {}:\n"
              "{}\nGraphemes and phonemes should be separated by white space."
              .format(line_idx, source_path, line))
          continue
        if source in dic:
          dic[source].append(target)
        else:
          dic[source] = [target]
  return dic

def split_graphemes_phonemes(input_line, cleanup=False):
  """Split line into graphemes and phonemes.

  Args:
    input_line: raw input line;
    cleanup: flag indicating whether to cleanup datasets from stress and
             comments.

  Returns:
    graphemes: graphemes string;
    phonemes: phonemes string.
  """
  line = input_line
  if cleanup:
    clean_pattern = re.compile(r"(\[.*\]|\{.*\}|\(.*\)|#.*)")
    stress_pattern = re.compile(r"(?<=[a-zA-Z])\d+")
    line = re.sub(clean_pattern, r"", line)
    line = re.sub(stress_pattern, r"", line)

  items = line.split()
  graphemes, phonemes = None, None
  if len(items) > 1:
    graphemes, phonemes = items[0].strip(), " ".join(items[1:]).strip()
  return graphemes, phonemes

def save_dic(dic, save_path):
  with tf.gfile.GFile(save_path, mode="w") as save_file:
    for word, pronunciations in dic.items():
      for pron in pronunciations:
        save_file.write(word + " " + pron + "\n")


#####
#  g2p_trainer_utils.py
#####

flags = tf.flags
FLAGS = flags.FLAGS

def add_problem_hparams(hparams, problem_name, model_dir, problem_instance):
  """Add problem hparams for the problems."""
  hparams.problems = []
  hparams.problem_instances = []
  p_hparams = problem_instance.get_hparams(hparams)
  hparams.problem_instances.append(problem_instance)
  hparams.problems.append(p_hparams)

def create_experiment_fn(params, problem_instance):
  use_validation_monitor = (params.schedule in
                            ["train_and_evaluate", "continuous_train_and_eval"]
                            and params.local_eval_frequency)
  return create_experiment_func(
      model_name=params.model_name,
      params=params,
      problem_instance=problem_instance,
      data_dir=os.path.expanduser(params.data_dir_name),
      train_steps=params.train_steps,
      eval_steps=params.eval_steps,
      min_eval_frequency=params.local_eval_frequency,
      schedule=params.schedule,
      export=params.export_saved_model,
      decode_hparams=decoding.decode_hparams(params.decode_hparams),
      use_tfdbg=params.tfdbg,
      use_dbgprofile=params.dbgprofile,
      use_validation_monitor=use_validation_monitor,
      eval_early_stopping_steps=params.eval_early_stopping_steps,
      eval_early_stopping_metric=params.eval_early_stopping_metric,
      eval_early_stopping_metric_minimize=\
        params.eval_early_stopping_metric_minimize,
      use_tpu=params.use_tpu)

def create_experiment_func(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn

def create_experiment(run_config,
                      hparams,
                      model_name,
                      params,
                      problem_instance,
                      data_dir,
                      train_steps,
                      eval_steps,
                      min_eval_frequency=2000,
                      schedule="train_and_evaluate",
                      export=False,
                      decode_hparams=None,
                      use_tfdbg=False,
                      use_dbgprofile=False,
                      use_validation_monitor=False,
                      eval_early_stopping_steps=None,
                      eval_early_stopping_metric=None,
                      eval_early_stopping_metric_minimize=True,
                      use_tpu=False):
  """Create Experiment."""
  # HParams
  hparams.add_hparam("data_dir", data_dir)
  add_problem_hparams(hparams, params.problem_name, params.model_dir, problem_instance)

  # Estimator
  estimator = trainer_lib.create_estimator(
      model_name,
      hparams,
      run_config,
      schedule=schedule,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu)

  # Input fns from Problem
  problem = hparams.problem_instances[0]
  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams)
  eval_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL, hparams)

  # Export
  export_strategies = export and [create_export_strategy(problem, hparams)]

  # Hooks
  hooks_kwargs = {}
  if not use_tpu:
    dbgprofile_kwargs = {"output_dir": run_config.model_dir}
    validation_monitor_kwargs = dict(
        input_fn=eval_input_fn,
        eval_steps=eval_steps,
        every_n_steps=min_eval_frequency,
        early_stopping_rounds=eval_early_stopping_steps,
        early_stopping_metric=eval_early_stopping_metric,
        early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
    train_monitors, eval_hooks = trainer_lib.create_hooks(
        use_tfdbg=use_tfdbg,
        use_dbgprofile=use_dbgprofile,
        dbgprofile_kwargs=dbgprofile_kwargs,
        use_validation_monitor=use_validation_monitor,
        validation_monitor_kwargs=validation_monitor_kwargs)
    hooks_kwargs = {"train_monitors": train_monitors, "eval_hooks": eval_hooks}

  # Experiment
  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps,
      min_eval_frequency=min_eval_frequency,
      train_steps_per_iteration=min(min_eval_frequency, train_steps),
      export_strategies=export_strategies,
      **hooks_kwargs)

def create_run_config(hp, params):
  return trainer_lib.create_run_config(
      model_dir=params.model_dir,
      master=params.master,
      iterations_per_loop=params.iterations_per_loop,
      num_shards=params.tpu_num_shards,
      log_device_placement=params.log_device_replacement,
      save_checkpoints_steps=max(params.iterations_per_loop,
                                 params.local_eval_frequency),
      keep_checkpoint_max=params.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=params.keep_checkpoint_every_n_hours,
      num_gpus=params.worker_gpu,
      gpu_order=params.gpu_order,
      shard_to_cpu=params.locally_shard_to_cpu,
      num_async_replicas=params.worker_replicas,
      gpu_mem_fraction=params.worker_gpu_memory_fraction,
      enable_graph_rewriter=params.experimental_optimize_placement,
      use_tpu=params.use_tpu,
      schedule=params.schedule,
      no_data_parallelism=params.no_data_parallelism,
      daisy_chain_variables=params.daisy_chain_variables,
      ps_replicas=params.ps_replicas,
      ps_job=params.ps_job,
      ps_gpu=params.ps_gpu,
      sync=params.sync,
      worker_id=params.worker_id,
      worker_job=params.worker_job)

def save_params(model_dir, hparams):
  """Save customizable model parameters in 'model.params' file.
  """
  params_to_save = {}
  for hp in hparams.split(","):
    param_split = hp.split("=")
    if len(param_split) == 2:
      param_name, param_value = param_split[0], param_split[1]
      params_to_save[param_name] = param_value
    else:
      raise ValueError("HParams line:{} can not be splitted\n"
                       .format(param_split))
  with open(join(model_dir, "model.params"), "w") as params_file:
    json.dump(params_to_save, params_file)

def load_params(model_dir):
  """Load customizable parameters from 'model.params' file.
  """
  params_file_path = join(model_dir, "model.params")
  if os.path.exists(params_file_path):
    model_params = json.load(open(params_file_path))
    hparams = ""
    for hp, hp_value in model_params.items():
      if hparams:
        hparams += ","
      hparams += hp + "=" + hp_value
    return hparams
  raise StandardError("File {} not exists.".format(params_file_path))


#####
#  g2p_params.py
#####

class Params(object):
  """Class with training parameters."""
  def __init__(self, model_dir, data_path, flags=None):
    self.model_dir = os.path.expanduser(model_dir)
    self.data_dir_name = os.path.dirname(data_path)
    # Set default parameters first. Then update the parameters that
    # pointed out in flags.
    self.hparams_set = "transformer_base"
    self.schedule = "train_and_evaluate"
    self.model_name = "transformer"
    self.problem_name = "grapheme_to_phoneme_problem"
    self.train_steps = 10
    self.eval_steps = 1
    self.iterations_per_loop = 2
    self.local_eval_frequency = 5
    self.hparams = "eval_drop_long_sequences=1,batch_size=1," +\
        "num_hidden_layers=1,hidden_size=4,filter_size=8,num_heads=1," +\
        "length_bucket_step=2.0,max_length=10,min_length_bucket=5"
    self.decode_hparams = "beam_size=1,alpha=0.6,return_beams=False"
    self.master = ""

    if flags:
      self.batch_size = flags.batch_size
      self.iterations_per_loop = min(1000, max(10, int(self.batch_size/10)))
      if flags.max_epochs > 0:
        self.train_steps = max(10000, 
                               int(len(open(data_path).readlines()) /\
                                   self.batch_size) *\
                               self.iterations_per_loop *\
                               flags.max_epochs)
      elif flags.train:
        self.train_steps = 200000

      self.eval_steps = min(200, int(self.train_steps/1000))
      self.local_eval_frequency = min(2000, max(20, int(self.train_steps/100)))

      self.hparams = "eval_drop_long_sequences=1" +\
          ",batch_size=" + str(flags.batch_size) +\
          ",num_hidden_layers=" + str(flags.num_layers) +\
          ",hidden_size=" + str(flags.size) +\
          ",filter_size=" + str(flags.filter_size) +\
          ",num_heads=" + str(flags.num_heads) +\
          ",length_bucket_step=" + str(flags.length_bucket_step) +\
          ",max_length=" + str(flags.max_length) +\
          ",min_length_bucket=" + str(flags.min_length_bucket)
      self.decode_hparams = "beam_size=" + str(flags.beam_size) +\
          ",alpha=" + str(flags.alpha)
      if flags.return_beams:
          self.decode_hparams += ",return_beams=True"
      else:
          self.decode_hparams += ",return_beams=False"

    self.tpu_num_shards = 8
    self.log_device_replacement = False
    self.keep_checkpoint_max = 1
    self.keep_checkpoint_every_n_hours = 1
    self.worker_gpu = 1
    self.gpu_order = ""
    self.locally_shard_to_cpu = False
    self.worker_replicas = 1
    self.worker_gpu_memory_fraction = 0.95
    self.experimental_optimize_placement = False
    self.use_tpu = False
    self.no_data_parallelism = False
    self.daisy_chain_variables = True
    self.ps_replicas = 0
    self.ps_job = "/job:ps"
    self.ps_gpu = 0
    self.sync = False
    self.worker_id = 0
    self.worker_job = "/job:localhost"
    self.export_saved_model = False
    self.tfdbg = False
    self.dbgprofile = False
    self.eval_early_stopping_steps = None
    self.eval_early_stopping_metric = "loss"
    self.eval_early_stopping_metric_minimize = True
    self.profile = False
    self.decode_shards = 1

    saved_hparams_path = join(self.model_dir, "hparams.json")
    if os.path.exists(saved_hparams_path):
      saved_hparams_dic = json.load(open(saved_hparams_path))
      self.hparams = ""
      for hparam_idx, (hparam, hparam_value) in enumerate(
          saved_hparams_dic.items()):
        self.hparams += hparam + "=" + str(hparam_value)
        if hparam_idx < len(saved_hparams_dic) - 1:
          self.hparams += ","


#####
#  cmu_to_ipa module
#####
#####
#  stress.py
#####

with open(join(os.path.abspath(os.path.dirname(__file__)), 'resources', 'phones.json'), "r") as phones_json:
    phones = json.load(phones_json)

def create_phones_json():
    """Creates the phones.json file in the resources directory from the phones.txt source file from CMU"""
    phones_dict = {}
    with open(join(os.path.abspath(os.path.dirname(__file__)),
                           'resources','CMU_source_files','cmudict-0.7b.phones.txt'), encoding="UTF-8") as phones_txt:
        # source link: http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.phones
        for line in phones_txt.readlines():
            phones_dict[line.split("	")[0].lower()] = line.split("	")[1].replace("\n", "")

    with open(join(os.path.abspath(os.path.dirname(__file__)),
                           'resources','phones.json'), "w") as phones_json:
        json.dump(phones_dict, phones_json)

def stress_type(stress):
    """Determine the kind of stress that should be evaluated"""
    stress = stress.lower()
    default = {"0": '|', "1": "ˈ", "2": "ˌ"}
    if stress == "primary":
        return {"0": '|', "1": "ˈ"}
    elif stress == "secondary":
        return {"0": '|', "2": "ˌ"}
    elif stress == "both" or stress == "place":
        return default
    else:
        logging.warning("WARNING: stress type parameter " + stress + " not recognized.")

    #default = {"1": "ˈ", "2": "ˌ"}
    #if stress == "primary":
    #    return {"1": "ˈ"}
    #elif stress == "secondary":
    #    return {"2": "ˌ"}
    #elif stress == "both" or stress == "place":
    #    return default
    #else:
    #    logging.warning("WARNING: stress type parameter " + stress + " not recognized.")
        # Use default stress
        return default

def place_stress(word):
    symbols = word.split(' ')
    new_word = []
    for c in symbols:
        try:
            if not c[-1] in ['0', '1', '2']:
                if phones[c] == "vowel":
                    c += '0'
        except:
            pass
        new_word.append(c)
    return ' '.join(new_word)

def map_phones(word):
    symbols = word.split(' ')
    mapped_word = []
    s = -1
    for c in symbols:
        try:
            if c[-1] in ['0', '1', '2']:
                if phones[c[:-1]] == "vowel":
                    s = int(c[-1]) # Marks vowels with their stress level
                    c = c[:-1]
            else:
                s = -1 # Marks consonants
        except:
            s = -1
        mapped_word.append((c, s))
    return mapped_word

def find_stress(word, type="all"):
    """Convert stress marking numbers from CMU into actual stress markings
    :param word: the CMU word string to be evaluated for stress markings
    :param type: type of stress to be evaluated (primary, secondary, or both)"""

    syll_count = cmu_syllable_count(word)

    if (not word.startswith("__IGNORE__")) and syll_count > 1:
        symbols = word.split(' ')
        stress_map = stress_type(type)
        new_word = []
        syllables = []
        clusters = ["sp", "st", "sk", "fr", "fl"]
        stop_set = ["nasal", "fricative", "vowel"]  # stop searching for where stress starts if these are encountered
        # for each CMU symbol
        for c in symbols:
            # if the last character is a 1 or 2 (that means it has stress, and we want to evaluate it)
            if c[-1] in stress_map.keys():
                # if the new_word list is empty
                if not new_word:
                    # append to new_word the CMU symbol, replacing numbers with stress marks
                    new_word.append(re.sub("\d", "", stress_map[re.findall("\d", c)[0]] + c))
                else:
                    stress_mark = stress_map[c[-1]]
                    placed = False
                    hiatus = False
                    new_word = new_word[::-1]  # flip the word and backtrack through symbols
                    for i, sym in enumerate(new_word):
                        sym = re.sub("[0-9|ˈˌ]", "", sym)
                        prev_sym = re.sub("[0-9|ˈˌ]", "", new_word[i-1])
                        prev_phone = phones[re.sub("[0-9|ˈˌ]", "", new_word[i-1])]
                        #sym = re.sub("[0-9ˈˌ]", "", sym)
                        #prev_sym = re.sub("[0-9ˈˌ]", "", new_word[i-1])
                        #prev_phone = phones[re.sub("[0-9ˈˌ]", "", new_word[i-1])]
                        if phones[sym] in stop_set or (i > 0 and prev_phone == "stop") or sym in ["er", "w", "j"]:
                            if sym + prev_sym in clusters:
                                new_word[i] = stress_mark + new_word[i]
                            elif not prev_phone == "vowel" and i > 0:
                                new_word[i-1] = stress_mark + new_word[i-1]
                            else:
                                if phones[sym] == "vowel":
                                    hiatus = True
                                    new_word = [stress_mark + re.sub("[0-9|ˈˌ]", "", c)] + new_word
                                    #new_word = [stress_mark + re.sub("[0-9|ˈˌ]", "", c)] + new_word
                                else:
                                    new_word[i] = stress_mark + new_word[i]
                            placed = True
                            break
                    if not placed:
                        if new_word:
                            new_word[len(new_word) - 1] = stress_mark + new_word[len(new_word) - 1]
                    new_word = new_word[::-1]
                    if not hiatus:
                        new_word.append(re.sub("\d", "", c))
                        hiatus = False
            else:
                if c.startswith("__IGNORE__"):
                    new_word.append(c)
                else:
                    new_word.append(re.sub("\d", "", c))

        return ' '.join(new_word)
    else:
        if word.startswith("__IGNORE__"):
            return word
        else:
            return re.sub("[0-9]", "", word)


#####
#  syllables.py
#####

with open(join(os.path.abspath(os.path.dirname(__file__)), 'resources','phones.json'), "r") as phones_json:
    PHONES = json.load(phones_json)

# list of adjacent vowel symbols that constitute separate nuclei
hiatus = [["er", "iy"], ["iy", "ow"], ["uw", "ow"], ["iy", "ah"], ["iy", "ey"], ["uw", "eh"], ["er", "eh"]]

def cmu_syllable_count(word):
    """count syllables based on CMU transcription"""
    word = re.sub("\d", "", word).split(' ')
    if "__IGNORE__" in word[0]:
        return 0
    else:
        nuclei = 0
        for i, sym in enumerate(word):
            try:
                prev_phone = PHONES[word[i-1]]
                prev_sym = word[i-1]
                if PHONES[sym] == 'vowel':
                    if i > 0 and not prev_phone == 'vowel' or i == 0:
                        nuclei += 1
                    elif [prev_sym, sym] in hiatus:
                        nuclei += 1
            except KeyError:
                pass
        return nuclei

def syllable_count(word: str):
    """transcribes a regular word to CMU to fetch syllable count"""
    if len(word.split()) > 1:
        return [syllable_count(w) for w in word.split()]
    word = G2pModel.get_cmu([G2pModel.preprocess(word)])
    return cmu_syllable_count(word[0][0])

def chartype_tokenize(word):
    cur_type = 'a'
    tokens = ['']
    for char in word:
        if char.isalpha():
            if cur_type != 'a':
                tokens.append(char)
            else:
                tokens[-1] += char
            cur_type = 'a'
        elif char.isdigit():
            if cur_type != 'd':
                tokens.append(char)
            else:
                tokens[-1] += char
            cur_type = 'd'
        else:
            if cur_type != 'o':
                tokens.append(char)
            else:
                tokens[-1] += char
            cur_type = 'o'
    return tokens
