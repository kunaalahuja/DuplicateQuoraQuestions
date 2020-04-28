import os
import sys
import json
import datetime
import pprint
# %tensorflow_version 1.15
import tensorflow as tf

# Authenticate, so we can access storage bucket and TPU
from google.colab import auth
auth.authenticate_user()

USE_TPU = True

BERT_MODEL = 'uncased_L-12_H-768_A-12'

BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
!gsutil ls $BERT_PRETRAINED_DIR

# Bucket for saving checkpoints and outputs
BUCKET = 'nlp_bert_quora' 
if BUCKET!="":
  OUTPUT_DIR = 'gs://{}/outputs'.format(BUCKET)
  tf.gfile.MakeDirs(OUTPUT_DIR)
elif USE_TPU:
  raise ValueError('Must specify an existing GCS bucket name for running on TPU')
else:
  OUTPUT_DIR = 'out_dir'
  os.mkdir(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

if USE_TPU:
  # getting info on TPU runtime
  assert 'COLAB_TPU_ADDR' in os.environ, 'ERROR: Not connected to a TPU runtime; Change notebook runtype to TPU'
  TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  print('TPU address is', TPU_ADDRESS)

!test -d bert || git clone -q https://github.com/google-research/bert.git
if not 'bert' in sys.path:
  sys.path += ['bert']

TASK_DATA_DIR = 'glue_data/QQP'
!test -d glue_data || git clone https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git glue_data
!test -d $TASK_DATA_DIR || python glue_data/download_glue_data.py --data_dir glue_data --tasks=QQP
!ls -als $TASK_DATA_DIR

import modeling
import optimization
import tokenization
import run_classifier

# Model Hyper Parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 5.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 200 

# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

class QQPProcessor(run_classifier.DataProcessor):
  @classmethod
  def _read_csv(self, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=",", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def get_train_examples(self, data_dir):
    """Reading train.tsv and converting to list of InputExample"""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,"train4.tsv")), 'train')

  def get_dev_examples(self, data_dir):
    """Reading dev.tsv and converting to list of InputExample"""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,"dev.tsv")), 'dev')
  
  def get_test_examples(self, data_dir):
    """Reading train.tsv and converting to list of InputExample"""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,"test.tsv")), 'test')
  
  def get_predict_examples(self, sentence_pairs):
    """Given question pairs, conevrting to list of InputExample"""
    examples = []
    for (i, qpair) in enumerate(sentence_pairs):
      guid = "predict-%d" % (i)
      # converting questions to utf-8 and creating InputExamples
      text_a = tokenization.convert_to_unicode(qpair[0])
      text_b = tokenization.convert_to_unicode(qpair[1])
      # We will add label  as 0, because None is not supported in converting to features
      examples.append(
          run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=0))
    return examples
  
  def _create_examples(self, lines, set_type):
    """Creates examples for the training, dev and test sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%d" % (set_type, i)
      if set_type=='test':
        # removing header and invalid data
        if i == 0 or len(line)!=3:
          print(guid, line)
          continue
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        label = 0 # We will use zero for test as convert_example_to_features doesn't support None
      else:
        # removing header and invalid data
        if i == 0:
          continue
        try:  
          text_a = tokenization.convert_to_unicode(line[3])
          text_b = tokenization.convert_to_unicode(line[4])
          label = int(line[5])
        except:
          print('failed on ', i, "th row")  
      examples.append(
          run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    "return class labels"
    return [0,1]

# Instantiate an instance of QQPProcessor and tokenizer
processor = QQPProcessor()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

print("################  Processing Training Data #####################")
# TASK_DATA_DIR = "glue_data"
TRAIN_TF_RECORD = os.path.join(OUTPUT_DIR, "train.tf_record")
train_examples = processor.get_train_examples(TASK_DATA_DIR)
num_train_examples = len(train_examples)
num_train_steps = int( num_train_examples / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
run_classifier.file_based_convert_examples_to_features(train_examples, label_list, MAX_SEQ_LENGTH, tokenizer, TRAIN_TF_RECORD)

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  # Bert Model instant 
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Getting output for last layer of BERT
  output_layer = model.get_pooled_output()
  
  # Number of outputs for last layer
  hidden_size = output_layer.shape[-1].value
  
  # We will use one layer on top of BERT pretrained for creating classification model
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9) # 0.1 dropout
    
    # Calcaulte prediction probabilites and loss
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  
    """The `model_fn` for TPUEstimator."""

    # reading features input
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    
    # checking if training mode
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    # create simple classification model
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)
    
    # getting variables for intialization and using pretrained init checkpoint
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # defining optimizar function
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      
      # Training estimator spec
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      # accuracy, loss, auc, F1, precision and recall metrics for evaluation
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predictions)
        auc = tf.metrics.auc(
            label_ids,
            predictions)
        recall = tf.metrics.recall(
            label_ids,
            predictions)
        precision = tf.metrics.precision(
            label_ids,
            predictions) 
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      # estimator spec for evalaution
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      # estimator spec for predictions
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

# Define TPU configs
# USE_TPU = True
# TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']

if USE_TPU:
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
else:
  tpu_cluster_resolver = None

run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

model_fn = model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=USE_TPU,
    use_one_hot_embeddings=True)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    predict_batch_size=PREDICT_BATCH_SIZE)

# Train the model.
print('QQP on BERT base model normally takes about 1 hour on TPU and 15-20 hours on GPU. Please wait...')
print('***** Started training at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(num_train_examples))
print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("  Num steps = %d", num_train_steps)
# we are using `file_based_input_fn_builder` for creating input function from TF_RECORD file
train_input_fn = run_classifier.file_based_input_fn_builder(TRAIN_TF_RECORD,
                                                            seq_length=MAX_SEQ_LENGTH,
                                                            is_training=True,
                                                            drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('***** Finished training at {} *****'.format(datetime.datetime.now()))

# eval the model on train set.
print('***** Started Train Set evaluation at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(num_train_examples))
print('  Batch size = {}'.format(EVAL_BATCH_SIZE))
# eval input function for train set
train_eval_input_fn = run_classifier.file_based_input_fn_builder(TRAIN_TF_RECORD,
                                                           seq_length=MAX_SEQ_LENGTH,
                                                           is_training=False,
                                                           drop_remainder=True)
# evalute on train set
result = estimator.evaluate(input_fn=train_eval_input_fn, 
                            steps=int(num_train_examples/EVAL_BATCH_SIZE))
print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
print("***** Eval results *****")
for key in sorted(result.keys()):
  print('  {} = {}'.format(key, str(result[key])))

result = list(estimator.predict(input_fn=train_eval_input_fn))
output_test_file = os.path.join(OUTPUT_DIR, "train_predictions.txt")
with tf.gfile.GFile(output_test_file, "w") as writer:
  for (example_i, predictions_i) in enumerate(result):
    writer.write("%s , %s\n" % (train_examples[example_i].guid, str(predictions_i['probabilities'][1])))

# Converting eval examples to features
print("################  Processing Dev Data #####################")
EVAL_TF_RECORD = os.path.join(OUTPUT_DIR, "eval.tf_record")
eval_examples = processor.get_dev_examples(TASK_DATA_DIR)
num_eval_examples = len(eval_examples)
run_classifier.file_based_convert_examples_to_features(eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer, EVAL_TF_RECORD)

# Eval the model on Dev set.
print('***** Started Dev Set evaluation at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(num_eval_examples))
print('  Batch size = {}'.format(EVAL_BATCH_SIZE))

# eval input function for dev set
eval_input_fn = run_classifier.file_based_input_fn_builder(EVAL_TF_RECORD,
                                                           seq_length=MAX_SEQ_LENGTH,
                                                           is_training=False,
                                                           drop_remainder=True)
# evalute on dev set
result = estimator.evaluate(input_fn=eval_input_fn, steps=int(num_eval_examples/EVAL_BATCH_SIZE))
print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
print("***** Eval results *****")
for key in sorted(result.keys()):
  print('  {} = {}'.format(key, str(result[key])))

result = list(estimator.predict(input_fn=eval_input_fn))
output_test_file = os.path.join(OUTPUT_DIR, "dev_predictions.txt")
with tf.gfile.GFile(output_test_file, "w") as writer:
  for (example_i, predictions_i) in enumerate(result):
    writer.write("%s , %s\n" % (eval_examples[example_i].guid, str(predictions_i['probabilities'][1])))

# Converting test examples to features
print("################  Processing Test Data #####################")
TEST_TF_RECORD = os.path.join(OUTPUT_DIR, "test.tf_record")
test_examples = processor.get_test_examples(TASK_DATA_DIR)
num_test_examples = len(test_examples)
run_classifier.file_based_convert_examples_to_features(test_examples, label_list, MAX_SEQ_LENGTH, tokenizer, TEST_TF_RECORD)

# Predictions on test set.
print('***** Started Prediction at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(num_test_examples))
print('  Batch size = {}'.format(PREDICT_BATCH_SIZE))
# predict input function for test set
test_input_fn = run_classifier.file_based_input_fn_builder(TEST_TF_RECORD,
                                                           seq_length=MAX_SEQ_LENGTH,
                                                           is_training=False,
                                                           drop_remainder=True)
tf.logging.set_verbosity(tf.logging.ERROR)
# predict on test set
result = list(estimator.predict(input_fn=test_input_fn))
print('***** Finished Prediction at {} *****'.format(datetime.datetime.now()))

# saving test predictions
output_test_file = os.path.join(OUTPUT_DIR, "test_predictions.txt")
with tf.gfile.GFile(output_test_file, "w") as writer:
  for (example_i, predictions_i) in enumerate(result):
    writer.write("%s , %s\n" % (test_examples[example_i].guid, str(predictions_i['probabilities'][1])))

