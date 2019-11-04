from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import time
from absl import app
from absl import flags
import numpy as np
import os
import tensorflow as tf
import glob
import json
import sys

import warnings 
warnings.simplefilter(action="ignore", category=FutureWarning)

from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit

from bandits.data.data_sampler import sample_stats_questions_data

from tf_params import *

from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.algorithms.uniform_sampling import UniformSampling


# Set up your file routes to the data files.

base_route = os.getcwd()
data_route = 'contextual_bandits/datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)

def sample_data(data_type, file_name, num_contexts=None, context_dim=None, num_actions=None):
  """Sample data from given 'data_type'.

  Args:
    data_type: Dataset from which to sample.
        file_name : File name of dataset.
    num_contexts: Number of contexts to sample.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
    opt_rewards: Vector of expected optimal reward for each context.
    opt_actions: Vector of optimal action for each context.
    num_actions: Number of available actions.
    context_dim: Dimension of each context.
  """

  if data_type in ['english', 'math', 'physics', 'stats', 'stackoverflow', 'stackoverflow_context']:
    dataset, opt_stats_questions = sample_stats_questions_data(file_name, context_dim,
                                                             num_actions, num_contexts,
                                                             shuffle_rows=False,
                                                             shuffle_cols=False)
    opt_rewards, opt_actions = opt_stats_questions

  return dataset, opt_rewards, opt_actions, num_actions, context_dim

def dump_results(id_exec, dump_file_per_algo, dump_file_h_stats, dump_file_overall,
                 algos, expert_representation, embedding_size, fusion_method,
                                 opt_rewards, opt_actions,
                                 h_rewards, h_actions, h_regrets, t_init, name):
    print('---------------------------------------------------')
    print('{} bandit completed after {} seconds.'.format(
        name, time.time() - t_init))
    print('---------------------------------------------------')

    performance_tuples = []
    for j, a in enumerate(algos):
        performance_tuples.append((a.name, np.sum(h_rewards[:, j]), np.sum(h_regrets[:, j])))
#   performance_tuples = sorted(performance_tuples,
#                             key=lambda elt: elt[0],
#                             reverse=True)

#   opt_total_reward = np.sum(opt_rewards)
#   opt_actions_freq = ' '.join(['%s:%s' % (elt, list(opt_actions).count(elt)) for elt in set(opt_actions)])

    tmp_list = []
    for i in range(h_actions.shape[0]):
        tmp_list.append('-'.join(['%s' % (j) for j in h_actions[i,:]]))
    str_h_actions = '|'.join(tmp_list)

    tmp_list = []
    for i in range(h_rewards.shape[0]):
        tmp_list.append('-'.join(['%s' % (j) for j in h_rewards[i,:]]))
    str_h_rewards = '|'.join(tmp_list)

    tmp_list = []
    for i in range(h_regrets.shape[0]):
        tmp_list.append('-'.join(['%s' % (j) for j in h_regrets[i,:]]))
    str_h_regrets = '|'.join(tmp_list)

#   time_spent = time.time() - t_init
#   for i, (alg_name, total_reward) in enumerate(performance_tuples):
#       dump_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (id_exec, alg_name,
#                                                                                                                 expert_representation,
#                                                                  embedding_size,
#                                                                                                                 fusion_method,
#                                                                                                                 name, total_reward,
#                                                                  time_spent,
#                                                                                                   opt_total_reward,
#                                                                  opt_actions_freq,
#                                                                                                           str_h_actions,
#                                                                  str_h_rewards,
#                                                                  str_h_regrets))

    time_spent = time.time() - t_init
    for i, (alg_name, total_reward, total_regret) in enumerate(performance_tuples):
        dump_file_per_algo.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (id_exec, alg_name,
                                                                                                               expert_representation,
                                                               embedding_size,
                                                                                                               fusion_method,
                                                                                                               name, total_reward,
                                                               total_regret,
                                                               time_spent))

    dump_file_h_stats.write('%s,%s,%s,%s,%s,%s,%s\n' % (id_exec,
                                                      expert_representation,
                                                      embedding_size,
                                                      fusion_method,
                                                      str_h_actions,
                                                      str_h_rewards,
                                                      str_h_regrets))

def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
  print('---------------------------------------------------')

  performance_pairs = []
  for j, a in enumerate(algos):
    performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
  performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')


flags.DEFINE_string('logdir',
                                        '/tmp/bandits/stats_questions/',
                                        'Base directory to save output')

def main(args):

    # Data type in {stats_questions}
    #data_type = 'stats_questions'
    data_type     = args[1]
    suffix        = args[2]
    experiment_id = args[3]

    # Base input directory
    #input_dir = '/home/matheus.silva/Mestrado/Datasets/%s/%s/%s/representations/%s/datasets/' % (database, data_type, suffix, experiment_id)
    # input_dir = '/home/matheus.silva/Mestrado/Datasets/%s/%s/representations/%s/datasets/' % (data_type, suffix, experiment_id)
    # C:/stackoverflow/
    input_dir = '/mnt/c/Users/mathe/OneDrive/Documentos/Mestrado/Files/datasets/stack_exchange/%s/non_filtered/%s/representations/%s/datasets' % (data_type, suffix, experiment_id)

    # Base output directory
    output_dir = '/%s/results/' % (input_dir)
    os.system('mkdir -p %s' % (output_dir))

    # Run for all datasets
    datasets_list = glob.glob('%s/*.npy' % (input_dir))

    # Since many algos are sthocastic, we need to run many times and average their results
    num_executions = 50

    dump_file_per_algo = open('%s/executions_embeddings_per_algo.csv' % (output_dir), 'w')
    dump_file_per_algo.write('id_exec,mab_name,expert_representation,embedding_size,fusion_method,dataset_id,total_reward,time_spent\n')

    dump_file_h_stats = open('%s/executions_embeddings_h_stats.csv' % (output_dir), 'w')
    dump_file_h_stats.write('id_exec,expert_representation,embedding_size,fusion_method,h_actions,h_rewards,h_regrets\n')

    dump_file_overall = open('%s/executions_embeddings_overall.csv' % (output_dir), 'w')
    dump_file_overall.write('algos,opt_total_reward,opt_actions_freq\n')

    print(datasets_list)
    for ds in datasets_list:
        # Getting execution data
        dataset_file_name        = os.path.join(input_dir, ds)
        fprefix                  = dataset_file_name.rfind('/')
        sprefix                  = dataset_file_name.find('.npy')
        prefix_dataset_file_name = dataset_file_name[fprefix+1:sprefix]

        print(dataset_file_name)
        print(fprefix)
        print(sprefix)
        print(prefix_dataset_file_name)

        for id_exec in range(num_executions):
    
            json_dataset_file_name = os.path.join(input_dir, '%s.json' % (prefix_dataset_file_name))
    
            f = open(json_dataset_file_name, 'r')
            exec_data = json.load(f)
            f.close()

            context_dim           = exec_data['context_dim']
            num_actions           = exec_data['num_actions'] 
            num_contexts          = exec_data['num_contexts']
            expert_representation = exec_data['expert_representation'] 
            embedding_size        = exec_data['embedding_size'] 
            fusion_method         = exec_data['fusion_method'] 
        
            sampled_vals = sample_data(data_type, dataset_file_name, num_contexts, context_dim, num_actions)
            dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals
            
            # Define hyperparameters and algorithms
            hparams           = tf.contrib.training.HParams(num_actions=num_actions)
            hparams_linear    = get_hparams_linear(num_actions, context_dim)
            hparams_rms       = get_hparams_rms(num_actions, context_dim)
            hparams_dropout   = get_hparams_dropout(num_actions, context_dim)
            hparams_bbb       = get_hparams_bbb(num_actions, context_dim)
            hparams_nlinear   = get_hparams_nlinear(num_actions, context_dim)
            hparams_nlinear2  = get_hparams_nlinear2(num_actions, context_dim)
            hparams_pnoise    = get_hparams_pnoise(num_actions, context_dim)
            hparams_alpha_div = get_hparams_alpha_div(num_actions, context_dim)
            hparams_gp        = get_hparams_gp(num_actions, context_dim)
            
            algos = [
                UniformSampling('Uniform Sampling', hparams),
                UniformSampling('Uniform Sampling 2', hparams),
#BAD                FixedPolicySampling('fixed1', [0.75, 0.25], hparams),
#BAD                FixedPolicySampling('fixed2', [0.25, 0.75], hparams),
                PosteriorBNNSampling('RMS', hparams_rms, 'RMSProp'),
                PosteriorBNNSampling('Dropout', hparams_dropout, 'RMSProp'),
                PosteriorBNNSampling('BBB', hparams_bbb, 'Variational'),
                NeuralLinearPosteriorSampling('NeuralLinear', hparams_nlinear),
                NeuralLinearPosteriorSampling('NeuralLinear2', hparams_nlinear2),
#BAD                LinearFullPosteriorSampling('LinFullPost', hparams_linear),
                BootstrappedBNNSampling('BootRMS', hparams_rms),
                ParameterNoiseSampling('ParamNoise', hparams_pnoise),
#               PosteriorBNNSampling('BBAlphaDiv', hparams_alpha_div, 'AlphaDiv'),
#               PosteriorBNNSampling('MultitaskGP', hparams_gp, 'GP'),
            ]
        
            # Run contextual bandit problem
            t_init = time.time()
            results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
            h_actions, h_rewards, h_regrets = results
            
            # Display results
            display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, data_type)
    
            # Dumping results
            dump_results(id_exec, dump_file_per_algo, dump_file_h_stats, dump_file_overall, 
                   algos, expert_representation, embedding_size, fusion_method,
                                     opt_rewards, opt_actions,
                                     h_rewards, h_actions, h_regrets, t_init, data_type)

            print(h_actions.shape, h_rewards.shape, num_contexts, opt_rewards.shape, h_regrets.shape)

    opt_total_reward = np.sum(opt_rewards)
    opt_actions_freq = ' '.join(['%s:%s' % (elt, list(opt_actions).count(elt)) for elt in set(opt_actions)])

    str_algos = '|'.join([a.name for a in algos])
    dump_file_overall.write('%s,%s,%s\n' % (str_algos, opt_total_reward, opt_actions_freq))

    dump_file_per_algo.flush()
    dump_file_per_algo.close()

    dump_file_h_stats.flush()
    dump_file_h_stats.close()

    dump_file_overall.flush()
    dump_file_overall.close()

if __name__ == '__main__':
    if(len(sys.argv) != 4):
        print('Usage: python run_stats_questions.py <fol> <dataset_id> <suffix>')
        print('dataset_id: stats_questions')
        print('suffix    : 40_context_dim_10_actions (when dataset == stats_questions)')
        print('experiment_id: ex: textual_embeddings (this will be used to create output_directory)')
        print('Aborting.')
        sys.exit(-1)
    
    app.run(main, sys.argv)
