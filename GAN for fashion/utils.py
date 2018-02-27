# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:33:53 2018

@author: dihnatov
"""

'''
This implementaton based on https://github.com/mari-linhares/DeepLearning/tree/master/GAN-fashion-MNIST
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os


def generate_4x4_figure(samples):
  '''Generate a 4x4 figure.'''
  fig = plt.figure(figsize=(1, 1))
  gs = gridspec.GridSpec(1, 1)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(200, 200, 3))

  return fig

def maybe_create_out_path(out_path):
  '''If output path does not exist it will be created.'''
  if not os.path.exists(out_path):
    os.makedirs(out_path)

def save_plot(samples, out_path, train_step):
  '''Generates a plot and saves it.'''
  fig = generate_4x4_figure(samples)
  
  file_name = 'step-{}.png'.format(str(train_step).zfill(3))
  full_path = os.path.join(out_path, file_name)
  
  print ('Saving image:', full_path)
  
  maybe_create_out_path(out_path)
  
  plt.savefig(full_path, bbox_inches='tight')
  plt.close(fig)
