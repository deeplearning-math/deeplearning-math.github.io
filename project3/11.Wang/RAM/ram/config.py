class Config(object):

  win_size = 12 ##
  bandwidth = win_size**2
  batch_size = 128 ##
  eval_batch_size = 128 ##
  loc_std = 0.22
  original_size = 100 ##
  num_channels = 1 # do not change, not tested
  num_scales = 4 ##
  sensor_size = win_size**2 * num_channels * num_scales
  minRadius = 8
  hg_size = hl_size = 128
  g_size = 256
  cell_output_size = 256
  loc_dim = 2
  cell_size = 256
  cell_out_size = cell_size
  num_glimpses = 8 ##
  num_classes = 10
  max_grad_norm = 5.

  step = 1000 ##
  lr_start = 1e-3
  lr_min = 1e-4

  # Monte Carlo sampling
  M = 10

  # Run name
  run_name = "{}x{}-{}glimpse-{}x{}-{}scales-{}batch-{}epochs".format(original_size,
                                                     original_size,
                                                     num_glimpses,
                                                     win_size,
                                                     win_size,
                                                     num_scales,
                                                     batch_size,
                                                     step
                                                    )
