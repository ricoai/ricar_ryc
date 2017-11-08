#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it. 

Usage:
    manage.py drive [--model=<model>] [--web=<True/False>] [--throttle=<Throttle 0.0-1.0>] [--js]
    manage.py train (--tub=<tub>) (--model=<model>)
    manage.py calibrate
    manage.py (calibrate)
    manage.py (check) [--tub=<tub1,tub2,..tubn>] [--fix]
    manage.py (analyze) [--tub=<tub1,tub2,..tubn>] (--op=<histogram>) (--rec=<"user/angle">)
"""


import os
from docopt import docopt
from vehicle import Vehicle, Lambda
from cameras import PiCamera
from joystick import JoystickController
from keras import KerasRicar
from actuators import PCA9685, PWMSteering, PWMThrottle
from tub import TubHandler, Tub
from utils import linear_bin
from config import load_config


def drive(cfg, model_path=None, use_joystick=True):
    """
    Drive the car.
    You will either drive to record data for training or drive to test the autonomous mode.
    Either use Web controls or Joystick to control the vehicle.
    If driving autonomous, give the model to load.
    :param cfg: Configuration for user defined values.
    :param model_path: Path to load the model.
    :param use_joystick Use parameter in startup to use joystick.
    """
    #Initialized car
    V = Vehicle()

    # Setup camera
    cam = PiCamera()
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    # Select if only use bluetooth PS3 controller
    ctr = JoystickController(max_throttle=cfg.JOYSTICK_MAX_THROTTLE,
                             steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                             auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)

    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)
    
    # See if we should even run the pilot module.
    # This is only needed because the part run_contion only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True
        
    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'], outputs=['run_pilot'])

    # Load the model configuration
    kl = KerasRicar()

    if model_path:
        print(model_path)
        kl.load(model_path)
    
    V.add(kl, inputs=['cam/image_array'], 
          outputs=['pilot/angle', 'pilot/throttle'],
          run_condition='run_pilot')

    # Choose what inputs should change the car.
    def drive_mode(mode, 
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle
        
        elif mode == 'local_angle':
            return pilot_angle, user_throttle
        
        else: 
            return pilot_angle, pilot_throttle
        
    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part, 
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'], 
          outputs=['angle', 'throttle'])

    # Configure the throttle and angle control hardware
    # Calibrate min/max for steering angle
    # Calibrate min/max/zero for throttle
    steering_controller = PCA9685(1)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=460, right_pulse=260,
                                    invert_steering_angle=cfg.INVERT_STEERING_ANGLE)
    
    throttle_controller = PCA9685(0)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=500, zero_pulse=370, min_pulse=220)
    
    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])
    
    # Add tub to save data
    inputs = ['cam/image_array',
              'user/angle', 'user/throttle',
              'pilot/angle', 'pilot/throttle',
              'user/mode']
    types = ['image_array',
             'float', 'float',
             'float', 'float',
             'str']
    
    th = TubHandler(path=cfg.DATA_PATH)
    tub_writer = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub_writer, inputs=inputs, run_condition='recording')
    
    # Run the vehicle for 20 seconds
    V.start(rate_hz=cfg.FPS, max_loop_count=100000)
    
    print("You can now go to <your pi ip address>:8887 to drive your car.")


def expand_path_masks(paths):
    '''
    take a list of paths and expand any wildcards
    returns a new list of paths fully expanded
    '''
    import glob
    expanded_paths = []
    for path in paths:
        if '*' in path or '?' in path:
            mask_paths = glob.glob(path)
            expanded_paths += mask_paths
        else:
            expanded_paths.append(path)

    return expanded_paths


def gather_tubs(cfg, tub_names):
    if tub_names:
        tub_paths = [os.path.expanduser(n) for n in tub_names.split(',')]
        tub_paths = expand_path_masks(tub_paths)
    else:
        tub_paths = [os.path.join(cfg.DATA_PATH, n) for n in os.listdir(cfg.DATA_PATH)]

    tubs = [Tub(p) for p in tub_paths]
    return tubs


def train(cfg, tub_names, model_name):
    """
    Train the model using the neural network based off the tubs given.
    The tubs contain the recorded data.
    :param cfg: Configuration for user settings.
    :param tub_names: Tubs to load.  This must be the full path.
    :param model_name: Name of the model to create.
    """
    # Get the configuration
    is_tensorboard = cfg.IS_TENSORBOARD
    is_plot = cfg.IS_PLOT_RESULTS
    epochs = cfg.EPOCHS
    lr = cfg.LEARNING_RATE
    is_stop_early = cfg.IS_EARLY_STOP
    early_stop_count = cfg.EARLY_STOP_COUNT
    dropout_1 = cfg.DROPOUT_1
    dropout_2 = cfg.DROPOUT_2
    optimizer = cfg.OPTIMIZER
    loss_weight_angle = cfg.LOSS_WEIGHT_ANGLE
    loss_weight_throttle = cfg.LOSS_WEIGHT_THROTTLE
    is_categorical = cfg.IS_CATEGORICAL
    is_lr_decay = cfg.IS_LR_DECAY

    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    def rt(record):
        record['user/angle'] = linear_bin(record['user/angle'])
        return record

    # Load the model
    kl = KerasRicar(dropout_1=dropout_1,
                    dropout_2=dropout_2,
                    optimizer=optimizer,
                    learning_rate=lr,
                    loss_weight_angle=loss_weight_angle,
                    loss_weight_throttle=loss_weight_throttle,
                    is_categorical=is_categorical,
                    is_lr_decay=is_lr_decay)

    tubs = gather_tubs(cfg, tub_names)

    import itertools

    gens = [tub.train_val_gen(X_keys, y_keys, record_transform=rt, batch_size=cfg.BATCH_SIZE, train_split=cfg.TRAIN_TEST_SPLIT) for tub in tubs]

    # Training data generator is the one that keeps cycling through training data generator of all tubs chained together
    # The same for validation generator
    train_gens = itertools.cycle(itertools.chain(*[gen[0] for gen in gens]))
    val_gens = itertools.cycle(itertools.chain(*[gen[1] for gen in gens]))

    model_path = os.path.expanduser(model_name)

    total_records = sum([t.get_num_records() for t in tubs])
    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' %(total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    # Train with the data loaded from the tubs
    kl.train(train_gens,
             val_gens,
             epochs=epochs,
             saved_model_path=model_path,
             is_early_stop=is_stop_early,
             early_stop_count=early_stop_count,
             is_tensorboard=is_tensorboard,
             is_plot_results=is_plot)


def calibrate():
    channel = int(input('Enter the channel your actuator uses (0-15).'))
    c = PCA9685(channel)
    
    for i in range(10):
        pmw = int(input('Enter a PWM setting to test(100-600)'))
        c.run(pmw)


def check(cfg, tub_names, fix=True):
    '''
    Check for any problems. Looks at tubs and find problems in any records or images that won't open.
    If fix is True, then delete images and records that cause problems.
    '''
    tubs = gather_tubs(cfg, tub_names)

    for tub in tubs:
        tub.check(fix=fix)


def anaylze(cfg, tub_names, op, record):
    '''
    look at the tub data and produce some analysis
    '''
    tubs = gather_tubs(cfg, tub_names)

    if op == 'histogram':
        import matplotlib.pyplot as plt
        samples = []
        for tub in tubs:
            num_records = tub.get_num_records()
            for iRec in range(0, num_records):
                json_data = tub.get_json_record(iRec)
                sample = json_data[record]
                samples.append(float(sample))

        plt.hist(samples, 50)
        plt.xlabel(record)
        plt.show()


if __name__ == '__main__':
    args = docopt(__doc__)

    # Put config.py in the same location as manage.py
    my_cfg = load_config()

    if args['drive']:
        drive(my_cfg, model_path=args['--model'])

    elif args['calibrate']:
        calibrate()

    elif args['train']:
        tub = args['--tub']
        model = args['--model']
        train(my_cfg, tub, model)

    elif args['check']:
        tub = args['--tub']
        fix = args['--fix']
        check(my_cfg, tub, fix)

    elif args['analyze']:
        tub = args['--tub']
        op = args['--op']
        rec = args['--rec']
        anaylze(my_cfg, tub, op, rec)



