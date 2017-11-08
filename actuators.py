"""
actuators.py
Classes to control the motors and servos. These classes 
are wrapped in a mixer class before being used in the drive loop.
"""

import time

import utils

        
class PCA9685:
    ''' 
    PWM motor controler using PCA9685 boards. 
    This is used for most RC Cars
    '''
    def __init__(self, channel, frequency=60):
        import Adafruit_PCA9685
        # Initialise the PCA9685 using the default address (0x40).
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(frequency)
        self.channel = channel

    def set_pulse(self, pulse):
        self.pwm.set_pwm(self.channel, 0, pulse) 

    def run(self, pulse):
        self.set_pulse(pulse)
        
class PWMSteering:
    """
    Wrapper over a PWM motor cotnroller to convert angles to PWM pulses.
    """
    LEFT_ANGLE = -1 
    RIGHT_ANGLE = 1

    def __init__(self, controller=None,
                       left_pulse=290,
                       right_pulse=490,
                       invert_steering_angle=False):

        self.controller = controller
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse

        '''
        It is suggest to invert the steering angle when running through the CNN.  
        We use 1/r instead of r to prevent a singularity when driving straight 
        (the turning radius for driving straight is infinity).
        1/r smoothly transitions through zero from left turns (negative values) 
        to right turns (positive values). 
        Set in config.py       
        '''
        self.invert_steering_angle = invert_steering_angle

    def run(self, angle):
        if self.invert_steering_angle:
            if angle != 0.0:
                angle = 1 / angle

        # map absolute angle to angle that vehicle can implement.
        pulse = utils.map_range(angle, 
                                self.LEFT_ANGLE, self.RIGHT_ANGLE,
                                self.left_pulse, self.right_pulse)

        self.controller.set_pulse(pulse)

    def shutdown(self):
        self.run(0) #set steering straight


class PWMThrottle:
    """
    Wrapper over a PWM motor cotnroller to convert -1 to 1 throttle
    values to PWM pulses.
    """
    MIN_THROTTLE = -1
    MAX_THROTTLE =  1

    def __init__(self, controller=None,
                       max_pulse=300,
                       min_pulse=490,
                       zero_pulse=350):

        self.controller = controller
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.zero_pulse = zero_pulse
        
        #send zero pulse to calibrate ESC
        self.controller.set_pulse(self.zero_pulse)
        time.sleep(1)


    def run(self, throttle):
        if throttle > 0:
            pulse = utils.map_range(throttle,
                                    0, self.MAX_THROTTLE, 
                                    self.zero_pulse, self.max_pulse)
        else:
            pulse = utils.map_range(throttle,
                                    self.MIN_THROTTLE, 0, 
                                    self.min_pulse, self.zero_pulse)

        self.controller.set_pulse(pulse)
        
    def shutdown(self):
        self.run(0) #stop vehicle



class MockController(object):
    def __init__(self):
        pass

    def run(self, pulse):
        pass

    def shutdown(self):
        pass
