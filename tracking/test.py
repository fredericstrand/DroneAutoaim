from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

factory = PiGPIOFactory()
servo = AngularServo(12, min_angle=-90, max_angle=90, min_pulse_width=0.0006, max_pulse_width=0.0024, pin_factory=factory)

while True:
    servo.angle = -90
    sleep(1)
    servo.angle = 0
    sleep(1)
    servo.angle = 90
    sleep(1)
    servo.angle = 0
    sleep(1)