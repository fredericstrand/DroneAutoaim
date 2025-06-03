#include <Servo.h>

Servo servoX;
Servo servoY;

String input = "";

void setup()
{
    Serial.begin(9600);
    servoX.attach(9);
    servoY.attach(10);
}

void loop()
{
    while (Serial.available() > 0)
    {
        char c = Serial.read();
        if (c == '\n')
        {
            int commaIndex = input.indexOf(',');
            if (commaIndex > 0)
            {
                int angleX = input.substring(0, commaIndex).toInt();
                int angleY = input.substring(commaIndex + 1).toInt();

                angleX = constrain(angleX, 0, 180);
                angleY = constrain(angleY, 0, 180);

                servoX.write(angleX);
                servoY.write(angleY);
            }
            input = "";
        }
        else
        {
            input += c;
        }
    }
}
