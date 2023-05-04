#include <Servo.h>
Servo servoVer; //Vertical Servo
Servo servoHor; //Horizontal Servo
int x;
int y;
int prevX;
int prevY;
void setup()
{
  pinMode(5,OUTPUT);
  Serial.begin(9600);
  servoVer.attach(9); //Attach Vertical Servo to Pin 5
  servoHor.attach(11); //Attach Horizontal Servo to Pin 6
  servoVer.write(90);
  servoHor.write(90);
}
void Pos()
{
  if(prevX != x || prevY != y)
  {
    int servoX = map(x, 600, 0, 0, 180);//frame zize min and max,servo min and max
    int servoY = map(y, 450, 0, 180, 0);//frame zize min and max,servo min and max
    servoX = min(servoX, 180);
    servoX = max(servoX, 0);
    servoY = min(servoY, 180);
    servoY = max(servoY, 0);
    
    servoHor.write(servoX);
    servoVer.write(servoY);
    
  }
}
void loop()
{
  if(Serial.available() > 0)
  {
    if(Serial.read() == 'X')
    {
      x = Serial.parseInt();
      if(Serial.read() == 'Y')
      {
        y = Serial.parseInt();
       Pos();
      }
    }
    while(Serial.available() > 0)
    {
      Serial.read();
    }
  }
}
