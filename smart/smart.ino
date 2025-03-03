
const int relay_pin = 2;

void setup() {
  // put your setup code here, to run once:
  pinMode( relay_pin, OUTPUT);
  Serial.begin(9600);
  digitalWrite( relay_pin,1);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()>0)
  {
    String command = Serial.readString();

    if(command == "ON"){
      digitalWrite( relay_pin, 0);
    }
    else if(command == "OFF"){
      digitalWrite( relay_pin, 1);
    }
  }
}
