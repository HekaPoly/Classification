#include <Arduino.h>

void setup() {
  Serial.begin(9600);
}

void loop() {
  Serial.println(analogRead(15));
  delay(10);
}