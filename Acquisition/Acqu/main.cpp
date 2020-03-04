#include <Arduino.h>
const uint8_t kNumberOfPins = 9;
const uint8_t readPin[kNumberOfPins] = {A0, A1, A2, A3, A4, A5, A6, A7, A8};
uint32_t data;

void setup() {
    Serial.begin(115200);
    for(size_t i = 1; i < kNumberOfPins; i++)
    {
        pinMode(readPin[i], INPUT);
    }
    pinMode(13, OUTPUT);
    digitalWriteFast(13, HIGH);
}

void loop() {
    for(size_t i = 1; i < kNumberOfPins; i++){
        Serial.print("#");
        Serial.print(i);
        data = analogRead(i);
        Serial.print(data);
        delay(500);
    }
  }