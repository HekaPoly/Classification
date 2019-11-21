/*
*   For Teensy 3.x it fills the buffer automatically using PDB
    For Teensy LC you can fill it yourself by pressing c

    For all boards you can see the buffer's contents by pressing p.
*/

#include <Arduino.h>


// asm instructions allow more precise delays than the built-in delay functions

const uint8_t kNumberOfPins = 1;
const uint8_t readPin[kNumberOfPins] = {A0};
//const uint8_t readPin[kNumberOfPins] = {A0, A1, A2, A3, A4, A5, A6};
const uint16_t kBufferSize = 700;
const uint8_t kPartitions = 2;
const uint16_t kPartitionSize = kBufferSize / kPartitions;

uint8_t pin_data[kNumberOfPins][kBufferSize];

void setup() {
    Serial.begin(115200);
    for(size_t i = 0; i < kNumberOfPins; i++)
    {
        pinMode(readPin[i], INPUT);
    }
    delay(5000);
    pinMode(13, OUTPUT);
    digitalWriteFast(13, HIGH);
}

void print_buffer(int index) {
    for(size_t i = 0; i < kNumberOfPins; i++)
    {
        Serial.write(i+1);
        Serial.write(pin_data[i] + index, kPartitionSize);
    
    }
}

uint16_t current_data;
uint16_t current_index = 0;
uint16_t print_index = 0;
void loop() {
    
    for(uint8_t i = 0; i < kNumberOfPins; i++)
    {
        current_data = analogRead(i);                                 // 10000110 10010011 >> 8 =  00000000 10000110
        pin_data[i][current_index] = current_data;              //  10010011
        pin_data[i][current_index + 1] = (current_data >> 8); // 10000110
        //Serial.println(analogRead(i));

    }
    current_index += 2;
    if (current_index % kPartitionSize == 0) 
    {
        print_index = current_index - kPartitionSize;
        print_buffer(print_index);
    }
    current_index = current_index % kBufferSize;

    delayMicroseconds(400);
}


