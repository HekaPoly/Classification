/*
*   For Teensy 3.x it fills the buffer automatically using PDB
    For Teensy LC you can fill it yourself by pressing c

    For all boards you can see the buffer's contents by pressing p.
*/

#include <Arduino.h>


// asm instructions allow more precise delays than the built-in delay functions
#define NOP5 "nop\n\t""nop\n\t""nop\n\t""nop\n\t""nop\n\t"
#define NOP4 "nop\n\t""nop\n\t""nop\n\t""nop\n\t"
#define NOP10 NOP5 NOP5
#define NOP20 NOP10 NOP10
#define NOP40 NOP20 NOP20

#define DELAY_1US __asm__(NOP40 NOP4)
#define DELAY_2US __asm__(NOP40 NOP40 NOP10 NOP4)
#define DELAY_3US __asm__(NOP40 NOP40 NOP40 NOP20)

const uint8_t kNumberOfPins = 7;
const uint8_t readPin[kNumberOfPins] = {A0, A1, A2, A3, A4, A5, A6};
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
        Serial.write(pin_data[i] + index, kPartitionSize);
    }
}

uint16_t current_data;
uint16_t current_index = 0;
uint16_t print_index = 0;
void loop() {
    
    for(uint8_t i = 0; i < kNumberOfPins; i++)
    {
        current_data = analogRead(i);
        pin_data[i][current_index] = current_data & 0xff;
        pin_data[i][current_index + 1] = (current_data >> 8);
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


