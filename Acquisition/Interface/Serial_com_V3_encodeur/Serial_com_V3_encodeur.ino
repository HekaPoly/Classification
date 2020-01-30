/*
*   For Teensy 3.x it fills the buffer automatically using PDB
    For Teensy LC you can fill it yourself by pressing c

    For all boards you can see the buffer's contents by pressing p.
*/

#include <Arduino.h>
#include <Encoder.h>

Encoder knob1(14, 15);
IntervalTimer myTimer;

// asm instructions allow more precise delays than the built-in delay functions

const uint8_t kNumberOfPins = 3;
const uint8_t readPin[kNumberOfPins] = {A3};
const uint16_t kBufferSize = 200;
const uint8_t kPartitions = 2;
const uint16_t kPartitionSize = kBufferSize / kPartitions;

uint8_t pin_data[1][kBufferSize]; //matrix of 8-bit unsigned numbers, from 0 to 255

void setup() {
    Serial.begin(115200);
    /*
    for(size_t i = 3; i < kNumberOfPins+3; i++)
    {
        pinMode(readPin[i], INPUT); //Init each sensor
    }
    delay(1000);
    */
    pinMode(13, OUTPUT); //Init LED 13
    //digitalWriteFast(13, HIGH); //Turn on LED 13
    while(!Serial.available());
    digitalWrite(13, HIGH);
    
    myTimer.begin(getMesures, 500);
}


uint16_t current_data; //16-bit unsigned numbers, from 0 to 65 536
uint16_t current_index = 0;
uint16_t print_index = 0;
uint16_t n = 1;
boolean printbuffer = false;
//double duree = micros();


void getMesures(){

  /*
    for(uint8_t i = 3 ; i < kNumberOfPins+3; i++)
    {   
        if(i<5){
          //current_data = analogRead(i);                                 // 10000110 10010011 >> 8 =  00000000 10000110
        }
        else{
          //current_data =  knob1.read()*360/1600; 
        }
        current_data =  knob1.read()*360/1600; 
        Serial.println(current_data);
        pin_data[i][current_index] = current_data;              //  10010011
        //Serial.println(pin_data[i][current_index]);
        pin_data[i][current_index + 1] = (current_data >> 8); // 10000110
        //Serial.println(analogRead(i));
        //Serial.println(pin_data[i][current_index+1]);

        n = n+1;
       // Serial.println(int(current_index));
    }*/

    current_data =  knob1.read()*360/1600; 
    //Serial.println(current_data);
    pin_data[0][current_index] = current_data;              //  10010011
    //Serial.println(pin_data[i][current_index]);
    pin_data[0][current_index + 1] = (current_data >> 8); // 10000110
    

    current_index += 2;
    if (current_index % kPartitionSize == 0) 
    {
        print_index = current_index - kPartitionSize;
        printbuffer = true;
    }  
    current_index = current_index % kBufferSize;
}

void loop() {
    //double duree = micros();
    //Serial.println("hello");
   
    if(printbuffer){
      for(size_t i = 0; i < 1; i++)
      {
          //Serial.println("PRINTING");
          //Serial.write(i+1); //Send the number of the sensor to the serial port as bytes
          Serial.write(0);
          Serial.write(0);
          Serial.write(i+1);
          Serial.write(pin_data[i] + print_index, kPartitionSize); //Send the last 10 bytes of data from the sensor i     
      }
      
      printbuffer = false;
    }
    
}






