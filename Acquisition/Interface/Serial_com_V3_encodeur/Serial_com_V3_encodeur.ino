/*
*   For Teensy 3.x it fills the buffer automatically using PDB
    For Teensy LC you can fill it yourself by pressing c

    For all boards you can see the buffer's contents by pressing p.
*/

#include <Arduino.h>
#include <Encoder.h>

Encoder Encoder1(22, 23);
IntervalTimer myTimer;

// asm instructions allow more precise delays than the built-in delay functions

const uint8_t kNumberOfElectrodes = 1;
const uint8_t kNumberOfEncodeur = 1;
const uint8_t readPin[kNumberOfElectrodes] = {A0};
const uint16_t kBufferSize = 200;
const uint8_t kPartitions = 2;
const uint16_t kPartitionSize = kBufferSize / kPartitions;

uint8_t pin_data[kNumberOfElectrodes+kNumberOfEncodeur][kBufferSize]; //matrix of 8-bit unsigned numbers, from 0 to 255

void setup() {
    Serial.begin(115200);
    for(size_t i = 0; i < kNumberOfElectrodes; i++)
    {
        pinMode(readPin[i], INPUT); //Init each sensor
    }
    delay(1000);
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
int Encodeur_val = 0;
//double duree = micros();


void getMesures(){

    for(uint8_t i = 0 ; i < kNumberOfElectrodes; i++)
    {   
        current_data = analogRead(i);
        if(current_data == 0){
          current_data = 1;
        }
        //Serial.println(current_data);
        pin_data[i][current_index] = current_data;              //  10010011
        //Serial.println(pin_data[i][current_index]);
        pin_data[i][current_index + 1] = (current_data >> 8); // 10000110
        //Serial.println(analogRead(i));
        //Serial.println(pin_data[i][current_index+1]);

        n = n+1;
       // Serial.println(int(current_index));
    }

    //Encodeur 1 : La valeur 0 n'est pas permise
    Encodeur_val =  Encoder1.read()*360/1600;
    if (Encodeur_val > 360){
      Encodeur_val = Encodeur_val%360; //modulo 360
    }
    else if (Encodeur_val < 0){
      Encodeur_val = Encodeur_val%360 + 360;
    } 
    else if (Encodeur_val == 0){
      Encodeur_val = 1;
    }
   
    //Serial.println(Encodeur_val);
    current_data = Encodeur_val;
    pin_data[kNumberOfElectrodes][current_index] = current_data;             
    //Serial.println(pin_data[i][current_index]);
    pin_data[kNumberOfElectrodes][current_index + 1] = (current_data >> 8);
    

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
      for(size_t i = 0; i < kNumberOfElectrodes + kNumberOfEncodeur ; i++)
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






