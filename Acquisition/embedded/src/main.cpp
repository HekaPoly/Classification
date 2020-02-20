/*
*   For Teensy 3.x it fills the buffer automatically using PDB
    For Teensy LC you can fill it yourself by pressing c

    For all boards you can see the buffer's contents by pressing p.
*/

#include <Arduino.h>
#include <Encoder.h>

IntervalTimer myTimer;
void getMeasures();

const uint8_t kNumberOfElectrodes = 2;
const uint8_t kNumberOfEncodeur = 0;
const uint8_t electrodPins[kNumberOfElectrodes] = {14, 15};

// Declare all encoders and there pins
Encoder enc1(22, 23);
Encoder encoderTab[kNumberOfEncodeur];// = {enc1};

// Data buffer
const uint16_t kBufferSize = 200;                           //Size du buffer en data
const uint8_t kPartitions = 2;                              //Nombre de byte par data
const uint16_t kPartitionSize = kBufferSize / kPartitions;  //Nombre de data a envoyer

// Buffer with all read data
uint8_t pin_data[kNumberOfElectrodes+kNumberOfEncodeur][kBufferSize]; //matrix of 8-bit unsigned numbers, from 0 to 255


uint16_t current_data; //16-bit unsigned numbers, from 0 to 65 536
uint16_t current_index = 0;
uint16_t print_index = 0;
bool printbuffer = false;
bool acquisitionStarted = false;
int Encodeur_val = 0;

void setup() {
    Serial.begin(115200);

    // Setup electrod pins
    for(size_t i = 0; i < kNumberOfElectrodes; i++)
    {
        pinMode(electrodPins[i], INPUT); //Init each sensor
    }

    delay(1000);
    pinMode(13, OUTPUT); //Init LED 13
    digitalWriteFast(13, HIGH); //Turn on LED 13

    while(!Serial.available());
    myTimer.begin(getMeasures, 500);
}

// Acquisition des données des élecrodes et encodeurs
void getMeasures(){
    if(acquisitionStarted){

        for(uint8_t i = 0 ; i < kNumberOfElectrodes; i++)
        {   
            current_data = analogRead(electrodPins[i]);
            if(current_data == 0)
            {
            current_data = 1;
            }

            pin_data[i][current_index] = current_data;              // 10010011
            pin_data[i][current_index + 1] = (current_data >> 8);   // 10000110
        }

        //Encodeur 1 : La valeur 0 n'est pas permise
        for(uint8_t i = 0; i< kNumberOfEncodeur; i++)
        {
            Encodeur_val =  encoderTab[i].read();

            if (Encodeur_val > 360){
            Encodeur_val = Encodeur_val%360; //modulo 360
            }
            else if (Encodeur_val < 0){
            Encodeur_val = Encodeur_val%360 + 360;
            } 
            else if (Encodeur_val == 0){
            Encodeur_val = 1;
            }
        
            current_data = Encodeur_val;
            pin_data[i+kNumberOfElectrodes][current_index] = current_data;             
            pin_data[i+kNumberOfElectrodes][current_index + 1] = (current_data >> 8);
        }
        

        current_index += 2;

        if (current_index % kPartitionSize == 0) 
        {
            print_index = current_index - kPartitionSize;
            printbuffer = true;
        }  
        current_index = current_index % kBufferSize;
    }
}

void loop() {
    if(printbuffer){
      for(size_t i = 0; i < kNumberOfElectrodes + kNumberOfEncodeur ; i++)
      {
          Serial.write(0);
          Serial.write(0);
          Serial.write(i+1);
          Serial.write(pin_data[i] + print_index, kPartitionSize); //Send the last 10 bytes of data from the sensor i     
      }
      
      printbuffer = false;
    }
}






