
#include <Arduino.h>
#include <Encoder.h>

IntervalTimer myTimer;
void getMeasures();

const uint8_t kNumberOfElectrodes = 2;
const uint8_t kNumberOfEncodeur = 1;
const uint8_t electrodPins[kNumberOfElectrodes] = {14, 15};

// Declare all encoders and their pins
Encoder enc1(19, 20);
Encoder encoderTab[kNumberOfEncodeur] = {enc1};

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
int received_byte = 0;

void restartAcquisition();

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
    
    myTimer.begin(getMeasures, 500);
}

// Acquisition des données des élecrodes et encodeurs
void getMeasures(){
    if(acquisitionStarted){

        for(uint8_t i = 0 ; i < kNumberOfElectrodes; i++)
        {   
            current_data = analogRead(electrodPins[i]);

            pin_data[i][current_index] = current_data;              // 10010011
            pin_data[i][current_index + 1] = (current_data >> 8);   // 10000110
        }

        //Encodeur 1 : La valeur 0 n'est pas permise
        for(uint8_t i = 0; i< kNumberOfEncodeur; i++)
        {
            Encodeur_val =  encoderTab[i].read()*360/1600;

            if (Encodeur_val > 360) 
                Encodeur_val = Encodeur_val%360; //modulo 360
        
            else if (Encodeur_val < 0)
                Encodeur_val = Encodeur_val%360 + 360;
            
            current_data = Encodeur_val;
            pin_data[i+kNumberOfElectrodes][current_index] = current_data;             
            pin_data[i+kNumberOfElectrodes][current_index + 1] = (current_data >> 8);
        }
        

        current_index += 2;

        if (current_index % kPartitionSize == 0) //Si current_index == 100 ou 200 
        {
            print_index = current_index - kPartitionSize;
            printbuffer = true;
        }  
        current_index = current_index % kBufferSize; //Met le count à 0 ou 100
    }
}

void loop(){
    if(printbuffer){
      for(size_t i = 0; i < kNumberOfElectrodes + kNumberOfEncodeur ; i++)
      {
          Serial.write(0xFF);
          Serial.write(0xFF);
          Serial.write(i+1);
          Serial.write(pin_data[i] + print_index, kPartitionSize); //Send the last 10 bytes of data from the sensor i     
      }
      
      printbuffer = false;
    }

    if(Serial.available() > 0){
        received_byte = Serial.read();

        if(received_byte == '#')
            acquisitionStarted = true;
        
        else if(received_byte == '!')
            restartAcquisition();
    }
}

void restartAcquisition(){
    acquisitionStarted = false;
    current_index = 0;
}




