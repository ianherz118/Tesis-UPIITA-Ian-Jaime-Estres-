#include <Brain.h>
Brain brain(Serial);

void setup(){
 // Inicializar la comunicación en serie:
  Serial.begin(9600);
  pinMode(10, INPUT); // Configuración para la detección LO +
  pinMode(11, INPUT); // Configuración para la detección LO -
}
void loop() {
if((digitalRead(10) == 1)||(digitalRead(11) == 1)){
}
else{
// Imprimir la lectura del puerto A0
    Serial.println(analogRead(A0));
if (brain.update()) {
    Serial.println(brain.readCSV());
  }
}

//Espere un poco para evitar que los datos en serie se saturen
delay(10);
}
