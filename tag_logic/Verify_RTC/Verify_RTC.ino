#include <Wire.h>

// Pin that the RV-3032-C7 CLKOUT is connected to.
// This MUST be an interrupt pin (pin 2 or 3 on the UNO).
const int CLKOUT_PIN = 2;

// This variable will be used inside the interrupt service routine (ISR).
// It must be declared 'volatile' because it can be changed by the ISR
// at any time.
volatile unsigned int pulseCount = 0;

// Variables for timing the one-second measurement interval.
unsigned long measurementStartTime;

// This is the Interrupt Service Routine (ISR).
// It's a tiny, fast function that runs every time a rising edge
// is detected on CLKOUT_PIN. All it does is increment our counter.
static volatile bool triggerLogic = false;

void countThePulses() {
  triggerLogic = !triggerLogic; // Flip the state on every pulse (1024 Hz)
  if (!triggerLogic) {
    return; // Do nothing on every other pulse
  }
  pulseCount++;
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Set the CLKOUT pin as an input with an internal pull-up resistor.
  // The pull-up isn't strictly necessary as the RTC provides a push-pull
  // signal, but it's good practice.
  pinMode(CLKOUT_PIN, INPUT_PULLUP);

  // Attach the interrupt.
  // digitalPinToInterrupt(CLKOUT_PIN) correctly maps pin 2 to interrupt 0.
  // countThePulses is the function to call when the interrupt occurs.
  // RISING means the interrupt triggers only on the rising edge of the clock signal.
  attachInterrupt(digitalPinToInterrupt(CLKOUT_PIN), countThePulses, RISING);

  Serial.println("Frequency Counter Started. Measuring pulses on Pin 2...");
  Serial.println("The count should be very close to 512 each second.");

  // Record the start time for our first measurement window.
  measurementStartTime = millis();
}

void loop() {
  // Check if one second (1000 milliseconds) has passed.
  if (millis() - measurementStartTime >= 1000) {

    // --- Critical Section: Safely read the pulse count ---
    
    // Temporarily disable interrupts to ensure pulseCount doesn't
    // change while we read it and print it.
    noInterrupts();
    
    // Print the number of pulses we counted in the last second.
    Serial.print("Frequency: ");
    Serial.print(pulseCount);
    Serial.println(" Hz");
    
    // Reset the counter for the next measurement interval.
    pulseCount = 0;
    
    // Re-enable interrupts so we can start counting again.
    interrupts();

    // Update the start time for the next one-second window.
    measurementStartTime = millis();
  }
}