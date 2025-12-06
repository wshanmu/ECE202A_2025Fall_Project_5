/*
 * PN Code Generator based on external RTC clock with verification.
 *
 * This sketch generates a PN code sequence on PIN1 and PIN2.
 * The timing is driven by an external 1024 Hz clock signal from an RV-3032-C7 RTC,
 * connected to RTC_CLK_PIN.
 *
 * The 1024 Hz input is divided by two in software to produce an effective
 * symbol rate of 512 Hz (1 symbol per ~1.95 ms).
 *
 * This version includes logic to measure and print the time taken to complete
 * one full sequence cycle (16,383 bits). The expected time is ~32 seconds.
 *
 * The process can be stopped with a button attached to BTN_STOP_ALARM.
 */

#include "kasami_packed_120.h"

// Pin Definitions
#define BTN_STOP_ALARM 0  // Stop button is attached to PIN 0 (IO0)
#define PIN1 13           // Output Pin 1
#define PIN2 16           // Output Pin 2
#define RTC_CLK_PIN 14    // Input pin for the 1024 Hz signal from the RTC

// --- Global variables for PN code generation ---
volatile uint32_t isrCounter = 0;
volatile uint32_t edgeCounter = 0;    // Counts every RTC edge (1024 Hz)
portMUX_TYPE isrMux = portMUX_INITIALIZER_UNLOCKED;

// --- Global variables for the external interrupt & verification ---
volatile bool isrAttached = true;
volatile bool cycleCompleted = false; // Flag to signal a full cycle is done
unsigned long cycleStartTime = 0;     // Stores the start time of a cycle
unsigned long lastCycleDuration = 0;  // Store last cycle duration for verification


// Read a bit from the packed sequence
uint8_t get_bit(uint32_t bit_idx) {
  if (bit_idx >= kasami_bit_len) return 0; // safety
  uint32_t byte_idx = bit_idx / 8;
  uint8_t bit_offset = bit_idx % 8;
  return (kasami_sequence[byte_idx] >> bit_offset) & 0x01;
}

// Interrupt Service Routine - triggered by the external 1024 Hz clock
// Optimized for minimal latency and atomic pin updates
void ARDUINO_ISR_ATTR onRtcPulse() {
  portENTER_CRITICAL_ISR(&isrMux);
  
  uint32_t edge = edgeCounter++;
  
  // Only update outputs on even edges (effective 512 Hz symbol rate)
  if (edge & 0x01) {
    portEXIT_CRITICAL_ISR(&isrMux);
    return;
  }
  
  // Get current bit index and increment for next symbol
  uint32_t idx = isrCounter;
  isrCounter++;
  
  // Handle wrap-around at sequence end
  if (isrCounter >= kasami_bit_len) {
    isrCounter = 0;
    cycleCompleted = true;  // Signal that we've completed one full sequence
  }
  
  portEXIT_CRITICAL_ISR(&isrMux);
  
  // Read the bit OUTSIDE the critical section to minimize lock time
  uint8_t bit = get_bit(idx);
  
  // CRITICAL: Update both pins atomically using direct register access
  // This is much faster than digitalWrite() and ensures true inverse outputs
  #ifdef ESP32
    if (bit == 1) {
      GPIO.out_w1tc = (1 << PIN1);  // Clear PIN1 (LOW)
      GPIO.out_w1ts = (1 << PIN2);  // Set PIN2 (HIGH)
    } else {
      GPIO.out_w1ts = (1 << PIN1);  // Set PIN1 (HIGH)
      GPIO.out_w1tc = (1 << PIN2);  // Clear PIN2 (LOW)
    }
  #else
    // Fallback for non-ESP32 platforms
    if (bit == 1) {
      digitalWrite(PIN1, LOW);
      digitalWrite(PIN2, HIGH);
    } else {
      digitalWrite(PIN1, HIGH);
      digitalWrite(PIN2, LOW);
    }
  #endif
}

void setup() {
  Serial.begin(115200);
  
  // Set button and output pin modes
  pinMode(BTN_STOP_ALARM, INPUT_PULLUP);
  pinMode(PIN1, OUTPUT);
  pinMode(PIN2, OUTPUT);

  // Set initial state for output pins
  digitalWrite(PIN1, LOW);
  digitalWrite(PIN2, HIGH);

  // --- External Interrupt Setup ---
  pinMode(RTC_CLK_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(RTC_CLK_PIN), onRtcPulse, RISING);

  // Record the start time of the very first cycle
  cycleStartTime = millis();

  Serial.println("System Initialized.");
  Serial.println("Generating PN code based on an effective 512 Hz clock from external RTC.");
  Serial.println("Will print cycle duration every ~32 seconds...");
  Serial.print("Sequence length: ");
  Serial.print(kasami_bit_len);
  Serial.println(" bits");
  Serial.print("Expected cycle time: ");
  Serial.print((kasami_bit_len * 1000.0) / 512.0);
  Serial.println(" ms");
}

void loop() {
  // Check if a full sequence cycle has been completed
  if (cycleCompleted) {
    // Safely reset the flag
    portENTER_CRITICAL(&isrMux);
    cycleCompleted = false;
    uint32_t edges = edgeCounter;
    portEXIT_CRITICAL(&isrMux);

    unsigned long endTime = millis();
    unsigned long duration = endTime - cycleStartTime;
    lastCycleDuration = duration;

    // Calculate timing accuracy
    float expectedTime = (kasami_bit_len * 1000.0) / 512.0;
    float error = ((float)duration - expectedTime) / expectedTime * 100.0;

    Serial.println("========================================");
    Serial.print("Full sequence completed. Time taken: ");
    Serial.print(duration);
    Serial.println(" ms");
    Serial.print("Expected time: ");
    Serial.print(expectedTime, 2);
    Serial.println(" ms");
    Serial.print("Timing error: ");
    Serial.print(error, 4);
    Serial.println("%");
    Serial.print("Total RTC edges counted: ");
    Serial.println(edges);
    Serial.print("Expected RTC edges: ");
    Serial.println(kasami_bit_len * 2);
    Serial.println("========================================");

    // Set the start time for the next cycle measurement
    cycleStartTime = endTime;
  }

  // If the stop button is pressed
  if (digitalRead(BTN_STOP_ALARM) == LOW) {
    // And if the interrupt is still attached
    if (isrAttached) {
      // Detach the interrupt to stop the PN code generation
      detachInterrupt(digitalPinToInterrupt(RTC_CLK_PIN));
      isrAttached = false; // Update flag
      Serial.println("\nPN Code generation stopped by user.");
      
      // Print final statistics
      portENTER_CRITICAL(&isrMux);
      uint32_t finalCounter = isrCounter;
      uint32_t finalEdges = edgeCounter;
      portEXIT_CRITICAL(&isrMux);
      
      Serial.print("Final bit index: ");
      Serial.println(finalCounter);
      Serial.print("Total edges processed: ");
      Serial.println(finalEdges);
    }
  }
}