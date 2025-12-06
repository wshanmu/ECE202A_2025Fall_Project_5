/*
 * PN Code Generator based on external RTC clock - RP2040 DEBUG VERSION
 *
 * This sketch generates a PN code sequence on PIN 26 (control signal).
 * The timing is driven by an external 1024 Hz clock signal from an RV-3032-C7 RTC,
 * connected to RTC_CLK_PIN (PIN 5).
 *
 * DEBUG FEATURES:
 * - Measures RTC clock input frequency 3 times at startup
 * - Displays frequency measurements before starting PN code generation
 * - Helps verify RTC clock signal is working correctly
 *
 * The 1024 Hz input is divided by two in software to produce an effective
 * symbol rate of 512 Hz (1 symbol per ~1.95 ms).
 *
 * PLATFORM: Adafruit Feather RP2040
 * - RTC clock input on PIN 5
 * - I2C pins (PIN 2, PIN 3) manually disabled to prevent interference
 * - Single control output on PIN 26 (HIGH = 1, LOW = 0)
 *
 * Expected sequence time: ~127.99 seconds (65535 bits / 512 Hz)
 */

#include "kasami_packed_16bit_dup8_120.h"

// Pin Definitions for RP2040
#define BTN_STOP_ALARM 0  // Stop button (optional)
#define CONTROL_PIN 26    // Single control output pin
#define RTC_CLK_PIN 5     // Input pin for the 1024 Hz signal from the RTC
#define I2C_SDA_PIN 2     // I2C SDA - will be disabled
#define I2C_SCL_PIN 3     // I2C SCL - will be disabled

// --- Global variables for PN code generation ---
volatile uint32_t isrCounter = 0;      // Bit index in sequence (0 to 65534)
volatile uint32_t edgeCounter = 0;     // Counts every RTC edge (1024 Hz)

// --- Global variables for frequency measurement ---
volatile uint32_t freqMeasureCounter = 0;
volatile bool freqMeasureComplete = false;

// --- Global variables for verification ---
volatile bool isrAttached = true;
volatile bool cycleCompleted = false;  // Flag to signal a full cycle is done
volatile unsigned long cycleEndTime = 0;    // Captured in ISR at exact wrap moment
unsigned long cycleStartTime = 0;      // Stores the start time of a cycle
unsigned long lastCycleDuration = 0;   // Store last cycle duration for verification

// Mutex for RP2040
auto_init_mutex(isrMutex);

// Mode control
enum OperationMode {
  MODE_FREQ_MEASURE,
  MODE_PN_GENERATION
};
volatile OperationMode currentMode = MODE_FREQ_MEASURE;


// Read a bit from the packed sequence
uint8_t get_bit(uint32_t bit_idx) {
  if (bit_idx >= kasami_bit_len) return 0; // safety
  uint32_t byte_idx = bit_idx / 8;
  uint8_t bit_offset = bit_idx % 8;
  return (kasami_sequence[byte_idx] >> bit_offset) & 0x01;
}

// Interrupt Service Routine for frequency measurement
void onRtcPulse_FreqMeasure() {
  freqMeasureCounter++;
}

// Interrupt Service Routine - triggered by the external 1024 Hz clock
// Optimized for minimal latency and atomic pin updates
void onRtcPulse() {
  uint32_t edge = edgeCounter++;

  // Only update outputs on even edges (effective 512 Hz symbol rate)
  if (edge & 0x01) {
    return;
  }

  // Get current bit index and increment for next symbol
  uint32_t idx = isrCounter;
  isrCounter++;

  // Handle wrap-around at sequence end for seamless periodic execution
  if (isrCounter >= kasami_bit_len) {
    isrCounter = 0;
    cycleCompleted = true;  // Signal that we've completed one full sequence
    cycleEndTime = millis(); // Capture timing at EXACT wrap moment
    edgeCounter = 0;         // Reset edge counter for next cycle measurement
  }

  // Read the bit and update control pin
  uint8_t bit = get_bit(idx);

  // Update control pin using direct register access for RP2040
  // RP2040 has sio_hw->gpio_set and sio_hw->gpio_clr for atomic operations
  if (bit == 1) {
    sio_hw->gpio_set = (1u << CONTROL_PIN);  // Set CONTROL_PIN HIGH
  } else {
    sio_hw->gpio_clr = (1u << CONTROL_PIN);  // Set CONTROL_PIN LOW
  }
}

// Function to measure RTC clock frequency
float measureRtcFrequency(int measurementTime_ms) {
  freqMeasureCounter = 0;

  // Attach interrupt for frequency measurement
  attachInterrupt(digitalPinToInterrupt(RTC_CLK_PIN), onRtcPulse_FreqMeasure, RISING);

  // Wait for the measurement period
  unsigned long startTime = millis();
  delay(measurementTime_ms);
  unsigned long endTime = millis();

  // Detach interrupt
  detachInterrupt(digitalPinToInterrupt(RTC_CLK_PIN));

  // Calculate frequency
  uint32_t pulseCount = freqMeasureCounter;
  float actualTime_s = (endTime - startTime) / 1000.0;
  float frequency = pulseCount / actualTime_s;

  return frequency;
}

void setup() {
  Serial.begin(115200);
  delay(1000); // Give serial time to initialize

  Serial.println("===========================================");
  Serial.println("PN Code Generator - RP2040 DEBUG VERSION");
  Serial.println("===========================================");
  Serial.println("Platform: Adafruit Feather RP2040");
  Serial.println();

  // CRITICAL: Disable I2C pins to prevent interference with RTC_CLK_IN
  // Set PIN 2 and PIN 3 to INPUT mode (high impedance)
  pinMode(I2C_SDA_PIN, INPUT);
  pinMode(I2C_SCL_PIN, INPUT);
  Serial.println("[SETUP] I2C pins disabled (set to INPUT mode)");

  // Set button and output pin modes
  pinMode(BTN_STOP_ALARM, INPUT_PULLUP);
  pinMode(CONTROL_PIN, OUTPUT);

  // Set initial state for control pin
  digitalWrite(CONTROL_PIN, LOW);
  Serial.println("[SETUP] Control pin initialized to LOW");

  // Set RTC clock pin as input
  pinMode(RTC_CLK_PIN, INPUT);
  Serial.print("[SETUP] RTC Clock Input: PIN ");
  Serial.println(RTC_CLK_PIN);
  Serial.println();

  // ========================================
  // FREQUENCY MEASUREMENT (3 iterations)
  // ========================================
  Serial.println("===========================================");
  Serial.println("FREQUENCY MEASUREMENT");
  Serial.println("===========================================");
  Serial.println("Measuring RTC clock input frequency...");
  Serial.println("Expected: 1024 Hz");
  Serial.println();

  const int measurementTime_ms = 2000; // 2 second measurement period
  float frequencies[3];
  float totalFreq = 0.0;

  for (int i = 0; i < 3; i++) {
    Serial.print("Measurement ");
    Serial.print(i + 1);
    Serial.print("/3: ");
    Serial.flush();

    frequencies[i] = measureRtcFrequency(measurementTime_ms);
    totalFreq += frequencies[i];

    Serial.print(frequencies[i], 2);
    Serial.print(" Hz (Error: ");
    float error = ((frequencies[i] - 1024.0) / 1024.0) * 100.0;
    Serial.print(error, 3);
    Serial.println("%)");

    if (i < 2) {
      delay(500); // Short delay between measurements
    }
  }

  // Calculate average frequency
  float avgFreq = totalFreq / 3.0;
  float avgError = ((avgFreq - 1024.0) / 1024.0) * 100.0;

  Serial.println();
  Serial.println("--- Measurement Summary ---");
  Serial.print("Average Frequency: ");
  Serial.print(avgFreq, 2);
  Serial.print(" Hz (Error: ");
  Serial.print(avgError, 3);
  Serial.println("%)");

  // Check if frequency is within acceptable range (±5%)
  if (abs(avgError) > 5.0) {
    Serial.println();
    Serial.println("WARNING: Frequency error exceeds ±5%");
    Serial.println("Check RTC clock connection and configuration!");
  } else {
    Serial.println();
    Serial.println("Frequency measurement OK!");
  }

  Serial.println("===========================================");
  Serial.println();
  delay(1000);

  // ========================================
  // START PN CODE GENERATION
  // ========================================
  Serial.println("===========================================");
  Serial.println("STARTING PN CODE GENERATION");
  Serial.println("===========================================");
  Serial.println("Generating PN code @ 512 Hz from 1024 Hz RTC");
  Serial.print("Control Output: PIN ");
  Serial.println(CONTROL_PIN);
  Serial.print("Sequence length: ");
  Serial.print(kasami_bit_len);
  Serial.println(" bits");
  Serial.print("Expected cycle time: ");
  Serial.print((kasami_bit_len * 1000.0) / 512.0, 2);
  Serial.println(" ms");
  Serial.println("Will print cycle duration after each sequence...");
  Serial.println("===========================================");

  // Switch to PN generation mode
  currentMode = MODE_PN_GENERATION;

  // Attach interrupt for PN code generation
  attachInterrupt(digitalPinToInterrupt(RTC_CLK_PIN), onRtcPulse, RISING);

  // Record the start time of the very first cycle
  cycleStartTime = millis();
}

void loop() {
  // Check if a full sequence cycle has been completed
  if (cycleCompleted) {
    // Safely reset the flag and read counters + timing
    mutex_enter_blocking(&isrMutex);
    cycleCompleted = false;
    uint32_t edges = edgeCounter;
    unsigned long endTime = cycleEndTime;  // Get the exact timing captured in ISR
    mutex_exit(&isrMutex);

    // Calculate cycle duration using ISR-captured timestamp
    unsigned long duration = endTime - cycleStartTime;
    lastCycleDuration = duration;

    // Calculate timing accuracy
    float expectedTime = (kasami_bit_len * 1000.0) / 512.0;
    float error = ((float)duration - expectedTime) / expectedTime * 100.0;

    // Expected edges = 2 * kasami_bit_len (because we get 1024 Hz input, use every 2nd edge)
    uint32_t expectedEdges = kasami_bit_len * 2;
    int32_t edgeError = edges - expectedEdges;

    Serial.println("========================================");
    Serial.print("CYCLE COMPLETE | Time: ");
    Serial.print(duration);
    Serial.print(" ms | Expected: ");
    Serial.print(expectedTime, 2);
    Serial.println(" ms");
    Serial.print("Timing error: ");
    Serial.print(error, 4);
    Serial.print("% | RTC edges: ");
    Serial.print(edges);
    Serial.print(" (expected: ");
    Serial.print(expectedEdges);
    Serial.print(", error: ");
    Serial.print(edgeError);
    Serial.println(")");
    Serial.println("========================================");

    // Set the start time for the next cycle measurement
    // Use the same timestamp to ensure accurate period measurement
    cycleStartTime = endTime;
  }

  // If the stop button is pressed
  if (digitalRead(BTN_STOP_ALARM) == LOW) {
    // And if the interrupt is still attached
    if (isrAttached) {
      // Detach the interrupt to stop the PN code generation
      detachInterrupt(digitalPinToInterrupt(RTC_CLK_PIN));
      isrAttached = false; // Update flag

      Serial.println("\n========================================");
      Serial.println("PN Code generation STOPPED by user");
      Serial.println("========================================");

      // Print final statistics
      mutex_enter_blocking(&isrMutex);
      uint32_t finalCounter = isrCounter;
      uint32_t finalEdges = edgeCounter;
      mutex_exit(&isrMutex);

      Serial.print("Final bit index: ");
      Serial.println(finalCounter);
      Serial.print("Total edges processed: ");
      Serial.println(finalEdges);
      Serial.print("Total symbols output: ");
      Serial.println(finalEdges / 2);

      if (lastCycleDuration > 0) {
        Serial.print("Last cycle duration: ");
        Serial.print(lastCycleDuration);
        Serial.println(" ms");
      }
      Serial.println("========================================");
    }
  }
}
