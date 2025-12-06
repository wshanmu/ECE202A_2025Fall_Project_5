/*
 * PN Code Verification Sketch - SIMPLE VERSION
 *
 * This sketch verifies the PN code sequence generation logic by:
 * - Reading bits from the packed Kasami sequence
 * - Updating at the correct timing: every 500/1024 seconds (~488.28 us)
 * - Printing bit index and bit value for each update
 *
 * This is a VERIFICATION tool - no external RTC clock required.
 * Uses internal timing (micros()) to simulate the 512 Hz symbol rate.
 *
 * Purpose: Verify get_bit() function and sequence logic are working correctly
 */

#include "kasami_packed_16bit_dup8_120.h"

// Timing configuration
// Symbol period = 2 / 1024 Hz = 1/512 Hz = 1.953125 ms = 1953.125 us
const unsigned long SYMBOL_PERIOD_US = 1953; // Approximately 500/1024 * 1000000 us

// State variables
uint32_t bitIndex = 0;
unsigned long lastUpdateTime = 0;
bool sequenceComplete = false;

// Read a bit from the packed sequence
uint8_t get_bit(uint32_t bit_idx) {
  if (bit_idx >= kasami_bit_len) return 0; // safety
  uint32_t byte_idx = bit_idx / 8;
  uint8_t bit_offset = bit_idx % 8;
  return (kasami_sequence[byte_idx] >> bit_offset) & 0x01;
}

void setup() {
  Serial.begin(115200);
  delay(2000); // Give serial time to initialize

  Serial.println("===========================================");
  Serial.println("PN Code Verification - SIMPLE VERSION");
  Serial.println("===========================================");
  Serial.print("Sequence length: ");
  Serial.print(kasami_bit_len);
  Serial.println(" bits");
  Serial.print("Symbol rate: 512 Hz (period = ");
  Serial.print(SYMBOL_PERIOD_US);
  Serial.println(" us)");
  Serial.println();
  Serial.println("Format: [Bit Index] = Bit Value");
  Serial.println("===========================================");
  Serial.println();

  // Print first 20 bits as a header
  Serial.println("First 20 bits of sequence:");
  for (uint32_t i = 0; i < 20 && i < kasami_bit_len; i++) {
    Serial.print("[");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(get_bit(i));
  }
  Serial.println();
  Serial.println("Starting timed generation...");
  Serial.println("===========================================");
  Serial.println();

  lastUpdateTime = micros();
}

void loop() {
  unsigned long currentTime = micros();

  // Check if it's time to update (every SYMBOL_PERIOD_US microseconds)
  if (currentTime - lastUpdateTime >= SYMBOL_PERIOD_US) {
    lastUpdateTime = currentTime;

    // Get current bit
    uint8_t bit = get_bit(bitIndex);

    // Print bit index and value
    Serial.print("[");
    Serial.print(bitIndex);
    Serial.print("] = ");
    Serial.println(bit);

    // Increment bit index
    bitIndex++;

    // Check for sequence completion
    if (bitIndex >= kasami_bit_len) {
      if (!sequenceComplete) {
        Serial.println();
        Serial.println("===========================================");
        Serial.println("SEQUENCE COMPLETE - Wrapping around...");
        Serial.println("===========================================");
        Serial.println();
        sequenceComplete = true;
      }
      bitIndex = 0; // Wrap around
    }
  }
}
