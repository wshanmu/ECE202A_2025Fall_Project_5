#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>

// ==========================================
// MOCKING HARDWARE ENVIRONMENT
// ==========================================

// 1. Mock the RP2040 Hardware Struct
struct SIO_HW_MOCK {
    uint32_t gpio_set; // Writing here simulates setting a pin HIGH
    uint32_t gpio_clr; // Writing here simulates setting a pin LOW
};

// Create the global pointer expected by your code
SIO_HW_MOCK mock_sio_storage;
SIO_HW_MOCK* sio_hw = &mock_sio_storage;

// 2. Mock Constants
#define CONTROL_PIN 26
const uint32_t kasami_bit_len = 24; // Small length for debugging logic
// Dummy sequence: 0x88 (10001000), 0x1B (00011011), 0xE4 (11100100)
// Note: Logic is LSB first. 
// 0x88 -> bits: 0,0,0,1,0,0,0,1
// 0x1B -> bits: 1,1,0,1,1,0,0,0
// 0xE4 -> bits: 0,0,1,0,0,1,1,1
const uint8_t kasami_sequence[] = { 0x88, 0x1B, 0xE4}; 
// 3. Mock Global Variables from your Sketch
volatile uint32_t isrCounter = 0;
volatile uint32_t edgeCounter = 0;
volatile bool cycleCompleted = false;
unsigned long cycleEndTime = 0;

// Mock millis() - simplistic version
unsigned long current_virtual_time = 0;
unsigned long millis() { return current_virtual_time; }

// ==========================================
// YOUR LOGIC (Copied from Sketch)
// ==========================================

// // Copy of your get_bit function
// uint8_t get_bit(uint32_t bit_idx) {
//   if (bit_idx >= kasami_bit_len) return 0; 
//   uint32_t byte_idx = bit_idx / 8;
//   uint8_t bit_offset = bit_idx % 8;
//   return (kasami_sequence[byte_idx] >> bit_offset) & 0x01;
// }
uint8_t get_bit(uint32_t bit_idx) {
  if (bit_idx >= kasami_bit_len) return 0; // safety
  
  uint32_t byte_idx = bit_idx / 8;
  uint8_t bit_offset = bit_idx % 8;
  
  // CHANGED: Shift by (7 - offset) to read from MSB down to LSB
  return (kasami_sequence[byte_idx] >> (7 - bit_offset)) & 0x01;
}

// Copy of your ISR (Interrupt Service Routine)
void onRtcPulse() {
  uint32_t edge = edgeCounter++;

  // Only update outputs on even edges (effective 512 Hz symbol rate)
  if (edge & 0x01) {
    return;
  }

  // Get current bit index and increment for next symbol
  uint32_t idx = isrCounter;
  isrCounter++;

  // Handle wrap-around
  if (isrCounter >= kasami_bit_len) {
    isrCounter = 0;
    cycleCompleted = true; 
    cycleEndTime = millis();
    edgeCounter = 0; 
  }

  uint8_t bit = get_bit(idx);

  // Update control pin mock
  if (bit == 1) {
    sio_hw->gpio_set = (1u << CONTROL_PIN); 
    // Clear the clear register to simulate strict state (optional for mock logic)
    sio_hw->gpio_clr = 0; 
  } else {
    sio_hw->gpio_clr = (1u << CONTROL_PIN);
    // Clear the set register
    sio_hw->gpio_set = 0;
  }
}

// ==========================================
// TEST RUNNER (The "Virtual Board")
// ==========================================

int main() {
    std::cout << "--- Starting Simulation ---" << std::endl;
    std::cout << "Target: 512 symbols/sec from 1024Hz Clock" << std::endl;
    
    // Check first few bits manually to verify "get_bit" logic
    // 0x88 is 1000 1000 in binary. 
    // LSB (offset 0) is 0. Offset 3 is 1. Offset 7 is 1.
    std::cout << "Bit 0 check (expect 0): " << (int)get_bit(0) << std::endl;
    std::cout << "Bit 3 check (expect 1): " << (int)get_bit(3) << std::endl;

    // Simulate 40 clock ticks (should produce 20 symbols)
    // We expect the sequence based on 0x88, 0x01:
    // Bits: 0, 0, 0, 1, 0, 0, 0, 1 | 1, 0...
    
    for (int tick = 0; tick < 60; tick++) {
        // 1. Simulate Time Passing (approx 0.97ms per tick for 1024Hz)
        if (tick % 1 == 0) current_virtual_time++; 

        // 2. Call the ISR (Simulate Rising Edge)
        onRtcPulse();

        // 3. Capture State
        bool pinHigh = (sio_hw->gpio_set & (1u << CONTROL_PIN));
        bool pinLow = (sio_hw->gpio_clr & (1u << CONTROL_PIN));
        
        // 4. Print Debug Info
        std::cout << "Tick " << std::setw(2) << tick 
                  << " | EdgeCnt: " << std::setw(2) << edgeCounter 
                  << " | ISRIdx: " << std::setw(2) << isrCounter;

        if (tick % 2 == 0) {
            std::cout << " | ACTIVE EDGE -> Output: " << (pinHigh ? "HIGH (1)" : "LOW  (0)");
        } else {
            std::cout << " | Skip Edge";
        }
        
        if (cycleCompleted) {
             std::cout << " [CYCLE COMPLETE TRIGGERED]";
             // Reset manual flag for simulation visualization
             cycleCompleted = false; 
        }
        std::cout << std::endl;
    }

    return 0;
}