#include <Wire.h>

// The 7-bit I2C address for the RV-3032-C7 is 0x51
#define RV3032_ADDRESS 0x51

// Register Addresses
#define REG_CONTROL_1       0x10
#define REG_EE_COMMAND      0x3F
#define REG_TEMP_LSB        0x0E
#define REG_EEPROM_PMU      0xC0
#define REG_EEPROM_CLKOUT_2 0xC3

void setup() {
  Wire.begin();
  Serial.begin(9600);
  while (!Serial);

  Serial.println("Configuring RV-3032-C7 for persistent 1024Hz clock output...");

  // --- Step 1: Disable Automatic EEPROM Refresh ---
  // To write to EEPROM, auto-refresh must be disabled by setting EERD bit (bit 2) to 1.
  // Register Control 1 (0x10) = 0b00000100 = 0x04
  Serial.println("1. Disabling auto-refresh.");
  Wire.beginTransmission(RV3032_ADDRESS);
  Wire.write(REG_CONTROL_1);
  Wire.write(0x04);
  Wire.endTransmission();
  delay(10);

  // --- Step 2: Configure Clock Output Registers (in RAM mirror) ---
  // a) Ensure CLKOUT is enabled in the Power Management Unit (PMU) register.
  // NCLKE bit (bit 6) = 0. We write 0x00 to keep all other settings at default.
  Serial.println("2a. Setting EEPROM PMU register (0xC0) to enable CLKOUT.");
  Wire.beginTransmission(RV3032_ADDRESS);
  Wire.write(REG_EEPROM_PMU);
  Wire.write(0x00);
  Wire.endTransmission();
  delay(10);

  // b) Set the frequency to 1024 Hz in the CLKOUT2 register.
  // OS bit (bit 7) = 0 (XTAL mode)
  // FD bits (6:5) = 01 (1024 Hz)
  // Register EEPROM Clkout 2 (0xC3) = 0b00100000 = 0x20
  Serial.println("2b. Setting EEPROM CLKOUT2 register (0xC3) for 1024Hz output.");
  Wire.beginTransmission(RV3032_ADDRESS);
  Wire.write(REG_EEPROM_CLKOUT_2);
  Wire.write(0x20);
  Wire.endTransmission();
  delay(10);

  // --- Step 3: Update the EEPROM ---
  // Write command 0x11 to the EE Command register (0x3F) to copy all
  // configuration RAM registers (C0h-CAh) to the EEPROM.
  Serial.println("3. Sending Update command to write RAM to EEPROM.");
  Wire.beginTransmission(RV3032_ADDRESS);
  Wire.write(REG_EE_COMMAND);
  Wire.write(0x11);
  Wire.endTransmission();

  // Wait for the EEPROM write to complete. This takes up to 46ms.
  // A simple delay is sufficient.
  delay(50);

  // --- Step 4: Re-enable Automatic EEPROM Refresh ---
  // Set EERD bit (bit 2) back to 0.
  // Register Control 1 (0x10) = 0x00
  Serial.println("4. Re-enabling auto-refresh.");
  Wire.beginTransmission(RV3032_ADDRESS);
  Wire.write(REG_CONTROL_1);
  Wire.write(0x00);
  Wire.endTransmission();

  Serial.println("\nConfiguration complete!");
  Serial.println("The RV-3032-C7 should now output a 1024Hz signal on the CLKOUT pin.");
  Serial.println("This setting will persist after a power cycle.");
}

void loop() {
  // The configuration is done. Nothing further is needed.
}