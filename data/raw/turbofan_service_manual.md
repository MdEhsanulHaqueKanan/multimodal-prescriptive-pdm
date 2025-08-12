# Turbofan Engine Model T-1000 Service Manual

## Section 1: Safety Precautions
- **1.1:** Always disconnect the main power supply before performing any maintenance.
- **1.2:** Use certified personal protective equipment (PPE), including safety glasses and gloves.
- **1.3:** Ensure the engine has cooled to ambient temperature (below 40°C) before touching internal components.

## Section 3: Mechanical Procedures
### 3.4: Bearing Assembly Replacement (Part #B-12-XT)
This procedure is critical when RUL predictions are low or when fault code P0301 is detected.

**Required Tools:**
- Torque wrench (calibrated)
- Bearing puller set (Model #BP-5)
- LN-35 industrial grease
- Replacement Bearing Assembly Kit (Part #B-12-XT)

**Steps:**
1.  Verify main power is disconnected and locked out.
2.  Remove the main turbine housing by unscrewing the eight (8) retaining bolts.
3.  Using the bearing puller, carefully extract the old bearing assembly. Do not use a hammer or apply blunt force.
4.  Clean the bearing seat thoroughly. Inspect for any signs of scoring or damage.
5.  Apply a thin layer of LN-35 grease to the new bearing assembly.
6.  Gently press the new bearing into the seat until it is fully seated.
7.  Replace the main turbine housing. Tighten the retaining bolts in a star pattern to a torque of 85 Nm.
8.  Restore power and run the engine in test mode for 15 minutes, monitoring vibration sensor 7 and temperature sensor 11. Readings should be within normal operating parameters.

## Section 5: Diagnostic Trouble Codes (DTC)
### 5.1: Error Code P0300 - Random/Multiple Misfire
- **Description:** This code indicates that the engine control unit (ECU) has detected random or multiple cylinder misfires.
- **Common Causes:**
  - Faulty spark plugs
  - Damaged fuel injectors
  - Low fuel pressure
  - Vacuum leak
- **Troubleshooting Steps:**
  1.  Check historical sensor data for anomalies in fuel flow (sensor 4) and pressure (sensor 14).
  2.  Inspect spark plugs for wear and tear. Replace if necessary.
  3.  Perform a fuel pressure test. The system should maintain a pressure of 400-450 kPa.
  4.  If the issue persists, connect the diagnostic tool to check the fuel injector signal for each cylinder.
### 5.2: Error Code P0420 - Overstrain Failure
- **Description:** This code is triggered when the system detects torque and rotational speed values that exceed the component's design limits for a sustained period. This is often associated with the "Overstrain Failure" prediction.
- **Common Causes:**
  - Incorrect operational parameters (e.g., tool feed rate too high).
  - Use of dull or improper tooling.
  - Mechanical obstruction or jam in the system.
- **Troubleshooting Steps:**
  1.  **Immediate Action:** Reduce the load on the machine and bring it to a safe stop.
  2.  Review the operational parameters for the last 100 cycles to identify any settings that exceed the recommended limits for the material being processed.
  3.  Inspect the tool for wear or damage. Replace if the tool wear value is above 200 minutes.
  4.  Check for any physical obstructions in the machine's path of operation.