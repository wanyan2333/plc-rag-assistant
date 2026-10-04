"""Generate two fictional demo manuals into data/raw/ so the project runs end-to-end
without downloading vendor PDFs.

    python -m scripts.generate_demo_manuals

The content is invented ("ACME" products) but follows the structure of real PLC and VFD
manuals: LED tables, diagnostic event IDs, fault-code tables and step-by-step procedures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.pdf_builder import build_manual_pdf

# --------------------------------------------------------------------------- PLC manual

PLC_TITLE = "ACME FlexLogic FX-200 PLC System Manual"

PLC_EVENTS = [
    ["16#02A0", "CPU changed to STOP: STOP switch operated", "Mode selector moved to STOP or STOP command from the engineering station.", "No fault. Set the mode selector to RUN to restart the user program."],
    ["16#02A4", "Cycle time exceeded", "The user program cycle time exceeded the configured maximum cycle time (default 150 ms).", "Optimise the program, move slow tasks to a cyclic interrupt OB or increase the maximum cycle time in the CPU properties."],
    ["16#2522", "Area length error when reading", "The program accessed a data block address outside the defined length.", "Check indirect addressing and array indices; enlarge the data block if needed."],
    ["16#39C4", "Distributed I/O station failure", "A PROFINET or fieldbus I/O station stopped responding.", "Check the network cable, switch ports and power supply of the I/O station. The CPU stays in RUN if OB86 is programmed."],
    ["16#6500", "Backup battery low", "The CR2032 backup battery voltage is below 2.6 V.", "Replace the backup battery within 7 days while the CPU is powered (see section 8.1)."],
    ["16#8085", "I/O module removed or not responding", "A signal module was pulled from the backplane, lost backplane contact or failed.", "Power down, re-seat the module, check the bus connector and verify the hardware configuration."],
    ["16#8086", "Module configuration mismatch", "The module type in the slot differs from the configured hardware.", "Install the configured module type or update the hardware configuration and download it."],
    ["16#8090", "Module parameter assignment error", "Invalid parameters were downloaded to a signal module (for example an unsupported measuring range).", "Correct the module parameters in the engineering tool and download the hardware configuration again."],
    ["16#8131", "Watchdog reset of the CPU", "Internal watchdog triggered due to a firmware or hardware fault.", "Read the diagnostic buffer, update the firmware to the latest version and contact ACME support if the event repeats."],
]

PLC_MODULE_FAULTS = [
    ["E-101", "Analog input wire break", "Input current below 3.6 mA on a 4-20 mA channel.", "Check the transmitter wiring and loop supply; verify the sensor with a loop calibrator."],
    ["E-102", "Analog input overflow", "Signal above the upper limit of the measuring range (more than 22.8 mA).", "Check the sensor range and the module measuring range setting."],
    ["E-110", "Digital output short circuit", "Output current exceeded 1.2 A on a 24 V DC output channel.", "Disconnect the load, measure the load resistance and replace the faulty actuator or cable."],
    ["E-111", "Missing load voltage L+", "The 24 V DC load supply of the output group is missing or below 19.2 V.", "Check the load power supply, fuse and terminal L+ / M wiring."],
    ["E-120", "Module overtemperature", "Internal module temperature above 85 degC.", "Improve cabinet ventilation, keep 25 mm clearance above and below the module and check ambient temperature."],
    ["E-130", "Encoder supply short circuit", "The 24 V encoder supply output of the high-speed counter is overloaded.", "Check the encoder cable for damage and measure encoder current consumption."],
    ["E-140", "Communication module bus error", "CRC errors on the RS-485 port exceeded the threshold.", "Check termination resistors at both bus ends, cable shielding and baud rate settings."],
]

PLC_ELEMENTS = [
    ("h1", "1 Product Overview"),
    ("p", "The ACME FlexLogic FX-200 is a modular programmable logic controller for packaging, conveying and assembly machines. "
          "A system consists of one CPU module, up to eight signal modules on the right-hand side and up to three communication modules on the left-hand side. "
          "The CPU executes the user program cyclically: it reads the process image of inputs, runs the program organisation blocks and then writes the process image of outputs."),
    ("p", "The CPU provides an integrated PROFINET interface with two switched ports, 14 digital inputs (24 V DC), 10 digital outputs and two analog inputs (0-10 V). "
          "Program memory is 150 KB and data memory is 1 MB. A CR2032 backup battery keeps the real-time clock running when the CPU is switched off."),
    ("h2", "1.1 Operating modes"),
    ("p", "The CPU has three operating modes: STOP, STARTUP and RUN. In STOP the user program is not executed and all outputs are set to their configured substitute values. "
          "In STARTUP the startup organisation blocks run once and the input image is initialised. In RUN the program cycle executes continuously. "
          "The mode selector on the front of the CPU, or a command from the engineering station, changes the operating mode."),
    ("h2", "1.2 Safety information"),
    ("p", "Only qualified personnel may install, wire or service the controller. Before opening the control cabinet or touching any terminal, switch off the supply and apply lockout/tagout according to the plant safety procedure. "
          "Verify the absence of voltage with a suitable tester. The FX-200 is not a safety controller: emergency stop circuits must be implemented with certified safety relays or a safety PLC."),
    ("h1", "2 Installation and Wiring"),
    ("p", "Mount the controller horizontally on a 35 mm DIN rail inside an enclosure rated at least IP54. Keep a clearance of 25 mm above and below the modules for convection cooling. "
          "The permitted ambient temperature is 0 to 55 degC for horizontal mounting and 0 to 45 degC for vertical mounting."),
    ("p", "Supply the CPU with 24 V DC (permissible range 20.4 to 28.8 V DC). Use a separate fuse of 2 A for the CPU supply and separate fuses for each output load group. "
          "Connect the functional earth terminal to the cabinet earth bar with a cable of at least 2.5 mm2 and keep it as short as possible."),
    ("p", "Use shielded twisted-pair cable for analog signals and encoder signals. Ground the cable shield at the cabinet entry using a shield clamp. "
          "Route signal cables separately from motor cables and power cables; keep a minimum distance of 200 mm or use a separating metal divider."),
    ("h1", "3 LED Status Indicators"),
    ("p", "The CPU has three status LEDs (RUN/STOP, ERROR and MAINT) and one LINK/RX-TX LED per PROFINET port. Signal modules have a DIAG LED and one channel LED per input or output. "
          "The table below lists the meaning of the CPU LED combinations."),
    ("table", ["LED", "State", "Meaning"], [
        ["RUN/STOP", "Steady green", "CPU is in RUN, user program running."],
        ["RUN/STOP", "Steady yellow", "CPU is in STOP."],
        ["RUN/STOP", "Flashing green/yellow", "CPU is in STARTUP."],
        ["ERROR", "Flashing red", "A diagnostic event is active, for example a module fault or a program error. Read the diagnostic buffer."],
        ["ERROR", "Steady red", "Hardware fault of the CPU. Cycle the power; if the LED stays on, replace the CPU."],
        ["MAINT", "Steady yellow", "Maintenance demanded, for example backup battery low or memory card needs replacing."],
        ["MAINT", "Flashing yellow", "Firmware update in progress. Do not switch off the supply."],
        ["All LEDs", "Flashing simultaneously", "Station identification requested by the engineering tool (flash test)."],
    ]),
    ("p", "A module DIAG LED flashing red indicates a channel or module fault such as a wire break or missing load voltage. A steady green DIAG LED means the module is configured and exchanging data with the CPU."),
    ("h1", "4 Diagnostics"),
    ("p", "Every fault detected by the CPU or a module is written as a diagnostic event to the diagnostic buffer. The buffer stores the latest 50 events with a time stamp, "
          "an event ID in the format 16#xxxx and a plain-text description. The diagnostic buffer is retained through power cycles as long as the backup battery is healthy."),
    ("p", "To read the buffer, connect the engineering station, go online and open Online and Diagnostics, then select Diagnostic buffer. Alternatively the integrated web server shows the buffer on the Diagnostics page. "
          "Events are listed with the most recent at the top; always investigate the first event in a sequence, because follow-up events are often only consequences of the original fault."),
    ("h2", "4.1 Diagnostic event IDs"),
    ("table", ["Event ID", "Description", "Cause", "Remedy"], PLC_EVENTS),
    ("h1", "5 Module Fault Codes"),
    ("p", "Signal modules report channel faults with a fault code shown in the engineering tool and in the diagnostic buffer. The module DIAG LED flashes red while a fault code is active."),
    ("table", ["Fault code", "Description", "Cause", "Remedy"], PLC_MODULE_FAULTS),
    ("h1", "6 Troubleshooting Procedures"),
    ("h2", "6.1 CPU does not go to RUN"),
    ("steps", [
        "Check that the mode selector is in the RUN position.",
        "Check the ERROR LED. If it flashes red, read the diagnostic buffer and correct the first reported event.",
        "Verify that the hardware configuration in the project matches the installed modules (event 16#8086 indicates a mismatch).",
        "Compile and download the complete project, including the hardware configuration.",
        "If the CPU still remains in STOP, reset the CPU to factory settings and download the project again.",
    ]),
    ("h2", "6.2 Loss of communication with an I/O station"),
    ("steps", [
        "Check the LINK LED on the CPU port and on the I/O station. If the LED is off, there is no physical connection.",
        "Inspect the Ethernet cable and connectors; replace damaged cables. Industrial cables must not exceed 100 m between two devices.",
        "Verify that the device name and IP address of the station match the project configuration.",
        "Check the 24 V supply of the I/O station.",
        "Use the topology view to locate the interrupted port, then acknowledge event 16#39C4 once communication is restored.",
    ]),
    ("h2", "6.3 Analog input shows wrong value"),
    ("steps", [
        "Check the module DIAG LED and the fault code (E-101 wire break or E-102 overflow).",
        "Verify that the configured measuring range matches the sensor (for example 4-20 mA versus 0-10 V).",
        "Measure the signal at the terminal with a calibrated multimeter.",
        "Check the cable shield is grounded at one end only to avoid ground loops.",
    ]),
    ("p", "If a fault cannot be resolved with these procedures, record the diagnostic buffer, the firmware versions and the hardware configuration and contact ACME technical support."),
    ("h1", "7 Communication"),
    ("p", "The integrated PROFINET interface supports up to 16 I/O devices with an update time of 1 to 512 ms. The default IP address of a new CPU is 192.168.0.1 with subnet mask 255.255.255.0. "
          "The RS-485 communication module CM-485 supports Modbus RTU master and slave at baud rates from 1200 to 115200 bit/s."),
    ("p", "For RS-485 networks, install a 120 ohm termination resistor at both physical ends of the bus and nowhere else. The maximum cable length is 1000 m at 115200 bit/s with twisted-pair cable. "
          "Repeated CRC errors (fault code E-140) usually indicate missing termination, a damaged shield or two devices with the same address."),
    ("h1", "8 Maintenance"),
    ("h2", "8.1 Replacing the backup battery"),
    ("p", "The backup battery (type CR2032, 3 V) has a typical life of five years. When the MAINT LED lights yellow and event 16#6500 is reported, replace the battery within seven days. "
          "Replace the battery while the CPU is powered so that the real-time clock and the diagnostic buffer are not lost."),
    ("steps", [
        "Open the front cover of the CPU.",
        "Pull out the battery holder using the tab.",
        "Insert a new CR2032 battery with the positive side facing up.",
        "Push the holder back until it clicks and close the cover. The MAINT LED switches off within 10 seconds.",
    ]),
    ("h2", "8.2 Firmware update"),
    ("p", "Firmware updates are installed from a memory card or via the engineering tool. During the update the MAINT LED flashes yellow and the CPU is in STOP. "
          "Never switch off the supply during a firmware update, otherwise the CPU may become inoperable and must be returned for repair. The update takes approximately three minutes."),
]

# --------------------------------------------------------------------------- VFD manual

VFD_TITLE = "ACME VectorDrive VD-500 Variable Frequency Drive Troubleshooting Guide"

VFD_FAULTS = [
    ["F-0001", "Overcurrent during acceleration", "Acceleration ramp too short, motor or cable short circuit, or the load is jammed.", "Increase P1120 acceleration time, check motor insulation with a megohmmeter and free the mechanical load."],
    ["F-0002", "Overcurrent at constant speed", "Sudden load change or motor phase-to-earth fault.", "Check the mechanical load and motor cable; verify motor data P0304 to P0310."],
    ["F-0003", "Overcurrent during deceleration", "Deceleration ramp too short for the load inertia.", "Increase P1121 deceleration time or install a braking resistor."],
    ["F-0011", "DC bus overvoltage", "Regenerative energy from an overhauling load or too short deceleration ramp; mains voltage too high.", "Increase deceleration time, enable the Vdc-max controller (P1240 = 1) or connect a braking resistor; check the mains voltage."],
    ["F-0012", "DC bus undervoltage", "Mains supply failure, phase loss or supply voltage below 320 V.", "Check all three mains phases, fuses and contactors; measure the supply voltage under load."],
    ["F-0021", "Earth fault", "Insulation failure between a motor phase and earth.", "Disconnect the motor cable and measure insulation resistance; it must be above 1 Mohm at 500 V DC."],
    ["F-0030", "Inverter overtemperature", "Heat sink temperature above 95 degC: blocked fan, dirty heat sink or ambient above 45 degC.", "Check the cooling fan, clean the heat sink and reduce the switching frequency P1800."],
    ["F-0031", "Motor overtemperature (I2t)", "Motor overloaded for too long or motor data incorrect.", "Check the load, verify rated current P0305 and allow the motor to cool before restarting."],
    ["F-0042", "Fieldbus communication timeout", "No telegram from the PLC within the monitoring time P2040 (default 100 ms).", "Check the fieldbus cable and connectors, PLC status and the monitoring time setting."],
    ["F-0051", "Encoder feedback loss", "Encoder signal missing in closed-loop vector control.", "Check encoder wiring, encoder supply (5 V or 24 V) and the setting of P0400."],
    ["F-0060", "Motor phase loss", "One output phase is open: loose terminal or broken cable.", "Check the U, V, W terminals and the motor cable continuity."],
    ["F-0085", "External fault", "Digital input configured as external fault (P2106) is active.", "Check the external device wired to that input, for example a motor thermostat."],
]

VFD_ALARMS = [
    ["A-0501", "Current limit active", "The drive limits output current to protect itself. Reduce load or increase ramp times."],
    ["A-0502", "Overvoltage limit active", "The Vdc-max controller is extending the ramp to avoid F-0011."],
    ["A-0910", "Fan maintenance due", "The cooling fan has exceeded 40 000 operating hours. Replace the fan at the next shutdown."],
]

VFD_ELEMENTS = [
    ("h1", "1 About this Guide"),
    ("p", "This guide helps maintenance technicians diagnose and correct faults on the ACME VectorDrive VD-500 series (0.75 kW to 30 kW, 380-480 V three-phase). "
          "It describes the operator panel, fault and alarm codes, parameter settings related to protection functions and step-by-step troubleshooting procedures."),
    ("h2", "1.1 Electrical safety"),
    ("p", "Dangerous voltage remains on the DC bus capacitors after the drive has been disconnected from the mains. Wait at least 5 minutes after switching off before working on the drive, "
          "then measure the DC bus voltage between terminals DC+ and DC- and confirm it is below 50 V DC. Apply lockout/tagout to the mains isolator. "
          "Never disconnect the motor cable while the drive is running, and never perform insulation tests on the drive terminals themselves."),
    ("h1", "2 Operator Panel and Status"),
    ("p", "The basic operator panel BOP-5 shows the actual frequency, faults (prefix F) and alarms (prefix A). A fault stops the motor and must be acknowledged; an alarm is a warning and the drive keeps running. "
          "To acknowledge a fault, press the FN key on the panel, apply a rising edge to the digital input set in P2103 or send the acknowledge bit via the fieldbus control word."),
    ("table", ["LED", "State", "Meaning"], [
        ["READY", "Steady green", "Drive ready, no fault."],
        ["READY", "Flashing green", "Drive running."],
        ["FAULT", "Steady red", "Fault active, motor stopped. See fault code on the panel."],
        ["FAULT", "Flashing red", "Alarm active, drive continues to run."],
        ["COM", "Flashing green", "Fieldbus data exchange active."],
    ]),
    ("h1", "3 Fault Codes"),
    ("p", "The following faults switch off the inverter immediately. The last eight faults are stored in parameter r0947 together with the operating hours at which they occurred."),
    ("table", ["Fault code", "Description", "Cause", "Remedy"], VFD_FAULTS),
    ("h1", "4 Alarm Codes"),
    ("p", "Alarms warn about conditions that may lead to a fault. They are cleared automatically when the cause disappears."),
    ("table", ["Alarm code", "Description", "Remedy"], VFD_ALARMS),
    ("h1", "5 Protection Parameters"),
    ("p", "The following parameters influence how the drive reacts to overload and supply conditions. Change them only after understanding the consequences for the driven machine."),
    ("table", ["Parameter", "Name", "Default", "Notes"], [
        ["P1120", "Acceleration time", "10 s", "Ramp-up time from 0 to maximum frequency. Too short causes F-0001."],
        ["P1121", "Deceleration time", "10 s", "Ramp-down time. Too short causes F-0003 or F-0011."],
        ["P1240", "Vdc controller", "1", "1 = Vdc-max controller enabled, extends deceleration automatically."],
        ["P1800", "Switching frequency", "4 kHz", "Higher values reduce motor noise but increase inverter losses."],
        ["P2040", "Fieldbus monitoring time", "100 ms", "0 disables the telegram timeout monitoring (not recommended)."],
        ["P0305", "Rated motor current", "-", "Enter the value from the motor rating plate."],
    ]),
    ("h1", "6 Troubleshooting Procedures"),
    ("h2", "6.1 Drive trips with overcurrent (F-0001 to F-0003)"),
    ("steps", [
        "Note in which phase the trip occurs: acceleration (F-0001), constant speed (F-0002) or deceleration (F-0003).",
        "Isolate the drive, apply lockout/tagout and wait 5 minutes for the DC bus to discharge.",
        "Disconnect the motor cable at the drive and measure the motor insulation resistance phase-to-earth with a 500 V megohmmeter (minimum 1 Mohm).",
        "Rotate the motor shaft by hand to check for mechanical jamming or bearing damage.",
        "Check that the motor data P0304 to P0310 match the rating plate and run the motor identification (P1910 = 1).",
        "Increase the ramp times P1120 / P1121 in steps of 50 percent and test again.",
    ]),
    ("h2", "6.2 DC bus overvoltage (F-0011)"),
    ("steps", [
        "Check whether the trip occurs during braking. If yes, the load is feeding energy back into the drive.",
        "Enable the Vdc-max controller with P1240 = 1 and increase the deceleration time P1121.",
        "For high-inertia loads such as fans or centrifuges, install a braking resistor sized for the duty cycle.",
        "Measure the mains voltage; values above 528 V can also cause this fault.",
    ]),
    ("h2", "6.3 Inverter overtemperature (F-0030)"),
    ("steps", [
        "Check that the cooling fan runs when the drive is enabled.",
        "Clean dust from the heat sink with dry compressed air.",
        "Measure the cabinet temperature; the drive is rated for 45 degC without derating.",
        "Reduce the switching frequency P1800 or the load.",
    ]),
    ("p", "After any repair, acknowledge the fault, run the motor without load first and then check the motor current under normal load against the rated current P0305."),
    ("h1", "7 Maintenance Intervals"),
    ("p", "Inspect the drive every 12 months: tighten power terminals to the specified torque, clean the heat sink and check the cooling fan. "
          "Replace the cooling fan every 40 000 operating hours (alarm A-0910). DC bus capacitors in drives stored without power for more than two years must be reformed before commissioning by applying reduced voltage for 1 hour."),
]


def generate(out_dir: Path) -> list[Path]:
    return [
        build_manual_pdf(out_dir / "ACME_FX200_PLC_System_Manual.pdf", PLC_TITLE, PLC_ELEMENTS,
                         PLC_TITLE, "ACME Automation Ltd - Edition 03/2026"),
        build_manual_pdf(out_dir / "ACME_VD500_VFD_Troubleshooting_Guide.pdf", VFD_TITLE, VFD_ELEMENTS,
                         "ACME VectorDrive VD-500 - Troubleshooting Guide", "ACME Drives - Document VD500-TG-EN Rev B"),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    for path in generate(args.out):
        print(f"wrote {path}")
