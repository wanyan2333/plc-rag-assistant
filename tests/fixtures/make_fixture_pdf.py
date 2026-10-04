"""Generates a small, fictional PLC manual used by the test-suite (no real manuals needed)."""

from __future__ import annotations

from pathlib import Path

from scripts.pdf_builder import build_manual_pdf

FIXTURE_TITLE = "ACME FX-100 Test Controller Manual"
HEADER = "ACME FX-100 Test Controller Manual"
FOOTER = "ACME Automation Ltd - Rev 1.0"

FAULT_TABLE = [
    ["E-101", "Overcurrent on axis 1", "Motor cable short circuit or drive gain set too high.", "Check motor cable insulation, reduce drive gain, then reset the fault."],
    ["E-102", "Overcurrent on axis 2", "Mechanical jam on axis 2.", "Remove the obstruction and check the coupling before restarting."],
    ["E-205", "Encoder signal lost", "Encoder cable disconnected or damaged.", "Re-seat the encoder connector and inspect the cable shield."],
    ["F-0042", "Fieldbus watchdog timeout", "No telegram received from the master within 100 ms.", "Check the network cable, termination resistors and master cycle time."],
    ["16#8085", "I/O module removed during operation", "Module pulled or backplane contact lost.", "Power down, re-insert the module and verify the rack configuration."],
]

ELEMENTS = [
    ("h1", "1 Introduction"),
    ("p", "The FX-100 is a compact programmable logic controller for small machines. It provides 16 digital inputs, 12 digital outputs and one fieldbus interface."),
    ("p", "The controller monitors the inter-<br/>nal bus and reports every detected problem as a fault code in the diagnostic buffer."),
    ("h2", "1.1 Safety"),
    ("p", "Always apply lockout/tagout before opening the cabinet. Only qualified electricians may work on live circuits."),
    ("break",),
    ("h1", "2 LED Status Indicators"),
    ("p", "The front panel has three LEDs. The RUN LED is green when the user program is executing. The ERROR LED flashes red when a fault code is active. The MAINT LED is yellow when maintenance is requested."),
    ("table", ["LED", "State", "Meaning"], [
        ["RUN", "Steady green", "Program running"],
        ["ERROR", "Flashing red", "Active fault, see diagnostic buffer"],
        ["MAINT", "Steady yellow", "Battery low or maintenance requested"],
    ]),
    ("break",),
    ("h1", "3 Fault Codes"),
    ("p", "The following table lists all fault codes of the FX-100."),
    ("table", ["Code", "Description", "Cause", "Remedy"], FAULT_TABLE),
    ("break",),
    ("h1", "4 Troubleshooting Procedures"),
    ("p", "When the ERROR LED flashes, follow these steps in order."),
    ("steps", [
        "Read the active fault code from the diagnostic buffer.",
        "Look up the code in chapter 3 and note the remedy.",
        "Apply lockout/tagout before touching wiring.",
        "Correct the cause and acknowledge the fault with the RESET button.",
    ]),
    ("p", "If the fault reappears within one minute, replace the affected module and contact ACME support."),
]


def make_fixture_pdf(path: str | Path) -> Path:
    return build_manual_pdf(path, FIXTURE_TITLE, ELEMENTS, HEADER, FOOTER)


if __name__ == "__main__":
    print(make_fixture_pdf(Path(__file__).with_name("fx100_manual.pdf")))
