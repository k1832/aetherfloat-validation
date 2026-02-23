import os, subprocess, re, urllib.request, shutil, sys, textwrap

# --- CONFIGURATION ---
YOSYS_EXEC = shutil.which("yosys")
if not YOSYS_EXEC:
    print("CRITICAL ERROR: 'yosys' not found.")
    sys.exit(1)

STA_EXEC = shutil.which("sta") or shutil.which("opensta")
if not STA_EXEC:
    print("CRITICAL ERROR: 'sta' (OpenSTA) not found.")
    sys.exit(1)

# Ensure Lib exists
sky130_url = "https://raw.githubusercontent.com/efabless/skywater-pdk-libs-sky130_fd_sc_hd/master/timing/sky130_fd_sc_hd__tt_025C_1v80.lib"
if not os.path.exists("sky130.lib") or os.path.getsize("sky130.lib") == 0:
    print("Downloading sky130.lib...")
    urllib.request.urlretrieve(sky130_url, "sky130.lib")

# Generate Verilog
verilog_code = textwrap.dedent("""
    module mac_fp8_base2 (
        input wire clk,
        input wire [3:0] mant_a,
        input wire [3:0] mant_b,
        input wire [7:0] mant_c,
        input wire [3:0] exp_diff,
        output reg [8:0] mac_out
    );
        wire [7:0] mult_out = mant_a * mant_b;
        wire [7:0] s0 = exp_diff[0] ? {1'b0, mant_c[7:1]} : mant_c;
        wire [7:0] s1 = exp_diff[1] ? {2'b0, s0[7:2]} : s0;
        wire [7:0] s2 = exp_diff[2] ? {4'b0, s1[7:4]} : s1;
        wire [7:0] s3 = exp_diff[3] ? 8'b0 : s2;
        always @(posedge clk) begin
            mac_out <= mult_out + s3;
        end
    endmodule

    module mac_af8_base4 (
        input wire clk,
        input wire [2:0] mant_a,
        input wire [2:0] mant_b,
        input wire [7:0] mant_c,
        input wire [2:0] exp_diff,
        output reg [8:0] mac_out
    );
        wire [5:0] mult_out = mant_a * mant_b;
        reg [7:0] shifted_c;
        always @(*) begin
            case (exp_diff)
                3'b000: shifted_c = mant_c;
                3'b001: shifted_c = {2'b0, mant_c[7:2]};
                3'b010: shifted_c = {4'b0, mant_c[7:4]};
                3'b011: shifted_c = {6'b0, mant_c[7:6]};
                3'b100: shifted_c = 8'b0;
                default: shifted_c = 8'b0;
            endcase
        end
        always @(posedge clk) begin
            mac_out <= mult_out + shifted_c;
        end
    endmodule
""")
with open("mac_eval.v", "w") as f: f.write(verilog_code)

def run_pipeline(module_name):
    netlist_file = f"synth_{module_name}.v"

    # --- YOSYS ---
    print(f"[{module_name}] Synthesis...", end="", flush=True)
    yosys_script = textwrap.dedent(f"""
        read_verilog mac_eval.v
        synth -top {module_name} -flatten
        dfflibmap -liberty sky130.lib
        abc -liberty sky130.lib
        opt_clean
        stat -liberty sky130.lib
        write_verilog -noattr {netlist_file}
    """)
    with open(f"run_yosys_{module_name}.ys", "w") as f: f.write(yosys_script)

    try:
        res = subprocess.run(
            [YOSYS_EXEC, f"run_yosys_{module_name}.ys"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120
        )
    except subprocess.TimeoutExpired:
        print(" FAILED (Timeout)")
        return 0.0, 0.0, 0.0

    if res.returncode != 0:
        print(f" FAILED (Error code {res.returncode})")
        return 0.0, 0.0, 0.0

    area_match = re.search(r'Chip area for module.*?:\s+([0-9.]+)', res.stdout)
    area = float(area_match.group(1)) if area_match else 0.0
    print(f" Done (Area: {area}).")

    # --- OPENSTA ---
    print(f"[{module_name}] STA...", end="", flush=True)
    sta_delay = 0.0
    sta_power_uW = 0.0

    tcl_filename = f"run_sta_{module_name}.tcl"
    sta_script = textwrap.dedent(f"""
        read_liberty sky130.lib
        read_verilog {netlist_file}
        link_design {module_name}

        create_clock -name clk -period 10.0 [get_ports clk]

        set_input_delay 0.0 -clock clk [all_inputs]
        set_output_delay 0.0 -clock clk [all_outputs]

        report_checks -path_delay max -digits 4

        set_power_activity -global -activity 0.2 -duty 0.2

        report_power
        exit
    """)
    with open(tcl_filename, "w") as f: f.write(sta_script)

    try:
        sta_res = subprocess.run(
            [STA_EXEC, tcl_filename],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120
        )
    except subprocess.TimeoutExpired:
        print(" FAILED (Timeout)")
        return area, 0.0, 0.0

    # --- PARSING ---
    # Parse OpenSTA report line-by-line.
    lines = sta_res.stdout.splitlines()
    power_values = []

    for line in lines:
        parts = line.split()

        # 1. Parse Delay: Look for line "3.0886 data arrival time"
        if "data" in parts and "arrival" in parts and "time" in parts:
            try:
                # The number is usually the FIRST element
                val = float(parts[0])
                if val > 0: sta_delay = val * 1000.0
            except ValueError: pass

        # 2. Parse Power: Look for rows starting with "Sequential", "Combinational", etc.
        # Format: "Sequential 4.75e-05 ..." -> we want column index 4 (Total)
        if len(parts) >= 5 and parts[0] in ["Sequential", "Combinational", "Clock", "Macro", "Pad"]:
            try:
                p_val = float(parts[4])
                power_values.append(p_val)
            except ValueError: pass

    if power_values:
        sta_power_uW = sum(power_values) * 1e6

    if sta_delay == 0.0 or sta_power_uW == 0.0:
        print(" FAILED to parse.")
    else:
        print(" Done.")

    return area, sta_delay, sta_power_uW

print(f"{'Module':<24} | {'Area (um^2)':<15} | {'Delay (ps)':<15} | {'Power (uW)':<15}")
print("-" * 75)
base2_a, base2_d, base2_p = run_pipeline("mac_fp8_base2")
base4_a, base4_d, base4_p = run_pipeline("mac_af8_base4")

print("-" * 75)
print(f"{'FP8 Base-2 MAC Baseline':<24} | {base2_a:<15.2f} | {base2_d:<15.2f} | {base2_p:<15.2f}")
print(f"{'AF8 Base-4 MAC (Ours)':<24} | {base4_a:<15.2f} | {base4_d:<15.2f} | {base4_p:<15.2f}")
