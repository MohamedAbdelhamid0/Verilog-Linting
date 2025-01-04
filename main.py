import re
import argparse

class Lintify:
    def __init__(self, name): #WORKING JUST FINE
        self.name = name
        self.violations = []

    def parse_verilog(self, filepath): #WORKING JUST FINE
        """Parses the Verilog file and returns its content."""
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()

            # Remove comments and blank lines
            cleaned_lines = []
            in_block_comment = False  # To handle multi-line comments

            for line in lines:
                line = line.strip()

                if line == "":
                    continue  # Skip empty lines

                # Handle single-line comments
                if line.startswith("//"):
                    continue  # Skip line comments

                # Handle multi-line comments
                if "/*" in line:
                    in_block_comment = True
                    line = re.sub(r"/\*.*", "", line)  # Remove part before block comment starts
                if "*/" in line and in_block_comment:
                    in_block_comment = False
                    line = re.sub(r".*\*/", "", line)  # Remove part after block comment ends

                if in_block_comment:
                    continue  # Skip lines inside block comments

                # Add cleaned line to list
                cleaned_lines.append(line)

            return cleaned_lines

        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            return []

    def check_arithmetic_overflow(self, lines): #WORKING JUST FINE
        """
        Detects potential arithmetic overflow issues in Verilog code.
        """
        # Pattern to detect both blocking (=) and non-blocking (<=) assignments
        assign_pattern = re.compile(r'(\w+)\s*(<=|=)\s*(.*?);')  # Match both blocking (=) and non-blocking (<=) assignments
        reg_pattern = re.compile(r'\breg\s*(?:\[(\d+):(\d+)\]\s*)?(\w+)(?:\s*,\s*(\w+))*\s*;')  # Registers with or without bit width
        wire_pattern = re.compile(r'\bwire\s*\[(\d+):(\d+)\]\s+(\w+);')  # Wires
        constant_pattern = re.compile(r'\b\d+\b')  # Constants in expressions
        operator_pattern = re.compile(r'[+\-*/]')  # Arithmetic operators

        variable_bit_width = {}  # Dictionary to store variable names and their bit widths

        # Phase 1: Collect variable declarations and their bit widths
        original_lines = []  # List to store original lines before stripping
        non_empty_lines = []  # List to store non-empty lines
        for line in lines:
            original_lines.append(line)  # Store the original line
            if line.strip():  # Skip empty lines
                non_empty_lines.append(line.strip())

        # Process non-empty lines for variable declarations
        for i, stripped_line in enumerate(non_empty_lines):
            stripped_line = stripped_line.strip()

            # Check for register declarations with or without bit width
            reg_match = reg_pattern.search(stripped_line)
            if reg_match:
                if reg_match.group(1) and reg_match.group(2):  # Register with bit width
                    msb, lsb = map(int, reg_match.groups()[:2])  # Convert to integers
                    bit_width = msb - lsb + 1
                else:  # Register without bit width
                    bit_width = 1  # Default bit width
                    
                # Now, handle the comma-separated list of registers
                registers = [reg_match.group(i) for i in range(3, len(reg_match.groups()) + 1) if reg_match.group(i)]  # Collect all registers

                for reg in registers:
                    variable_bit_width[reg] = bit_width

            # Check for wire declarations
            wire_match = wire_pattern.search(stripped_line)
            if wire_match:
                msb, lsb = map(int, wire_match.groups()[:2])  # Convert to integers
                var_name = wire_match.group(3)  # Extract variable name
                bit_width = msb - lsb + 1
                variable_bit_width[var_name] = bit_width

        # Phase 2: Analyze assignments for potential overflow
        for i, stripped_line in enumerate(non_empty_lines):
            stripped_line = stripped_line.strip()
            assign_match = assign_pattern.search(stripped_line)

            if assign_match:
                var_name, assign_type, expression = assign_match.groups()  # Capture the assignment type as well
                var_name = var_name.strip()

                if var_name not in variable_bit_width:
                    continue  # Skip if bit width is unknown

                assigned_bit_width = variable_bit_width[var_name]

                # Extract variables and constants from the expression using regex
                variables = re.findall(r'\b\w+\b', expression)  # Find all words (variables)
                constants = [int(const) for const in constant_pattern.findall(expression)]  # Extract constants
                operators = operator_pattern.findall(expression)  # Extract operators


                # Check for constant overflow
                for const in constants:
                    if const >= (1 << assigned_bit_width):
                        # Append violation instead of using add_violation
                        line_number = original_lines.index(non_empty_lines[i]) + 1  # Get original line number
                        self.violations.append({
                            "name": "Arithmetic Overflow",
                            "line": line_number,
                            "impact": f"Constant {const} exceeds the maximum value for {assigned_bit_width}-bit width."
                        })

                # Check for overflow when variables are involved in operations
                if operators:
                    max_operand_bit_width = max(variable_bit_width.get(var, 0) for var in variables)

                    # Check if the result of the operation exceeds the destination width
                    if "+" in operators or "-" in operators:
                        # When adding or subtracting, the result can have an additional bit for carry
                        required_bit_width = max_operand_bit_width + 1
                        if required_bit_width > assigned_bit_width:
                            # Append violation instead of using add_violation
                            line_number = original_lines.index(non_empty_lines[i]) + 1  # Get original line number
                            self.violations.append({
                                "name": "Arithmetic Overflow",
                                "line": line_number,
                                "impact": f"Addition/Subtraction operation exceeds {assigned_bit_width}-bit width for '{var_name}'."
                            })

                    if "*" in operators:
                        # For multiplication, the result can have a wider bit width
                        required_bit_width = max_operand_bit_width * 2  # Simplified assumption
                        if required_bit_width > assigned_bit_width:
                            # Append violation instead of using add_violation
                            line_number = original_lines.index(non_empty_lines[i]) + 1  # Get original line number
                            self.violations.append({
                                "name": "Arithmetic Overflow",
                                "line": line_number,
                                "impact": f"Multiplication operation exceeds {assigned_bit_width}-bit width for '{var_name}'."
                            })

    def check_unreachable_blocks(self, lines): #WORKING JUST FINE
        """Detects unreachable blocks in Verilog, including overlapping cases and other original checks."""
        case_block_pattern = re.compile(r'\b(case|casex|casez)\b')
        case_end_pattern = re.compile(r'\bendcase\b')
        condition_pattern = re.compile(r'(\S+):')  # Matches conditions in case statements
        inside_case = False
        current_case_conditions = []
        expanded_conditions = []
        current_case_line = None

        signals_defined = set()
        signals_used = set()

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            # Track signal definitions
            signal_def_match = re.search(r'\b(?:reg|wire)\b\s*(?:\[\d+:\d+\])?\s+(\w+)', stripped_line)
            if signal_def_match:
                signals_defined.add(signal_def_match.group(1))

            # Track signal usage
            signal_use_match = re.findall(r'\b\w+\b', stripped_line)
            for signal in signal_use_match:
                if signal in signals_defined:
                    signals_used.add(signal)

            # Detect the start of a case block
            if case_block_pattern.search(stripped_line):
                inside_case = True
                current_case_conditions.clear()
                expanded_conditions.clear()
                current_case_line = i + 1

            elif inside_case:
                # Detect case conditions
                condition_match = condition_pattern.search(stripped_line)
                if condition_match:
                    condition = condition_match.group(1)
                    # Expand the condition to its binary representations (handles wildcards)
                    expanded = self.expand_condition(condition)

                    # Check for overlaps with already expanded conditions
                    for existing_condition in expanded_conditions:
                        if self.check_overlap(expanded, existing_condition):
                            self.violations.append({
                                "name": "Unreachable Block",
                                "line": i + 1,
                                "impact": f"Condition '{condition}' is unreachable due to overlap with a previous condition."
                            })
                            break

                    # Add the expanded condition to the list
                    expanded_conditions.append(expanded)
                    current_case_conditions.append(condition)

                # Detect the end of a case block
                if case_end_pattern.search(stripped_line):
                    inside_case = False

            # Detect if/else blocks with trivially true/false conditions
            if re.search(r'\bif\b', stripped_line):
                condition = re.search(r'\((.*?)\)', stripped_line)
                if condition:
                    condition = condition.group(1)
                    if condition in ["1'b1", "1'b0", "0", "1"]:
                        truth_value = "true" if condition in ["1", "1'b1"] else "false"
                        self.violations.append({
                            "name": "Unreachable If Block",
                            "line": i + 1,
                            "impact": f"If condition '{condition}' is trivially {truth_value}."
                        })

        # Check for unused signals
        unused_signals = signals_defined - signals_used
        for signal in unused_signals:
            self.violations.append({
                "name": "Unused Signal",
                "line": None,
                "impact": f"Signal '{signal}' is defined but never used."
            })

    def check_uninitialized_registers(self, lines): #WORKING JUST FINE
        """
        Detects uninitialized registers, excluding output registers.
        Tracks both blocking ('=') and non-blocking ('<=') assignments.
        Flags registers used in comparisons but not initialized, ensuring the check
        is performed only once for registers found in an always block.
        """
        pattern = re.compile(r'\breg\s*\[(\d+):(\d+)\]\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;')  # Handle reg with bit-widths
        initialized_registers = set()  # Track initialized registers
        output_signals = set()  # Set to store output signals
        blocking_assignments = set()  # Registers initialized with blocking assignments
        non_blocking_assignments = set()  # Registers initialized with non-blocking assignments
        used_registers = set()  # Track registers used in comparisons
        checked_registers = set()  # Track registers for which comparison checks have been performed

        # Detect all output signals in the lines, including those with bit-widths
        for line in lines:
            if re.search(r'\boutput\s+(reg|wire)?\s*\[(\d+):(\d+)\]\s+(\w+)', line):
                match = re.search(r'\boutput\s+(reg|wire)?\s*\[(\d+):(\d+)\]\s+(\w+)', line)
                if match:
                    signal = match.group(4)  # The signal name (fourth group in the match)
                    output_signals.add(signal)

        # Track registers initialized in initial or always blocks
        initial_block_pattern = re.compile(r'\b(initial|always)\b')
        blocking_assign_pattern = re.compile(r'(\w+)\s*=\s*(.+);')  # Blocking assignment
        non_blocking_assign_pattern = re.compile(r'(\w+)\s*<=\s*(.+);')  # Non-blocking assignment
        comparison_pattern = re.compile(r'(\w+)\s*(==|!=)\s*')  # Comparison pattern for register usage

        for i, line in enumerate(lines):
            # Search for initial or always block assignments
            if initial_block_pattern.search(line):
                for j in range(i + 1, len(lines)):
                    if "end" in lines[j]:
                        break
                    # Check for blocking assignments
                    match = blocking_assign_pattern.search(lines[j])
                    if match:
                        signal = match.group(1)
                        blocking_assignments.add(signal)
                        initialized_registers.add(signal)
                    # Check for non-blocking assignments
                    match = non_blocking_assign_pattern.search(lines[j])
                    if match:
                        signal = match.group(1)
                        non_blocking_assignments.add(signal)
                        initialized_registers.add(signal)
            
            # Track registers used in comparisons (e.g., if (reg == 2'b01))
            match = comparison_pattern.search(line)
            if match:
                reg = match.group(1)  # Register in comparison
                used_registers.add(reg)

        # Perform the uninitialized register check only once for registers in comparisons found in always blocks
        for reg in used_registers:
            if reg not in initialized_registers and reg not in checked_registers:
                self.violations.append({
                    "name": "Uninitialized Register",
                    "line": i + 1,
                    "impact": f"Register '{reg}' used in comparison but not initialized."
                })
                checked_registers.add(reg)  # Mark this register as checked for comparison

        # Check for uninitialized registers that are not assigned in initial or always blocks
        for i, line in enumerate(lines):
            match = pattern.search(line)
            if match:
                signal = match.group(3)  # Extract the signal name
                # Skip output signals
                if signal in output_signals:
                    continue
                # Flag uninitialized registers that are not initialized in initial or always block
                if signal not in initialized_registers:
                    self.violations.append({
                        "name": "Uninitialized Register",
                        "line": i + 1,
                        "impact": f"Register '{signal}' is uninitialized."
                    })

    def check_multi_driven_registers(self, lines): #WORKING JUST FINE
        """Detects multi-driven output registers and overlapping assignments."""

        output_signals = set()  # Track output registers
        always_assignments = {}  # Track assignments in always blocks
        case_assignments = {}  # Track assignments in case blocks
        inside_always = False
        inside_case = False
        current_always = None  # Track the current always block
        nested_blocks = []  # Track nested blocks for proper 'end' handling
        signal_drivers = {}  # Track drivers for each signal


        for i, line in enumerate(lines):
            line_number = i + 1
            stripped_line = line.strip()

            # Identify output registers
            output_match = re.search(r'\boutput\s+reg\s*(?:\[\d+:\d+\])?\s+(\w+)', line)
            if output_match:
                signal = output_match.group(1)
                output_signals.add(signal)

            # Detect always blocks
            if "always" in stripped_line:
                inside_always = True
                current_always = line_number
                nested_blocks.append("always")

            # Detect end of blocks
            elif "end" in stripped_line:
                if nested_blocks:
                    ended_block = nested_blocks.pop()
                    if ended_block == "always":
                        inside_always = False
                        current_always = None
                    elif ended_block == "case":
                        inside_case = False

            # Detect case blocks
            if "casez" in stripped_line or "case" in stripped_line:
                inside_case = True
                nested_blocks.append("case")

            # Track assignments within always and case blocks
            assign_match = re.search(r'(\w+)\s*=\s*(\w+)', line)
            if assign_match:
                target_signal = assign_match.group(1)
                driver_signal = assign_match.group(2)

                # Track drivers for the target signal
                if target_signal not in signal_drivers:
                    signal_drivers[target_signal] = []
                signal_drivers[target_signal].append((driver_signal, line_number))

                # Check for multi-driven registers
                if inside_always and target_signal in output_signals:
                    if target_signal not in always_assignments:
                        always_assignments[target_signal] = {}
                    if current_always not in always_assignments[target_signal]:
                        always_assignments[target_signal][current_always] = []
                    always_assignments[target_signal][current_always].append(line_number)

                if inside_case and target_signal in output_signals:
                    if target_signal not in case_assignments:
                        case_assignments[target_signal] = {}
                    case_value_match = re.search(r'\d+\'[bBzZ]\s*\w+', line)
                    if case_value_match:
                        case_value = case_value_match.group(0).strip()
                        if case_value not in case_assignments[target_signal]:
                            case_assignments[target_signal][case_value] = []
                        case_assignments[target_signal][case_value].append(line_number)


        # Check for multi-driven violations
        for signal in output_signals:
            # Check for multiple always block assignments
            if signal in always_assignments:
                if len(always_assignments[signal]) > 1:
                    self.violations.append({
                        "name": "Multi-Driven Register",
                        "line": list(always_assignments[signal].keys()),
                        "impact": f"Signal '{signal}' assigned in multiple always blocks."
                    })

            # Check for overlapping case conditions
            if signal in case_assignments:
                conditions = list(case_assignments[signal].keys())
                for i in range(len(conditions)):
                    for j in range(i + 1, len(conditions)):
                        if not self.check_overlap(conditions[i], conditions[j]):
                            self.violations.append({
                                "name": "Multi-Driven Register (Case Overlap)",
                                "line": case_assignments[signal][conditions[i]] + case_assignments[signal][conditions[j]],
                                "impact": f"Signal '{signal}' assigned under overlapping case conditions '{conditions[i]}' and '{conditions[j]}'."
                            })

        # Check for two registers assigning the same signal
        for target_signal, drivers in signal_drivers.items():
            if len(drivers) > 1:
                self.violations.append({
                    "name": "Conflicting Drivers",
                    "line": [driver[1] for driver in drivers],
                    "impact": f"Signal '{target_signal}' is driven by multiple sources: {[driver[0] for driver in drivers]}"
                })

    def is_condition_constant(self, condition): #WORKING JUST FINE
        """Check if a condition is a constant 1'b0 or 1'b1."""
        return condition.strip() in ["1'b0", "1'b1"]

    def check_non_full_parallel_cases(self, lines): #WORKING JUST FINE
        """Detects non-full or non-parallel case statements."""
        case_block_pattern = re.compile(r'\b(case|casex|casez)\b')
        case_end_pattern = re.compile(r'\bendcase\b')
        inside_case = False
        has_default = False
        current_case_conditions = []
        current_case_signal = None
        current_case_line = None
        bit_width = None  # Start with undefined bit width

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            # Detect start of case statement
            if case_block_pattern.search(stripped_line):
                inside_case = True
                has_default = False
                current_case_conditions.clear()
                current_case_line = i + 1

                # Capture the signal being checked in the case statement
                signal_match = re.search(r'case[zx]?\s*\((\w+|\S+)\)', stripped_line)
                if signal_match:
                    current_case_signal = signal_match.group(1)

            elif inside_case:
                # Detect `default` case
                if re.match(r'default\s*:', stripped_line):
                    has_default = True

                # Capture case conditions (e.g., 2'b00:)
                case_condition_match = re.search(r'(\d+)\'b[01?z]+', stripped_line)
                if case_condition_match:
                    condition = case_condition_match.group(0)
                    current_case_conditions.append(condition)

                    # Dynamically determine the bit width from the condition
                    if bit_width is None:  # Set bit width only once
                        bit_width = int(case_condition_match.group(1))

                # Detect end of case block
                if case_end_pattern.search(stripped_line):
                    # Calculate total combinations if bit width is known
                    if bit_width is not None:
                        total_combinations = 2 ** bit_width
                    else:
                        total_combinations = None  # Undefined due to missing bit-width info

                    # Check for non-full case if no default
                    if not has_default and total_combinations is not None:
                        self.check_missing_combinations(current_case_signal, current_case_conditions, current_case_line, bit_width)

                    # Check for non-parallel conditions (overlapping cases)
                    self.check_for_non_parallel(current_case_conditions, current_case_line)

                    inside_case = False

    def check_missing_combinations(self, signal, conditions, line_number, bit_width): #WORKING JUST FINE
        """Check if all combinations of the signal are covered in the case block."""
        if bit_width is None:
            return  # Can't determine possible combinations without bit width

        total_combinations = 2 ** bit_width  # Total combinations based on bit width

        # Expand conditions for case statement
        expanded_conditions = set()
        for condition in conditions:
            expanded_conditions.update(self.expand_condition(condition))

        # Check if all combinations are covered
        if len(expanded_conditions) < total_combinations:
            missing_combinations = total_combinations - len(expanded_conditions)
            self.violations.append({
                "name": "Non-Full Case",
                "line": line_number,
                "impact": f"Case statement does not cover all {total_combinations} combinations. "
                        f"Missing {missing_combinations} combinations."
            })

    def check_for_non_parallel(self, conditions, line_number): #WORKING JUST FINE
        """Check if any conditions overlap in a case block."""
        expanded_conditions = [self.expand_condition(condition) for condition in conditions]

        for i in range(len(expanded_conditions)):
            for j in range(i + 1, len(expanded_conditions)):
                if self.check_overlap(expanded_conditions[i], expanded_conditions[j]):
                    self.violations.append({
                        "name": "Non-Parallel Case",
                        "line": line_number,
                        "impact": f"Overlapping case conditions: {conditions[i]} and {conditions[j]}."
                    })

    def expand_condition(self, condition): #WORKING JUST FINE
        """Expand a condition with '?' or 'z' into all possible binary values."""
        if '?' in condition or 'z' in condition:
            condition = condition.replace('z', '?')  # Treat 'z' as '?'
            options = []
            num_wildcards = condition.count('?')
            for i in range(2 ** num_wildcards):
                binary_str = format(i, f'0{num_wildcards}b')
                expanded = condition
                for bit in binary_str:
                    expanded = expanded.replace('?', bit, 1)
                options.append(expanded)
            return options
        return [condition]  # If no wildcards, return condition as is

    def check_overlap(self, expanded_condition1, expanded_condition2):
        """
        Checks if two expanded conditions overlap. Overlap occurs when there is
        at least one common value between the sets of expanded conditions.
        """

        set1 = set(expanded_condition1)
        set2 = set(expanded_condition2)
         
        return not set1.isdisjoint(set2)  # Check for any intersection

    def check_inferred_latches(self, lines): #WORKING JUST FINE
            """Detects inferred latches due to missing else statements in always blocks and missing case conditions."""
            
            inside_always = False
            inside_if = False
            signal_assigned_in_if = set()  # Track signals assigned in the if block
            signal_assigned_in_else = set()  # Track signals assigned in the else block
            signal_assigned = set()  # Track signals in any block
            output_signals = set()  # Track output signals (only reg type output)
            assign_pattern = re.compile(r'(\w+)\s*=\s*')  # Pattern to detect assignments
            if_pattern = re.compile(r'\bif\b\s*\((.*?)\)')  # Capture the condition in the if statement
            else_pattern = re.compile(r'\belse\b')  # Match if there's an else statement
            case_pattern = re.compile(r'case[zx]?\s*\(')  # Match case, casex, casez
            reg_pattern = re.compile(r'\s*output\s+reg\s+(\w+)')  # Match output reg type declaration

            # First pass: Identify output registers in the module
            for line in lines:
                match = reg_pattern.search(line)
                if match:
                    output_signals.add(match.group(1))  # Add the signal to the output signals set

            # Track nesting level for 'always' blocks
            always_nesting_level = 0

            for i, line in enumerate(lines):
                if "always" in line:
                    inside_always = True
                    always_nesting_level += 1
                    inside_if = False  # Reset the if flag
                    signal_assigned_in_if.clear()  # Reset signal tracking for 'if' block
                    signal_assigned_in_else.clear()  # Reset signal tracking for 'else' block
                    signal_assigned.clear()  # General reset
                elif inside_always:
                    if "begin" in line:
                        # If we encounter a 'begin' inside an 'if', we track it for nested block
                        pass
                    elif "end" in line:
                        # End of always block
                        always_nesting_level -= 1
                        if always_nesting_level == 0:
                            # Only check for inferred latch when we exit the innermost 'always' block
                            if inside_if and not signal_assigned_in_else:
                                # If there's an if block but no else, flag as inferred latch
                                self.violations.append({
                                    "name": "Inferred Latch",
                                    "line": i + 1,
                                    "impact": "Potential latch due to missing else statement in always block."
                                })
                            inside_always = False  # End the always block

                    elif if_pattern.search(line):
                        # Found 'if' statement, reset the flags
                        inside_if = True
                        signal_assigned_in_if.clear()  # Reset signal assignments for this block
                    elif else_pattern.search(line):
                        # Found 'else' statement, reset the flag and clear assignments
                        inside_if = False  # No need to track 'if' anymore after 'else'
                        signal_assigned_in_else.clear()  # Reset signal assignments for 'else'

                    # Track signal assignments
                    assign_match = assign_pattern.search(line)
                    if assign_match:
                        signal = assign_match.group(1)
                        signal_assigned.add(signal)  # Mark signal as assigned in this block
                        if signal not in output_signals:
                            continue  # Skip signals that aren't output registers
                        if signal not in output_signals:
                            output_signals.add(signal)  # Track output signals
                        if inside_if:
                            signal_assigned_in_if.add(signal)  # Assign signal to 'if'
                        elif inside_always and not inside_if:  # This must be in 'else'
                            signal_assigned_in_else.add(signal)  # Assign signal to 'else'

                # Now checking case, casex, casez for incomplete conditions
                elif case_pattern.search(line):
                    # We found a case statement
                    case_lines = []
                    for j in range(i + 1, len(lines)):
                        if "endcase" in lines[j]:
                            break
                        case_lines.append(lines[j])

                    # Check if all conditions are covered
                    all_values_covered = False
                    case_values = [line.strip().split(":")[0] for line in case_lines]  # Left side of each case statement
                    case_values_set = set(case_values)

                    # Check for missing combinations
                    if len(case_values_set) < len(case_values):  # Duplicate conditions indicate missing cases
                        all_values_covered = False
                        self.violations.append({
                            "name": "Incomplete Case Statement",
                            "line": i + 1,
                            "impact": f"Case statement does not cover all possible conditions."
                        })

            # Now check for inferred latch when output signals are not assigned in all conditions
            for signal in output_signals:
                # If a signal is assigned in if but not else, flag a latch violation
                if signal in signal_assigned_in_if and signal not in signal_assigned_in_else:
                    self.violations.append({
                        "name": "Inferred Latch",
                        "line": len(lines),
                        "impact": f"Output signal '{signal}' is not assigned in all conditions, leading to potential latch."
                    })
                # If a signal is assigned in else but not in if, flag a latch violation
                elif signal in signal_assigned_in_else and signal not in signal_assigned_in_if:
                    self.violations.append({
                        "name": "Inferred Latch",
                        "line": len(lines),
                        "impact": f"Output signal '{signal}' is not assigned in all conditions, leading to potential latch."
                    })

    def check_unreachable_fsm_states(self, lines):
            """Detect all potential reasons for unreachable FSM states, including states reachable via default transitions."""

            # Patterns for state parsing
            state_pattern = re.compile(r'\bparameter\b.?(\w+)\s=\s*\d+\'b\d+')  # No change needed
            transition_to_pattern = re.compile(r'(\w+)\s*(?:<=|=)\s*(\w+);')  # Detect transitions to states
            case_condition_pattern = re.compile(r'^\s*(\w+)\s*:')  # Detect state conditions in case
            case_start_pattern = re.compile(r'case\s*\(\s*(\w+)\s*\)')  # Start of a case block
            case_end_pattern = re.compile(r'endcase')  # End of a case block
            initial_state_pattern = re.compile(r'(\w+)\s*(?:<=|=)\s*(\w+);')  # Detect initial state assignment
            if_else_pattern = re.compile(r'\s*if\s*\(.*\)\s*else')  # Detect if-else structure

            # Initialize data structures
            declared_states = {}  # Map of state name to its line number
            transitions = {}  # Map of each state to its outgoing transitions
            default_transitions = []  # Store default transitions
            reachable_states = set()  # Track reachable states
            initial_state = None
            inside_case = False
            current_state = None
            state_if_else = {}  # To track if a state has an if-else structure


            # Step 1: Parse declared states
            for i, line in enumerate(lines):
                state_match = state_pattern.search(line)
                if state_match:
                    declarations = line.split(',')
                    for declaration in declarations:
                        single_state_match = re.search(r'(\w+)\s*=\s*\d+\'b\d+', declaration)
                        if single_state_match:
                            state_name = single_state_match.group(1)
                            declared_states[state_name] = i + 1  # Store state and its line number
                            transitions[state_name] = []  # Initialize outgoing transitions
                            state_if_else[state_name] = False  # Default, no if-else structure

            # Step 2: Parse transitions and check for if-else structures
            for i, line in enumerate(lines):
                stripped_line = line.strip()

                # Detect initial state
                if initial_state is None:
                    initial_match = initial_state_pattern.search(stripped_line)
                    if initial_match:
                        state_variable, initial_state = initial_match.groups()  # Capture the state variable and initial state

                # Detect the start of a case block and the associated state variable
                case_start_match = case_start_pattern.search(stripped_line)
                if case_start_match:
                    state_variable = case_start_match.group(1) if case_start_match.groups() else None
                    if state_variable:
                        inside_case = True
                        current_state = None
                    else:
                        inside_case = False
                        continue

                # Detect state conditions and outgoing transitions
                if inside_case:
                    condition_match = case_condition_pattern.search(stripped_line)
                    if condition_match:
                        current_state = condition_match.group(1)
                        # Check for "default" and handle appropriately
                        if current_state == "default":
                            current_state = None  # Reset current_state for default transitions
                            continue  # Skip adding transitions for "default"

                    # Parse inline transitions
                    transition_match = transition_to_pattern.search(stripped_line)
                    if transition_match:
                        groups = transition_match.groups()
                        if len(groups) == 2:  # Ensure two values are captured
                            transition_variable, target_state = groups
                            if transition_variable == state_variable:  # Ensure the detected variable matches the state variable
                                if current_state and target_state in declared_states:
                                    transitions[current_state].append(target_state)  # Outgoing transition
                                elif not current_state:  # Handle default case
                                    default_transitions.append(target_state)

                    # End of case block
                    if case_end_pattern.search(stripped_line):
                        inside_case = False


            # Add default transitions to states with no explicit outgoing transitions
            for state in declared_states:
                if not transitions[state]:  # If the state has no explicit outgoing transitions
                    transitions[state] = default_transitions.copy()

            # Step 3: Include states in default transitions as reachable
            reachable_states.update(default_transitions)

            # Step 4: Determine reachable states using DFS

            def dfs(state):
                """Depth-First Search to mark reachable states."""
                if state in reachable_states:
                    return  # Skip already visited states
                reachable_states.add(state)  # Mark the state as reachable
                for next_state in transitions.get(state, []):
                    dfs(next_state)

            if initial_state:
                dfs(initial_state)

            # *Include states with incoming transitions as reachable*
            for state, targets in transitions.items():
                for target_state in targets:
                    if target_state in declared_states and target_state not in reachable_states:
                        reachable_states.add(target_state)


            # Step 5: Detect unreachable states
            unreachable_states = set(declared_states.keys()) - reachable_states
            for state in unreachable_states:
                self.violations.append({
                    "name": "Unreachable FSM State",
                    "line": declared_states[state],
                    "impact": f"State '{state}' is declared but unreachable from the initial state."
                })

            # Step 6: Detect dead-end states (no outgoing transitions)
            for state, targets in transitions.items():
                if state in reachable_states:
                    if not targets:  # No outgoing transitions
                        self.violations.append({
                            "name": "Dead-End FSM State",
                            "line": declared_states[state],
                            "impact": f"State '{state}' has no outgoing transitions and can trap the FSM."
                        })
                    elif len(targets) == 1 and targets[0] == state:  # Only self-loop
                        self.violations.append({
                            "name": "Dead-End FSM State (Self-Loop)",
                            "line": declared_states[state],
                            "impact": f"State '{state}' transitions only to itself and can trap the FSM."
                        })

            # Step 7: Detect invalid transitions
            for state, targets in transitions.items():
                for target_state in targets:
                    if target_state not in declared_states:
                        print(f"DEBUG: Invalid transition detected: {state} -> {target_state}")
                        self.violations.append({
                            "name": "Invalid FSM Transition",
                            "line": None,
                            "impact": f"State '{state}' transitions to undeclared state '{target_state}'."
                        })

    def check_violations(self, filepath): #WORKING JUST FINE
        """Runs all checks on the given Verilog file."""
        lines = self.parse_verilog(filepath)
        if not lines:
            return

        self.check_arithmetic_overflow(lines)
        self.check_unreachable_blocks(lines)
        self.check_uninitialized_registers(lines)
        self.check_multi_driven_registers(lines)
        self.check_non_full_parallel_cases(lines)
        self.check_inferred_latches(lines)
        self.check_unreachable_fsm_states(lines)

    def generate_report(self, report_path): #WORKING JUST FINE
        """Generates a report of all violations."""
        with open(report_path, 'w') as report:
            for violation in self.violations:
                report.write(f"Violation: {violation['name']}\n")
                report.write(f"Line(s): {violation['line']}\n")
                report.write(f"Impacted: {violation['impact']}\n")
                report.write("-" * 40 + "\n")

if __name__ == "__main__":  # WORKING JUST FINE
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Verilog Linter")
    parser.add_argument("verilog_file", help="Path to the Verilog file to lint")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Example Usage
    linter = Lintify("Lintify")
    
    verilog_file = args.verilog_file  # Get the Verilog file from the command line argument
    report_file = f"{verilog_file.split('.')[0]}_lint_report.txt"  # Create a report filename based on the Verilog file name

    linter.check_violations(verilog_file)
    linter.generate_report(report_file)
    
    print(f"Linting completed. Report generated at {report_file}")

