"""
Hybrid AI MCP Server
=====================

A Model Context Protocol server that implements transparent "post-NN calculators"
using McCulloch-Pitts neurons for explainable AI and safety-critical decision making.

This server enables hybrid AI architectures where:
1. Complex neural networks handle perception and pattern recognition
2. Simple, transparent MCP neurons make final, auditable decisions
3. Business rules and safety constraints are explicitly enforced

Core Concepts:
- MCP Neuron: Binary threshold neuron (fires if weighted sum >= threshold)
- Logic Gates: AND, OR, NOT, NAND, NOR, XOR built from MCP neurons
- Decision Networks: Composable networks for transparent decision-making
- Explainability: Every decision is traceable to explicit rules
"""

import atexit
import json
import logging
import signal
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from fastmcp import Context, FastMCP

# Configure logging to stderr for MCP servers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    "Hybrid-AI",
    dependencies=["numpy>=1.24.0", "pydantic>=2.0.0"],
)

# ==================== Core MCP Neuron Implementation ====================


class MCPNeuron:
    """
    McCulloch-Pitts neuron: A binary threshold neuron.

    The neuron computes: output = 1 if sum(x_i * w_i) >= threshold else 0

    Args:
        weights: Weight for each input (including bias as w[0])
        threshold: Activation threshold (default: 0)

    Note: weights[0] is the bias (equivalent to -threshold in standard notation)
    """

    def __init__(
        self, weights: Sequence[float], threshold: float = 0.0
    ):
        self.weights = np.array(weights, dtype=float)
        self.threshold = threshold

    def activate(self, inputs: Sequence[float]) -> int:
        """
        Activate the neuron with given inputs.

        Args:
            inputs: Input values (x_1, x_2, ..., x_n)

        Returns:
            1 if neuron fires, 0 otherwise
        """
        # Insert x_0 = 1 for bias
        x = np.insert(inputs, 0, 1.0)

        # Compute weighted sum
        signal = np.dot(x, self.weights)

        # Apply step function
        return 1 if signal >= self.threshold else 0

    def __call__(self, inputs: Sequence[float]) -> int:
        """Allow neuron to be called like a function."""
        return self.activate(inputs)


# ==================== Logic Gates ====================


class LogicGates:
    """Pre-configured MCP neurons implementing standard logic gates."""

    @staticmethod
    def AND(p: float, q: float) -> int:
        """Logical AND: fires only when both inputs are 1."""
        neuron = MCPNeuron([-1.0, 0.6, 0.6])
        return neuron([p, q])

    @staticmethod
    def OR(p: float, q: float) -> int:
        """Logical OR: fires when at least one input is 1."""
        neuron = MCPNeuron([-0.5, 1.0, 1.0])
        return neuron([p, q])

    @staticmethod
    def NOT(p: float) -> int:
        """Logical NOT: inverts the input."""
        neuron = MCPNeuron([0.5, -1.0])
        return neuron([p])

    @staticmethod
    def NAND(p: float, q: float) -> int:
        """Logical NAND: NOT(AND(p, q))."""
        neuron = MCPNeuron([1.0, -0.6, -0.6])
        return neuron([p, q])

    @staticmethod
    def NOR(p: float, q: float) -> int:
        """Logical NOR: NOT(OR(p, q))."""
        neuron = MCPNeuron([0.5, -1.0, -1.0])
        return neuron([p, q])

    @staticmethod
    def XOR(p: float, q: float) -> int:
        """Logical XOR: exclusive OR using composition of gates."""
        # XOR = AND(NAND(p,q), OR(p,q))
        return LogicGates.AND(LogicGates.NAND(p, q), LogicGates.OR(p, q))


# ==================== Decision Network ====================


class DecisionNetwork:
    """
    A network of MCP neurons for transparent decision-making.

    This represents the "post-NN calculator" layer that makes final,
    explainable decisions based on neural network outputs.
    """

    def __init__(self):
        self.neurons: dict[str, MCPNeuron] = {}
        self.connections: dict[str, list[str]] = {}
        self.decision_log: list[dict[str, Any]] = []

    def add_neuron(self, name: str, weights: Sequence[float], threshold: float = 0.0):
        """Add a neuron to the network."""
        self.neurons[name] = MCPNeuron(weights, threshold)
        self.connections[name] = []
        logger.info(f"Added neuron '{name}' with weights {weights}")

    def add_connection(self, from_neuron: str, to_neuron: str):
        """Add a connection between neurons."""
        if from_neuron not in self.connections:
            self.connections[from_neuron] = []
        self.connections[from_neuron].append(to_neuron)

    def evaluate(
        self, inputs: Mapping[str, Sequence[float]], neuron_name: str
    ) -> tuple[int, dict[str, Any]]:
        """
        Evaluate a neuron with given inputs and return the result with explanation.

        Args:
            inputs: Dictionary mapping input names to their values
            neuron_name: Name of the neuron to evaluate

        Returns:
            (output, explanation) tuple
        """
        if neuron_name not in self.neurons:
            raise ValueError(f"Neuron '{neuron_name}' not found")

        neuron = self.neurons[neuron_name]
        input_values = inputs.get(neuron_name, [])
        output = neuron.activate(input_values)

        # Create explainable log
        explanation = {
            "neuron": neuron_name,
            "inputs": list(input_values),
            "weights": neuron.weights.tolist(),
            "threshold": neuron.threshold,
            "signal": float(np.dot(np.insert(input_values, 0, 1.0), neuron.weights)),
            "output": output,
            "fired": bool(output),
        }

        self.decision_log.append(explanation)
        return output, explanation


# Global decision network
decision_network = DecisionNetwork()


# ==================== MCP Tools ====================


@mcp.tool()
def create_mcp_neuron(
    weights: list[float],
    threshold: float = 0.0,
    name: str | None = None,
) -> dict[str, Any]:
    """
    Create a McCulloch-Pitts neuron for transparent decision-making.

    Args:
        weights: Weights for the neuron [bias, w1, w2, ..., wn]
        threshold: Activation threshold (default: 0.0)
        name: Optional name for the neuron (for use in networks)

    Returns:
        Dictionary with neuron configuration and usage instructions

    Example:
        # Create an AND gate neuron
        create_mcp_neuron([-1.0, 0.6, 0.6], name="safety_check")
    """
    try:
        if name:
            decision_network.add_neuron(name, weights, threshold)
            logger.info(f"Created named neuron '{name}'")

        return {
            "status": "success",
            "weights": weights,
            "threshold": threshold,
            "name": name,
            "description": (
                f"Neuron fires when: bias + "
                f"sum(input[i] * weight[i]) >= {threshold}"
            ),
            "usage": (
                "Use evaluate_neuron() to test this neuron "
                "or use it in a decision network"
            ),
        }
    except Exception as e:
        logger.error(f"Error creating neuron: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def evaluate_neuron(
    inputs: list[float],
    weights: list[float],
    threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate an MCP neuron with given inputs and return explainable result.

    Args:
        inputs: Input values [x1, x2, ..., xn]
        weights: Neuron weights [bias, w1, w2, ..., wn]
        threshold: Activation threshold

    Returns:
        Detailed evaluation with explanation

    Example:
        # Check if both safety conditions are met
        evaluate_neuron([1, 1], [-1.0, 0.6, 0.6])
        # Returns: {"output": 1, "fired": true, ...}
    """
    try:
        neuron = MCPNeuron(weights, threshold)
        output = neuron.activate(inputs)

        # Compute signal for explanation
        x = np.insert(inputs, 0, 1.0)
        signal = np.dot(x, neuron.weights)

        return {
            "output": int(output),
            "fired": bool(output),
            "inputs": inputs,
            "weights": weights,
            "threshold": threshold,
            "signal": float(signal),
            "explanation": (
                f"Signal = {signal:.2f}, "
                f"Threshold = {threshold:.2f}, "
                f"Neuron {'FIRED' if output else 'did not fire'}"
            ),
        }
    except Exception as e:
        logger.error(f"Error evaluating neuron: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def logic_gate(
    gate_type: str,
    inputs: list[float],
) -> dict[str, Any]:
    """
    Apply a standard logic gate using MCP neurons.

    Args:
        gate_type: Type of gate (AND, OR, NOT, NAND, NOR, XOR)
        inputs: Input values (1-2 inputs depending on gate)

    Returns:
        Gate output with explanation

    Example:
        # Safety check: require both conditions
        logic_gate("AND", [1, 1])  # Returns 1 (safe)
        logic_gate("AND", [1, 0])  # Returns 0 (not safe)
    """
    try:
        gate_type = gate_type.upper()
        gates = LogicGates()

        if gate_type == "NOT":
            if len(inputs) != 1:
                return {
                    "status": "error",
                    "message": "NOT gate requires exactly 1 input",
                }
            output = gates.NOT(inputs[0])

        elif gate_type in ["AND", "OR", "NAND", "NOR", "XOR"]:
            if len(inputs) != 2:
                return {
                    "status": "error",
                    "message": f"{gate_type} gate requires exactly 2 inputs",
                }

            gate_func = getattr(gates, gate_type)
            output = gate_func(inputs[0], inputs[1])

        else:
            return {
                "status": "error",
                "message": f"Unknown gate type: {gate_type}",
            }

        return {
            "gate": gate_type,
            "inputs": inputs,
            "output": int(output),
            "fired": bool(output),
            "explanation": f"{gate_type}({inputs}) = {output}",
        }

    except Exception as e:
        logger.error(f"Error in logic gate: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def create_decision_rule(
    rule_name: str,
    weights: list[float],
    threshold: float = 0.0,
    description: str = "",
) -> dict[str, Any]:
    """
    Create a named decision rule in the network for reuse.

    This is useful for encoding business rules, safety constraints,
    or other explicit decision logic that should be transparent and auditable.

    Args:
        rule_name: Unique name for this rule
        weights: Neuron weights [bias, w1, w2, ..., wn]
        threshold: Activation threshold
        description: Human-readable description of what this rule checks

    Returns:
        Rule creation status

    Example:
        # Safety rule: both conditions must be true
        create_decision_rule(
            "safety_check",
            [-1.0, 0.6, 0.6],
            description="Requires both safety_sensor_1 and safety_sensor_2"
        )
    """
    try:
        decision_network.add_neuron(rule_name, weights, threshold)

        return {
            "status": "success",
            "rule_name": rule_name,
            "weights": weights,
            "threshold": threshold,
            "description": description or "No description provided",
            "usage": f"Use apply_decision_rule('{rule_name}', inputs) to evaluate",
        }
    except Exception as e:
        logger.error(f"Error creating decision rule: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def apply_decision_rule(
    rule_name: str,
    inputs: list[float],
) -> dict[str, Any]:
    """
    Apply a named decision rule with full explanation.

    This provides complete transparency for why a decision was made,
    making it suitable for regulated industries, safety-critical systems,
    or any application requiring explainability.

    Args:
        rule_name: Name of the rule to apply
        inputs: Input values to evaluate

    Returns:
        Decision output with complete explanation trace

    Example:
        apply_decision_rule("safety_check", [1, 1])
        # Returns detailed explanation of why safety check passed
    """
    try:
        inputs_dict = {rule_name: inputs}
        output, explanation = decision_network.evaluate(inputs_dict, rule_name)

        return {
            "status": "success",
            "rule_name": rule_name,
            "output": int(output),
            "decision": "APPROVED" if output else "REJECTED",
            "explanation": explanation,
            "reason": (
                f"Signal ({explanation['signal']:.2f}) "
                f"{'≥' if output else '<'} "
                f"threshold ({explanation['threshold']:.2f})"
            ),
        }
    except Exception as e:
        logger.error(f"Error applying decision rule: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def get_decision_log() -> dict[str, Any]:
    """
    Get the complete decision log for auditability.

    Returns all decisions made by the network with their explanations,
    perfect for compliance, debugging, or understanding system behavior.

    Returns:
        Complete log of all decisions with explanations
    """
    return {
        "status": "success",
        "total_decisions": len(decision_network.decision_log),
        "decisions": decision_network.decision_log,
    }


@mcp.tool()
def clear_decision_log() -> dict[str, Any]:
    """Clear the decision log."""
    count = len(decision_network.decision_log)
    decision_network.decision_log.clear()
    return {
        "status": "success",
        "cleared_decisions": count,
    }


@mcp.tool()
async def post_nn_decision(
    ctx: Context,
    nn_outputs: dict[str, float],
    decision_rule: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Hybrid AI workflow: Use NN outputs with transparent decision rules.

    This implements the "post-NN calculator" pattern where:
    1. A neural network provides perception/analysis (nn_outputs)
    2. Simple, transparent MCP neurons make the final decision
    3. Every decision is fully explainable and auditable

    Args:
        ctx: FastMCP context
        nn_outputs: Dictionary of NN outputs (e.g., {"is_safe": 0.95, "is_valid": 0.88})
        decision_rule: Name of the rule to apply or logic expression
        threshold: Threshold to convert NN probabilities to binary (default: 0.5)

    Returns:
        Final decision with complete explanation chain

    Example:
        # NN says object detection confidence is high
        post_nn_decision(
            {"pedestrian_detected": 0.95, "collision_imminent": 0.92},
            "emergency_brake",
            threshold=0.9
        )
    """
    try:
        await ctx.info("Processing hybrid AI decision...")

        # Convert NN probabilities to binary inputs
        binary_inputs = {
            key: 1 if value >= threshold else 0 for key, value in nn_outputs.items()
        }

        await ctx.info(
            f"Converted NN outputs to binary: {json.dumps(binary_inputs, indent=2)}"
        )

        # Apply decision rule
        input_values = list(binary_inputs.values())

        if decision_rule in decision_network.neurons:
            # Use named rule
            inputs_dict = {decision_rule: input_values}
            output, explanation = decision_network.evaluate(inputs_dict, decision_rule)
            rule_type = "named_rule"
        else:
            # Try as logic gate
            gate_type = decision_rule.upper()
            gates = LogicGates()

            if gate_type == "NOT":
                if len(input_values) < 1:
                    return {
                        "status": "error",
                        "message": "NOT gate requires at least 1 input",
                    }
                output = gates.NOT(input_values[0])
            elif gate_type in ["AND", "OR", "NAND", "NOR", "XOR"]:
                if len(input_values) < 2:
                    return {
                        "status": "error",
                        "message": f"{gate_type} gate requires at least 2 inputs",
                    }
                gate_func = getattr(gates, gate_type)
                output = gate_func(input_values[0], input_values[1])
            else:
                return {
                    "status": "error",
                    "message": f"Unknown gate type: {gate_type}",
                }

            explanation = {
                "gate": gate_type,
                "inputs": input_values[:2] if gate_type != "NOT" else input_values[:1],
                "output": int(output),
                "fired": bool(output),
            }
            rule_type = "logic_gate"

        return {
            "status": "success",
            "nn_outputs": nn_outputs,
            "binary_inputs": binary_inputs,
            "threshold": threshold,
            "decision_rule": decision_rule,
            "rule_type": rule_type,
            "final_decision": int(output),
            "decision": "APPROVED" if output else "REJECTED",
            "explanation": explanation,
            "explainability": (
                "This decision is fully transparent: "
                f"NN provided probabilities, "
                f"converted to binary at threshold {threshold}, "
                f"then evaluated using explicit {rule_type}"
            ),
        }
    except Exception as e:
        logger.error(f"Error in hybrid AI decision: {e}")
        return {"status": "error", "message": str(e)}


# ==================== Resources ====================


@mcp.resource("logic-gates://truth-tables")
def get_truth_tables() -> str:
    """
    Get truth tables for all standard logic gates.

    This resource provides reference information about how each
    logic gate behaves, useful for designing decision rules.
    """
    truth_tables = """
    # Logic Gate Truth Tables

    ## AND Gate
    | A | B | Output |
    |---|---|--------|
    | 0 | 0 |   0    |
    | 0 | 1 |   0    |
    | 1 | 0 |   0    |
    | 1 | 1 |   1    |

    ## OR Gate
    | A | B | Output |
    |---|---|--------|
    | 0 | 0 |   0    |
    | 0 | 1 |   1    |
    | 1 | 0 |   1    |
    | 1 | 1 |   1    |

    ## NOT Gate
    | A | Output |
    |---|--------|
    | 0 |   1    |
    | 1 |   0    |

    ## XOR Gate (Exclusive OR)
    | A | B | Output |
    |---|---|--------|
    | 0 | 0 |   0    |
    | 0 | 1 |   1    |
    | 1 | 0 |   1    |
    | 1 | 1 |   0    |

    ## NAND Gate
    | A | B | Output |
    |---|---|--------|
    | 0 | 0 |   1    |
    | 0 | 1 |   1    |
    | 1 | 0 |   1    |
    | 1 | 1 |   0    |

    ## NOR Gate
    | A | B | Output |
    |---|---|--------|
    | 0 | 0 |   1    |
    | 0 | 1 |   0    |
    | 1 | 0 |   0    |
    | 1 | 1 |   0    |
    """
    return truth_tables.strip()


@mcp.resource("examples://hybrid-ai")
def get_hybrid_ai_examples() -> str:
    """
    Get examples of hybrid AI architectures using MCP neurons.

    This resource demonstrates practical applications of the
    "post-NN calculator" pattern for explainable AI.
    """
    examples = """
    # Hybrid AI Examples

    ## Example 1: Self-Driving Car Safety
    ```
    # NN provides perception
    nn_outputs = {
        "pedestrian_detected": 0.95,
        "red_light": 0.98,
        "collision_risk": 0.92
    }

    # MCP neuron enforces safety rule
    # Rule: BRAKE if (pedestrian_detected OR red_light OR collision_risk)
    safety_rule = OR(OR(pedestrian, red_light), collision)

    Decision: BRAKE (explainable: any safety condition triggered)
    ```

    ## Example 2: Loan Approval
    ```
    # NN analyzes applicant data
    nn_outputs = {
        "credit_worthiness": 0.85,
        "income_stable": 0.92,
        "debt_manageable": 0.78
    }

    # MCP neuron applies lending policy
    # Rule: APPROVE if ALL conditions met (AND gate)
    approval_rule = AND(AND(credit, income), debt)

    Decision: APPROVED (explainable: all criteria met)
    ```

    ## Example 3: Medical Diagnosis Support
    ```
    # NN provides diagnostic confidence
    nn_outputs = {
        "symptom_match": 0.88,
        "risk_factor_present": 0.95,
        "test_result_positive": 0.82
    }

    # MCP neuron requires majority vote
    # Rule: FLAG for review if at least 2 of 3 conditions
    diagnostic_rule = OR(
        AND(symptom, risk),
        OR(AND(symptom, test), AND(risk, test))
    )

    Decision: FLAG for review (explainable: 2/3 criteria met)
    ```

    ## Why This Approach Works

    1. **Explainability**: Every decision traces to explicit rules
    2. **Safety**: Critical constraints are enforced transparently
    3. **Compliance**: Auditable decision logs for regulations
    4. **Trust**: Stakeholders understand why decisions are made
    5. **Flexibility**: Easy to update rules without retraining NNs
    """
    return examples.strip()


# ==================== Cleanup ====================


def cleanup():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Hybrid AI MCP server")
    decision_network.decision_log.clear()


def signal_handler(signum: int, frame: Any):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down gracefully")
    cleanup()
    sys.exit(0)


# Register cleanup handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup)


# ==================== Main ====================

if __name__ == "__main__":
    try:
        logger.info("Starting Hybrid AI MCP server")
        logger.info(
            "Implementing transparent post-NN calculators for explainable AI"
        )
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        cleanup()
