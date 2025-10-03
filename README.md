# Hybrid AI MCP Server 🧠

A Model Context Protocol server that implements **transparent "post-NN calculators"** using McCulloch-Pitts neurons for explainable AI and safety-critical decision making.

## 🎯 The Problem This Solves

Modern neural networks are powerful but opaque "black boxes." You can't easily explain *why* they made a decision, which is problematic for:
- **Safety-critical systems** (self-driving cars, medical diagnosis)
- **Regulated industries** (finance, healthcare, insurance)
- **Trust and accountability** (users want to understand decisions)
- **Debugging and improvement** (hard to fix what you can't understand)

## 💡 The Solution: Hybrid AI

This server implements the **"post-NN calculator"** pattern:

1. **Large Neural Network** (The Perceiver)
   - Handles complex perception and pattern recognition
   - Processes messy, real-world data
   - Outputs high-level features (probabilities, classifications)

2. **MCP Neurons** (The Decider) ← This Server
   - Simple, transparent logical rules
   - Makes final decisions based on NN outputs
   - Fully explainable and auditable
   - Enforces business rules and safety constraints

```
                           🎯 Your Application
                                   ↓
┌────────────────────┐    ┌────────────────────┐
│  Neural Network    │───→│  MCP Neuron        │
│  (Black Box)       │    │  (Transparent)     │
│                    │    │                    │
│  • Image Analysis  │    │  • Business Rules  │
│  • Pattern Match   │    │  • Safety Checks   │
│  • Probabilities   │    │  • Explainable     │
└────────────────────┘    └────────────────────┘
     "I'm 95% sure           "IF safety_check
      it's a cat"            AND valid_input
                             THEN approve"
```

## 🚀 Quick Start

### Installation

```bash
cd /home/ty/Repositories/ai_workspace/hybrid-ai-mcp

# Create virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install dependencies
uv add fastmcp numpy pydantic
```

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hybrid-ai": {
      "command": "uv",
      "args": [
        "--directory",
        "/your-path-to/hybrid-ai-mcp",
        "run",
        "server.py"
      ]
    }
  }
}
```

### Testing

```bash
# Test the server
fastmcp dev server.py
```

## 📚 Core Concepts

### MCP Neuron

A simple binary threshold neuron:

```
output = 1 if (w₀ + x₁·w₁ + x₂·w₂ + ... + xₙ·wₙ) ≥ threshold else 0
```

- **Transparent**: You can see exactly why it fired
- **Auditable**: Every decision is logged with explanation
- **Deterministic**: Same inputs always give same output

### Logic Gates

Built from MCP neurons:

- **AND**: Fires only if ALL inputs are true
- **OR**: Fires if ANY input is true
- **NOT**: Inverts input
- **XOR**: Fires if inputs differ
- **NAND/NOR**: Negated versions

### Decision Networks

Compose multiple MCP neurons to create complex, explainable logic:

```python
# Safety rule: require both conditions
safety_rule = AND(sensor_1, sensor_2)

# Approval rule: require majority (2 of 3)
approval_rule = OR(
    AND(credit_check, income_check),
    OR(
        AND(credit_check, employment_check),
        AND(income_check, employment_check)
    )
)
```

## 🛠️ Available Tools

### Basic Operations

**`create_mcp_neuron(weights, threshold, name)`**
- Create a neuron with specified weights
- Optionally name it for reuse
- Returns configuration and usage instructions

**`evaluate_neuron(inputs, weights, threshold)`**
- Evaluate a neuron with given inputs
- Returns output with full explanation
- Shows signal calculation and firing decision

**`logic_gate(gate_type, inputs)`**
- Apply standard logic gates (AND, OR, NOT, XOR, NAND, NOR)
- Pre-configured for common patterns
- Fully explainable outputs

### Decision Rules

**`create_decision_rule(rule_name, weights, threshold, description)`**
- Create named rules for reuse
- Add human-readable descriptions
- Build a library of business logic

**`apply_decision_rule(rule_name, inputs)`**
- Apply a named rule with full explanation
- Complete audit trail
- Clear pass/fail reasoning

### Hybrid AI Workflow

**`post_nn_decision(nn_outputs, decision_rule, threshold)`**
- Main tool for hybrid AI pattern
- Takes NN probabilities → converts to binary → applies transparent rules
- Returns decision with complete explanation chain

**`get_decision_log()`**
- Retrieve all decisions with explanations
- Perfect for auditing and compliance
- Debug and understand system behavior

## 📖 Usage Examples

### Example 1: Self-Driving Car Safety

```python
# Neural network provides perception
nn_outputs = {
    "pedestrian_detected": 0.95,
    "red_light": 0.98,
    "obstacle_ahead": 0.82
}

# Use hybrid AI decision
post_nn_decision(
    nn_outputs=nn_outputs,
    decision_rule="OR",  # ANY safety concern
    threshold=0.9
)

# Result: BRAKE decision
# Explanation: "red_light (0.98 ≥ 0.9) triggered safety rule"
# ✓ Fully explainable
# ✓ Auditable
# ✓ Deterministic
```

### Example 2: Loan Approval

```python
# Create approval rule
create_decision_rule(
    "loan_approval",
    [-1.0, 0.6, 0.6, 0.6],  # Requires all 3 conditions
    description="Approve if credit, income, and debt criteria all met"
)

# Neural network analyzes application
nn_outputs = {
    "credit_score_good": 0.85,
    "income_stable": 0.92,
    "debt_manageable": 0.78
}

# Apply rule
post_nn_decision(
    nn_outputs=nn_outputs,
    decision_rule="loan_approval",
    threshold=0.7
)

# Result: APPROVED
# Explanation: "All 3 criteria met (credit: 0.85, income: 0.92, debt: 0.78)"
# ✓ Compliant with lending regulations
# ✓ Explainable to applicants
# ✓ Auditable for regulators
```

### Example 3: Medical Diagnosis Support

```python
# Create diagnostic rule (majority vote: 2 of 3)
create_decision_rule(
    "requires_review",
    [-1.0, 0.6, 0.6, 0.6],
    description="Flag for doctor review if 2 of 3 indicators present"
)

# NN analyzes patient data
nn_outputs = {
    "symptom_match": 0.88,
    "risk_factor_present": 0.95,
    "test_result_positive": 0.68
}

# Apply rule
post_nn_decision(
    nn_outputs=nn_outputs,
    decision_rule="requires_review",
    threshold=0.75
)

# Result: FLAG FOR REVIEW
# Explanation: "2 of 3 indicators met threshold (symptom: 0.88, risk: 0.95)"
# ✓ Doctor can see exactly why
# ✓ Patient can understand reasoning
# ✓ Meets medical transparency requirements
```

## 🎓 Why This Approach Works

### 1. Explainability
Every decision traces back to explicit, understandable rules. No more "the AI said so."

### 2. Safety & Reliability
- Enforce hard constraints that NNs might violate
- Deterministic behavior in critical scenarios
- Fail-safe defaults

### 3. Regulatory Compliance
- Complete audit trails
- Explainable decisions for regulators
- Clear accountability

### 4. Trust & Adoption
- Stakeholders understand decisions
- Easy to explain to non-technical users
- Builds confidence in AI systems

### 5. Maintainability
- Update rules without retraining NNs
- Test rules independently
- Clear separation of concerns

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Your Application                │
│  (e.g., Claude, Custom Agent)           │
└─────────────────┬───────────────────────┘
                  │ MCP Protocol
                  ↓
┌─────────────────────────────────────────┐
│      Hybrid AI MCP Server               │
│  ┌─────────────────────────────────┐   │
│  │  MCP Neuron Engine              │   │
│  │  • Binary threshold neurons     │   │
│  │  • Logic gate composition       │   │
│  │  • Decision networks            │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Explainability Layer           │   │
│  │  • Decision logging             │   │
│  │  • Audit trails                 │   │
│  │  • Human-readable explanations  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## 📚 Resources

The server provides reference resources:

- `logic-gates://truth-tables` - Truth tables for all gates
- `examples://hybrid-ai` - Detailed examples and patterns

## 🧪 Testing

```bash
# Run tests (when available)
pytest tests/

# Format code
ruff format .

# Lint
ruff check .
```

## 🤝 Contributing

This is an open demonstration of the hybrid AI concept. Feel free to:
- Add more examples
- Create additional decision patterns
- Improve explanations
- Add visualizations

## 📄 License

MIT License - Use this pattern to build explainable, trustworthy AI systems!

## 🙏 Acknowledgments

Built on the pioneering work of:
- **Warren McCulloch & Walter Pitts** (1943) - Original MCP neuron
- **FastMCP** - Making MCP servers simple and Pythonic
- **Model Context Protocol** - Standardizing AI tool integration

## 🚀 Next Steps

1. **Install and test** the server
2. **Try the examples** with your own data
3. **Create custom decision rules** for your domain
4. **Share your patterns** - help others build explainable AI

---

**Made with 💙 for transparent, trustworthy AI systems**
