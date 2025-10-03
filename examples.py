"""
Practical Examples: Hybrid AI with MCP Neurons
===============================================

This file demonstrates real-world applications of the hybrid AI pattern
using transparent MCP neurons as "post-NN calculators."

Run these examples after starting the MCP server to see how
explainable AI works in practice.
"""

# Example 1: Autonomous Vehicle Safety System
# ============================================
"""
Scenario: Self-driving car must make a brake/no-brake decision

Problem with pure NN:
- NN might be 85% confident about pedestrian
- Do we brake? What if it's wrong?
- Can't explain decision to regulators/users

Solution with Hybrid AI:
- NN provides perception (is there a pedestrian? confidence level)
- MCP neuron enforces safety rule (brake if ANY risk detected)
- Decision is transparent and explainable
"""

autonomous_vehicle_example = {
    "tool": "post_nn_decision",
    "args": {
        "nn_outputs": {
            "pedestrian_detected": 0.95,
            "red_light": 0.98,
            "collision_risk": 0.82,
            "road_clear": 0.15
        },
        "decision_rule": "OR",  # Brake if ANY danger detected
        "threshold": 0.8
    },
    "expected_output": {
        "final_decision": 1,
        "decision": "BRAKE",
        "explainability": (
            "NN detected multiple risks above threshold: "
            "pedestrian (0.95), red_light (0.98), collision_risk (0.82). "
            "Safety rule: brake if ANY risk > 0.8. DECISION: BRAKE"
        )
    }
}

# Example 2: Credit Card Fraud Detection
# =======================================
"""
Scenario: Decide whether to flag transaction as fraudulent

Problem with pure NN:
- NN might be 60% confident it's fraud
- Block the card? False positives anger customers
- No clear explanation for why card was blocked

Solution with Hybrid AI:
- NN analyzes transaction patterns
- MCP neuron applies business rules (e.g., require 2 of 3 risk factors)
- Customer service can explain exact reason
"""

fraud_detection_example = {
    "steps": [
        {
            "step": 1,
            "tool": "create_decision_rule",
            "args": {
                "rule_name": "fraud_flag",
                "weights": [-1.0, 0.6, 0.6, 0.6],  # Requires 2 of 3
                "threshold": 0.0,
                "description": "Flag if at least 2 fraud indicators present"
            }
        },
        {
            "step": 2,
            "tool": "post_nn_decision",
            "args": {
                "nn_outputs": {
                    "unusual_location": 0.88,
                    "suspicious_amount": 0.92,
                    "vendor_mismatch": 0.45
                },
                "decision_rule": "fraud_flag",
                "threshold": 0.7
            }
        }
    ],
    "expected_output": {
        "final_decision": 1,
        "decision": "FLAG_FOR_REVIEW",
        "explainability": (
            "2 of 3 fraud indicators exceeded threshold: "
            "unusual_location (0.88) and suspicious_amount (0.92). "
            "Customer service explanation: 'We noticed this purchase was "
            "made in an unusual location with an unusual amount.'"
        )
    }
}

# Example 3: Medical Diagnosis Support
# ====================================
"""
Scenario: AI assists doctor in diagnosis decision

Problem with pure NN:
- NN suggests diagnosis with 75% confidence
- Doctor needs to understand WHY
- Medical liability requires explainability

Solution with Hybrid AI:
- NN analyzes symptoms, test results, patient history
- MCP neuron applies medical guidelines (e.g., positive if 2 of 3 criteria)
- Doctor sees exact reasoning chain
"""

medical_diagnosis_example = {
    "steps": [
        {
            "step": 1,
            "tool": "create_decision_rule",
            "args": {
                "rule_name": "diagnosis_positive",
                "weights": [-1.0, 0.6, 0.6, 0.6],
                "threshold": 0.0,
                "description": "Positive diagnosis if 2+ of 3 clinical criteria met"
            }
        },
        {
            "step": 2,
            "tool": "post_nn_decision",
            "args": {
                "nn_outputs": {
                    "symptom_correlation": 0.85,
                    "lab_result_abnormal": 0.92,
                    "imaging_findings": 0.68
                },
                "decision_rule": "diagnosis_positive",
                "threshold": 0.75
            }
        }
    ],
    "expected_output": {
        "final_decision": 1,
        "decision": "POSITIVE_DIAGNOSIS_LIKELY",
        "explainability": (
            "2 of 3 clinical criteria met threshold: "
            "symptom_correlation (0.85) and lab_result_abnormal (0.92). "
            "Recommend: Consult specialist for confirmation."
        )
    }
}

# Example 4: Loan Approval System
# ==============================
"""
Scenario: Bank decides whether to approve loan

Problem with pure NN:
- NN predicts loan risk but can't explain
- Regulators require clear lending criteria
- Applicants have right to know why denied

Solution with Hybrid AI:
- NN assesses creditworthiness from complex data
- MCP neuron enforces explicit lending policy
- Decision meets regulatory requirements
"""

loan_approval_example = {
    "steps": [
        {
            "step": 1,
            "tool": "create_decision_rule",
            "args": {
                "rule_name": "loan_approval",
                "weights": [-1.0, 0.6, 0.6, 0.6],  # ALL must be true
                "threshold": 0.0,
                "description": "Approve if ALL criteria met: credit, income, debt-to-income"
            }
        },
        {
            "step": 2,
            "tool": "post_nn_decision",
            "args": {
                "nn_outputs": {
                    "credit_score_adequate": 0.88,
                    "income_sufficient": 0.92,
                    "debt_to_income_ok": 0.82
                },
                "decision_rule": "loan_approval",
                "threshold": 0.75
            }
        }
    ],
    "expected_output": {
        "final_decision": 1,
        "decision": "APPROVED",
        "explainability": (
            "All lending criteria met: "
            "credit_score_adequate (0.88), "
            "income_sufficient (0.92), "
            "debt_to_income_ok (0.82). "
            "Approved per bank policy."
        )
    }
}

# Example 5: Quality Control in Manufacturing
# ==========================================
"""
Scenario: Automated inspection system for product defects

Problem with pure NN:
- NN detects possible defects with varying confidence
- Can't afford false rejects (waste money)
- Can't afford false accepts (customer complaints)

Solution with Hybrid AI:
- NN analyzes images for defects
- MCP neuron applies quality control rules
- Clear audit trail for quality assurance
"""

quality_control_example = {
    "tool": "post_nn_decision",
    "args": {
        "nn_outputs": {
            "surface_defect": 0.45,
            "dimension_error": 0.88,
            "color_mismatch": 0.35
        },
        "decision_rule": "OR",  # Reject if ANY defect above threshold
        "threshold": 0.7
    },
    "expected_output": {
        "final_decision": 1,
        "decision": "REJECT",
        "explainability": (
            "Product failed quality check: "
            "dimension_error (0.88) exceeded threshold (0.7). "
            "Reason code: DIMENSION_OUT_OF_SPEC"
        )
    }
}

# Example 6: Content Moderation
# ============================
"""
Scenario: Social media platform moderates content

Problem with pure NN:
- NN flags content as potentially harmful
- Users demand to know why content removed
- Platform needs defensible policies

Solution with Hybrid AI:
- NN analyzes text, images, context
- MCP neuron applies community guidelines
- Users see clear explanation
"""

content_moderation_example = {
    "steps": [
        {
            "step": 1,
            "tool": "create_decision_rule",
            "args": {
                "rule_name": "content_violation",
                "weights": [-0.5, 1.0, 1.0, 1.0],  # Remove if ANY violation
                "threshold": 0.0,
                "description": "Remove if any guideline violation detected"
            }
        },
        {
            "step": 2,
            "tool": "post_nn_decision",
            "args": {
                "nn_outputs": {
                    "hate_speech": 0.35,
                    "violence": 0.88,
                    "spam": 0.15
                },
                "decision_rule": "content_violation",
                "threshold": 0.7
            }
        }
    ],
    "expected_output": {
        "final_decision": 1,
        "decision": "REMOVE_CONTENT",
        "explainability": (
            "Content removed for community guideline violation: "
            "violence (0.88) exceeded threshold (0.7). "
            "This violates our policy against violent content."
        )
    }
}

# Example 7: Smart Home Security
# =============================
"""
Scenario: Security system decides whether to sound alarm

Problem with pure NN:
- NN detects motion, analyzes video
- False alarms annoy homeowners
- Missed threats are dangerous

Solution with Hybrid AI:
- NN provides perception (motion, person detected, known face?)
- MCP neuron applies security rules
- Homeowner understands why alarm triggered
"""

security_system_example = {
    "steps": [
        {
            "step": 1,
            "tool": "create_decision_rule",
            "args": {
                "rule_name": "trigger_alarm",
                "weights": [-1.0, 1.0, -1.0],  # Unknown person AND motion
                "threshold": 0.0,
                "description": "Alarm if motion AND unknown person"
            }
        },
        {
            "step": 2,
            "tool": "post_nn_decision",
            "args": {
                "nn_outputs": {
                    "motion_detected": 0.95,
                    "known_face": 0.15
                },
                "decision_rule": "AND",
                "threshold": 0.8
            }
        }
    ],
    "expected_output": {
        "final_decision": 1,
        "decision": "SOUND_ALARM",
        "explainability": (
            "Alarm triggered: motion detected (0.95) AND "
            "face not recognized (known_face: 0.15). "
            "Location: Front door. Time: 2:35 AM"
        )
    }
}

# How to Use These Examples
# ========================
"""
1. Start the MCP server:
   ```
   cd /home/ty/Repositories/ai_workspace/hybrid-ai-mcp
   fastmcp dev server.py
   ```

2. In Claude or your MCP client, use the tools with the example parameters

3. Observe the fully explainable outputs

4. Modify the thresholds and rules to see how decisions change

5. Create your own rules for your domain!

Key Insights:
- NN provides powerful perception but lacks explainability
- MCP neurons add transparent decision logic
- Every decision has a clear audit trail
- Easy to update rules without retraining NNs
- Perfect for regulated industries and safety-critical systems
"""

# Testing Template
# ===============
def test_example(example_name: str, example: dict):
    """
    Template for testing examples.

    Args:
        example_name: Name of the example
        example: The example dictionary
    """
    print(f"\n{'='*60}")
    print(f"Testing: {example_name}")
    print('='*60)

    if "steps" in example:
        for step in example["steps"]:
            print(f"\nStep {step['step']}: {step['tool']}")
            print(f"Args: {step['args']}")
    else:
        print(f"\nTool: {example['tool']}")
        print(f"Args: {example['args']}")

    print("\nExpected Output:")
    print(f"Decision: {example['expected_output']['decision']}")
    print(f"Explanation: {example['expected_output']['explainability']}")

if __name__ == "__main__":
    examples = {
        "Autonomous Vehicle Safety": autonomous_vehicle_example,
        "Credit Card Fraud Detection": fraud_detection_example,
        "Medical Diagnosis Support": medical_diagnosis_example,
        "Loan Approval System": loan_approval_example,
        "Quality Control": quality_control_example,
        "Content Moderation": content_moderation_example,
        "Smart Home Security": security_system_example,
    }

    print("="*60)
    print("HYBRID AI PRACTICAL EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate the 'post-NN calculator' pattern")
    print("where simple, transparent MCP neurons make final decisions")
    print("based on neural network outputs.")
    print("\n" + "="*60)

    for name, example in examples.items():
        test_example(name, example)
