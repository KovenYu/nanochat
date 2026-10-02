"""Challenge (2): synthetic Orrery trajectory -> View A / View B SFT exports.

Mapping decisions (Koven, 2026-09-18):
  1. View A discards the inner loop entirely; View B renders verifiers as TOOL calls
     (python / python_output parts) so the model learns user-preference vs tool-feedback
     are different channels: tools can be re-queried until green, users cannot.
  2. Failed attempts inside the last turn (v2) ARE supervised: strong-teacher trajectories,
     v2 teaches "respond to user feedback", v2b teaches "respond to verifier feedback".
  3. View A: intent + constraints in one user turn.
  4. One View B sample per trajectory (merged supervision == sliding samples, cheaper).
CC fill-ins: passing verdicts are included too (in-distribution rationale); the trailing
user ACCEPT is not exported.
"""

TRAJ = {
    "intent": "Design a bracket mounting a 40mm fan to a 2020 rail.",
    "constraints": "hole spacing 32mm; thickness >= 3mm",
    "outer": [
        {"attempts": [
            {"code": "bracket = box(50, 50, 5)", "verdict": "compile PASS; collision PASS; vlm PASS"}],
         "present": "Here is a solid bracket.",
         "user_feedback": "Too bulky, cut the weight."},
        {"attempts": [
            {"code": "bracket = bax(40, 40, 3)", "verdict": "compile FAIL: name 'bax' is not defined"},
            {"code": "bracket = box(40, 40, 3)", "verdict": "compile PASS; collision PASS; vlm PASS"}],
         "present": "Lighter bracket, 3mm thick.",
         "user_feedback": "ACCEPT"}],
}


def _user_turn(traj):
    return {"role": "user", "content": traj["intent"] + "\n" + traj["constraints"]}


def export_view_a(traj):
    """(intent + constraints) -> final accepted code. Both loops collapsed."""
    final_code = traj["outer"][-1]["attempts"][-1]["code"]
    return {"messages": [
        _user_turn(traj),
        {"role": "assistant", "content": final_code},
    ]}


def export_view_b(traj):
    """Multi-turn revision sample. Verifiers are tool parts inside each assistant turn;
    loss placement is delegated to mask_history (last turn: attempts supervised,
    verdicts not)."""
    messages = [_user_turn(traj)]
    for i, outer in enumerate(traj["outer"]):
        parts = []
        for att in outer["attempts"]:
            parts.append({"type": "python", "text": att["code"]})
            parts.append({"type": "python_output", "text": att["verdict"]})
        parts.append({"type": "text", "text": outer["present"]})
        messages.append({"role": "assistant", "content": parts})
        if i < len(traj["outer"]) - 1:            # trailing ACCEPT is not exported
            messages.append({"role": "user", "content": outer["user_feedback"]})
    return {"messages": messages}
