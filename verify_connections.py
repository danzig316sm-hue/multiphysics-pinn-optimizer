"""
verify_connections.py
Run this ONCE from your Multiphysics-Pinn-Optimizer root to confirm all 4
missing modules are correctly installed and wired.

    cd M:\Multiphysics-Pinn-Optimizer
    python verify_connections.py
"""
import sys, traceback

PASS = "✓"
FAIL = "✗"

results = {}

# 1. GeometrySpec
try:
    from solvers.base_solver import GeometrySpec
    geo = GeometrySpec()
    assert len(geo.to_vector()) == 36
    t = geo.to_tensor()
    assert t.shape == (1, 36)
    print(f"{PASS} GeometrySpec  — hash={geo.design_hash()}, tensor={t.shape}")
    results["GeometrySpec"] = True
except Exception as e:
    print(f"{FAIL} GeometrySpec  — {e}")
    results["GeometrySpec"] = False

# 2. FEAToolSolver
try:
    from solvers.featool_solver import FEAToolSolver
    solver = FEAToolSolver(verbose=False)
    res = solver.run_all(geo)
    assert "electromagnetic" in res or "aerodynamic" in res
    print(f"{PASS} FEAToolSolver — solver={res.get('solver')}, all_passed={res.get('all_passed')}")
    results["FEAToolSolver"] = True
except Exception as e:
    print(f"{FAIL} FEAToolSolver — {e}")
    traceback.print_exc()
    results["FEAToolSolver"] = False

# 3. SolidWorksVerification
try:
    from solvers.sw_verification import SolidWorksVerification
    sw = SolidWorksVerification(watch_folder="sw_verification_test", verbose=False)
    dummy_results = {
        "electromagnetic": {"efficiency_pct": 95.5, "cogging_Nm": 18.0, "Br_min_T": 0.65, "passed": True},
        "thermal":         {"T_winding_C": 142.0, "T_magnet_C": 48.0, "passed": True},
        "structural":      {"safety_factor": 2.1, "passed": True},
    }
    h = sw.flag_for_verification(geo, dummy_results, priority="low", notes="verify_connections test")
    readiness = sw.check_cutover_readiness()
    print(f"{PASS} SWVerification — hash={h}, pending={readiness['pending_count']}")
    results["SolidWorksVerification"] = True
except Exception as e:
    print(f"{FAIL} SWVerification — {e}")
    traceback.print_exc()
    results["SolidWorksVerification"] = False

# 4. TurboQuantPINN
try:
    import torch
    from utils.turboquant_wrapper import TurboQuantPINN, MultiPhysicsLoss, compression_report
    model = TurboQuantPINN(input_dim=36, bits=4, device="cpu")
    x = geo.to_tensor()
    preds = model(x)
    assert set(preds.keys()) == {"em", "thermal", "structural"}
    loss_fn = MultiPhysicsLoss(physics_weight=0.1)
    targets = {
        "em":         torch.zeros(1, 4),
        "thermal":    torch.zeros(1, 2),
        "structural": torch.zeros(1, 1),
    }
    loss, breakdown = loss_fn(preds, targets, x)
    print(f"{PASS} TurboQuantPINN — {model.quant_info()['mode']}, loss={loss.item():.4f}")
    results["TurboQuantPINN"] = True
except Exception as e:
    print(f"{FAIL} TurboQuantPINN — {e}")
    traceback.print_exc()
    results["TurboQuantPINN"] = False

# 5. Master pipeline import check
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline", "master_multiphysics_pipeline.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"{PASS} master_multiphysics_pipeline.py — imports OK")
    results["MasterPipeline"] = True
except Exception as e:
    print(f"{FAIL} master_multiphysics_pipeline.py — {e}")
    results["MasterPipeline"] = False

print(f"\n{'='*50}")
passed = sum(results.values())
total  = len(results)
print(f"Result: {passed}/{total} components connected")
if passed == total:
    print("All systems GO — run python master_multiphysics_pipeline.py")
else:
    failed = [k for k, v in results.items() if not v]
    print(f"Still failing: {failed}")
