import numpy as np
import matplotlib.pyplot as plt
import time
from fin_2d_forward_euler import evaluate_fin_design

# Grid search over Q9 shape parameters for ONE fin
# evaluate_fin_design will determine how many are needed for 500W
thicknesses = np.linspace(0.0015, 0.004, 5)   # 1.5-4 mm
heights = np.linspace(0.018, 0.025, 5)        # 18-25 mm (TP3: max 25mm)
tip_ratios = np.linspace(0.7, 1.0, 3)         # 0.7=tapered, 1.0=rectangular
round_factors = np.linspace(0.2, 0.8, 3)      # curvature

TIME_LIMIT = 30  # seconds total

best_mass = np.inf
best_params = None

results = []
start_time = time.time()
timeout = False

for t in thicknesses:
    if timeout: break
    for h in heights:
        if timeout: break
        for tr in tip_ratios:
            if timeout: break
            for rf in round_factors:

                # Check time limit
                if time.time() - start_time > TIME_LIMIT:
                    print(f"\nTime limit reached ({TIME_LIMIT}s)")
                    timeout = True
                    break

                result = evaluate_fin_design(
                    fin_thickness=t,
                    fin_height=h,
                    tip_ratio=tr,
                    round_factor=rf
                )

                if result and result['feasible']:
                    results.append({
                        't': t, 'h': h, 'tr': tr, 'rf': rf,
                        'mass': result['m_total'],
                        'Q_fin': result['Q_fin'],
                        'n_req': result['n_required']
                    })

                    if result['m_total'] < best_mass:
                        best_mass = result['m_total']
                        best_params = (t, h, tr, rf)
                        print(f"New best: t={t*1e3:.2f}mm h={h*1e3:.2f}mm "
                              f"tip={tr:.2f} round={rf:.2f} → {best_mass*1e3:.1f}g")

if best_params:
    t, h, tr, rf = best_params
    print(f"\nOptimal design:")
    print(f"  thickness = {t*1e3:.2f} mm")
    print(f"  height = {h*1e3:.2f} mm")
    print(f"  tip_ratio = {tr:.2f}")
    print(f"  round_factor = {rf:.2f}")
    print(f"  total mass = {best_mass*1e3:.1f} g")

    # Plot results
    masses = [r['mass']*1e3 for r in results]
    Qs = [r['Q_fin'] for r in results]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter([r['h']*1e3 for r in results], masses, c=Qs, cmap='viridis')
    plt.colorbar(label='Q per fin [W]')
    plt.xlabel('Height [mm]')
    plt.ylabel('Total mass [g]')
    plt.axvline(h*1e3, color='r', linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.scatter([r['t']*1e3 for r in results], masses, c=Qs, cmap='viridis')
    plt.colorbar(label='Q per fin [W]')
    plt.xlabel('Thickness [mm]')
    plt.ylabel('Total mass [g]')
    plt.axvline(t*1e3, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
