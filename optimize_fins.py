# optimize_fins.py

from fin_2d_forward_euler import evaluate_fin_design

def optimize_fins():
    """
    Optimiza el diseño barriendo:
    - thickness
    - height
    - tip_ratio (punta más o menos angosta)
    - round_factor (lados más o menos redondeados)
    """

    best = None

    # POCOS valores para que corra rápido
    thickness_values = [0.0015, 0.0020]      # 1.5 mm y 2.0 mm
    height_values    = [0.020, 0.025]        # 20 mm y 25 mm

    # Forma: parámetro 1 → angostura de la punta
    tip_ratios       = [1.0, 0.7, 0.5]       # 1.0 rectangular, 0.7 y 0.5 más angosta

    # Forma: parámetro 2 → redondez
    round_factors    = [0.0, 0.4, 0.8]       # 0 = lados rectos, 0.8 = bien redondeada

    for t in thickness_values:
        for h in height_values:
            for tr in tip_ratios:
                for rf in round_factors:

                    print(f"\nProbando diseño:")
                    print(f"  thickness   = {t*1000:.2f} mm")
                    print(f"  height      = {h*1000:.1f} mm")
                    print(f"  tip_ratio   = {tr:.2f}")
                    print(f"  round_fact  = {rf:.2f}")

                    res = evaluate_fin_design(
                        fin_thickness=t,
                        fin_height=h,
                        fin_length=0.05,
                        gap=0.001,
                        tip_ratio=tr,
                        round_factor=rf
                    )

                    if (res is None) or (not res["feasible"]):
                        print("  → no factible")
                        continue

                    print(f"  → factible, masa total = {res['m_total']:.4f} kg")

                    if (best is None) or (res["m_total"] < best["m_total"]):
                        best = res
                        print("  *** nuevo mejor diseño ***")

    return best


if __name__ == "__main__":
    best = optimize_fins()

    print("\n========================")
    print("       MEJOR DISEÑO     ")
    print("========================")

    if best is None:
        print("No se encontró ningún diseño factible.")
    else:
        print(f"Espesor (t):       {best['thickness']*1000:.2f} mm")
        print(f"Altura (h):        {best['height']*1000:.1f} mm")
        print(f"tip_ratio:         {best['tip_ratio']:.2f}")
        print(f"round_factor:      {best['round_factor']:.2f}")
        print(f"Q por aleta:       {best['Q_fin']:.3f} W")
        print(f"Masa por aleta:    {best['m_fin']:.6f} kg")
        print(f"Aletas requeridas: {best['n_required']:.1f}")
        print(f"Aletas máximas:    {best['n_max']}")
        print(f"Masa total aletas: {best['m_total']:.4f} kg")
