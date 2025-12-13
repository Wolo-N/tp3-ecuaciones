import numpy as np

import fin_geometry as geom
from fin_2d_forward_euler import solve_transient

# ---------------------------------------------------------------------------
# Helpers to evaluate one fin geometry
# ---------------------------------------------------------------------------
def simulate_single_fin(
    fin_thickness_base,
    fin_height,
    fin_length=0.05,
    fin_thickness_tip=None,
    *,
    k=205.0,
    h=300.0,
    rho=2700.0,
    cp=900.0,
    T_inf=45.0,
    T_base=90.0,
    dt=1e-2,
    t_final=5.0,
    tol=1e-6,
):
    """
    Rebuild the grid for a given fin geometry, run the transient solver, and
    return temperature field, volume, and heat dissipated by convection for a
    single fin.
    """
    geom.rebuild_grid_for_geometry(
        new_fin_thickness=fin_thickness_base,
        new_fin_height=fin_height,
        new_fin_length=fin_length,
        new_fin_thickness_tip=fin_thickness_tip,
    )

    T, coords, center_nodes = solve_transient(
        k=k,
        h=h,
        rho=rho,
        cp=cp,
        T_inf=T_inf,
        T_base=T_base,
        dt=dt,
        t_final=t_final,
        tol=tol,
    )

    # Convection from exposed faces
    q_conv = 0.0
    for c in center_nodes:
        A = geom.boundary_areas[c]
        q_conv += h * A * (T[c] - T_inf)

    fin_volume = float(np.sum(geom.volumes))

    return {
        "T": T,
        "coords": coords,
        "center_nodes": center_nodes,
        "q_conv": q_conv,
        "fin_volume": fin_volume,
    }


# ---------------------------------------------------------------------------
# Layout and packing on the 50mm x 50mm base
# ---------------------------------------------------------------------------
def fins_per_base(fin_thickness_base, fin_gap, base_width=0.05):
    """
    Compute how many fins fit across the base width given a gap for airflow.
    """
    pitch = fin_thickness_base + fin_gap
    if pitch <= 0.0:
        raise ValueError("Pitch must be positive")

    n_fins = max(1, int(np.floor((base_width + fin_gap) / pitch)))
    used_width = n_fins * pitch - fin_gap  # remove trailing gap
    return n_fins, used_width


# ---------------------------------------------------------------------------
# Evaluate one design against requirements
# ---------------------------------------------------------------------------
def evaluate_design(
    fin_thickness_base,
    fin_height,
    fin_thickness_tip=None,
    *,
    fin_gap=0.0015,
    fin_length=0.05,
    base_width=0.05,
    base_length=0.05,
    base_thickness=0.005,
    required_heat=500.0,
    height_limit=0.025,
    k=205.0,
    h=300.0,
    rho=2700.0,
    cp=900.0,
    T_inf=45.0,
    T_base=90.0,
    dt=1e-2,
    t_final=5.0,
    tol=1e-6,
):
    sim = simulate_single_fin(
        fin_thickness_base=fin_thickness_base,
        fin_height=fin_height,
        fin_length=fin_length,
        fin_thickness_tip=fin_thickness_tip,
        k=k,
        h=h,
        rho=rho,
        cp=cp,
        T_inf=T_inf,
        T_base=T_base,
        dt=dt,
        t_final=t_final,
        tol=tol,
    )

    n_fins, used_width = fins_per_base(fin_thickness_base, fin_gap, base_width)

    fin_mass = sim["fin_volume"] * rho
    base_mass = base_width * base_length * base_thickness * rho
    total_mass = base_mass + n_fins * fin_mass

    q_total = sim["q_conv"] * n_fins

    return {
        "design": {
            "fin_thickness_base": fin_thickness_base,
            "fin_thickness_tip": fin_thickness_tip,
            "fin_height": fin_height,
            "fin_gap": fin_gap,
            "fin_length": fin_length,
            "base_width": base_width,
            "base_length": base_length,
            "base_thickness": base_thickness,
        },
        "n_fins": n_fins,
        "used_width": used_width,
        "fin_volume": sim["fin_volume"],
        "fin_mass": fin_mass,
        "base_mass": base_mass,
        "total_mass": total_mass,
        "q_conv_single": sim["q_conv"],
        "q_conv_total": q_total,
        "meets_power": q_total >= required_heat,
        "meets_height": fin_height <= height_limit,
        "T_field": sim["T"],
        "coords": sim["coords"],
        "center_nodes": sim["center_nodes"],
    }


# ---------------------------------------------------------------------------
# Simple grid search optimizer
# ---------------------------------------------------------------------------
def grid_search_designs(
    base_thicknesses,
    fin_thicknesses,
    fin_heights,
    fin_gaps,
    fin_tip_thicknesses=None,
    *,
    required_heat=500.0,
    height_limit=0.025,
    fin_length=0.05,
    k=205.0,
    h=300.0,
    rho=2700.0,
    cp=900.0,
    T_inf=45.0,
    T_base=90.0,
    dt=1e-2,
    t_final=5.0,
    tol=1e-6,
):
    results = []
    best = None

    tip_list = fin_tip_thicknesses if fin_tip_thicknesses is not None else [None]

    for base_t in base_thicknesses:
        for gap in fin_gaps:
            for th_base in fin_thicknesses:
                for height in fin_heights:
                    for th_tip in tip_list:
                        res = evaluate_design(
                            fin_thickness_base=th_base,
                            fin_height=height,
                            fin_thickness_tip=th_tip,
                            fin_gap=gap,
                            fin_length=fin_length,
                            base_thickness=base_t,
                            required_heat=required_heat,
                            height_limit=height_limit,
                            k=k,
                            h=h,
                            rho=rho,
                            cp=cp,
                            T_inf=T_inf,
                            T_base=T_base,
                            dt=dt,
                            t_final=t_final,
                            tol=tol,
                        )
                        results.append(res)

                        if res["meets_power"] and res["meets_height"]:
                            if best is None or res["total_mass"] < best["total_mass"]:
                                best = res

    # Sort results: feasible first, then by total mass
    results.sort(
        key=lambda r: (
            not (r["meets_power"] and r["meets_height"]),
            r["total_mass"],
        )
    )

    return best, results


if __name__ == "__main__":
    # Example search spanning rectangular and tapered fins
    base_thicknesses = [0.004, 0.005]  # 4-6 mm base
    fin_thicknesses = [0.0015, 0.0020]  # 1.5-3.0 mm
    fin_tip_thicknesses = [None, 0.0010]  # None = rectangular
    fin_heights = [0.025]  # up to 25 mm limit
    fin_gaps = [0.0010]  # airflow spacing

    best, results = grid_search_designs(
        base_thicknesses=base_thicknesses,
        fin_thicknesses=fin_thicknesses,
        fin_heights=fin_heights,
        fin_gaps=fin_gaps,
        fin_tip_thicknesses=fin_tip_thicknesses,
    )

    print(f"Checked {len(results)} designs")
    if best:
        print("Lightest feasible design:")
        print(best)
    else:
        print("No design met the 500 W target within constraints.")
